import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  # Prevent OpenMP thread conflicts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from collections import deque

import numpy as np
import open3d as o3d
import pandas as pd
import torch
import yaml
from scipy import sparse

from L3ROcc.utils import (
    convert_pointcloud_world_to_camera,
    create_mesh_from_map,
    estimate_intrinsics,
    homogenize_points,
    interpolate_extrinsics,
    load_images_as_tensor,
    pcd_to_voxels,
    plot_camera_poses,
    point_transform_2d_batch,
    preprocess,
    voxel2points,
    voxels_to_pcd,
)

# NOTE: Pi3 / Pi3X are imported lazily inside _load_pretrained_model (see there).
# Importing the model modules (dinov2 / flash-attention CUDA init) at top level breaks
# safetensors' mmap load on Windows; the lazy import lets us pre-load weights to CPU first.
from third_party.pi3.pi3.utils.basic import (  # Assuming you have a helper function
    write_ply,
)

# from pi3.utils.geometry import homogenize_points
from third_party.pi3.pi3.utils.geometry import depth_edge


class DataGenerator:
    """
    Base class for data generation pipelines.
    Handles 3D reconstruction, occupancy grid generation, ray casting for visibility,
    and sequence data serialization.
    """

    def __init__(
        self,
        config_path="./L3ROcc/configs/config.yaml",
        save_dir="./outputs",
        model_dir="./ckpt",
        model_type="pi3x",
    ):
        """
        Initialize the DataGenerator.

        Args:
            config_path (str): Path to the configuration YAML file.
            save_dir (str): Directory where output files will be saved.
            model_dir (str): Root directory containing ckpt sub-folders:
                             model_dir/pi3x  -- Pi3X weights (accepts optional depth/intrinsic
                                                conditioning at the model layer)
                             model_dir/pi3   -- base Pi3 weights (RGB-only forward)
            model_type (str): Which backbone to load. One of {"pi3", "pi3x"}.
                              The checkpoint sub-directory is ``model_dir/<model_type>``.
                              Pi3X.forward consumes depth/intrinsic kwargs when supplied;
                              Pi3.forward only consumes imgs (any K/depth provided by the
                              caller is still honored by post-processing — e.g. calibrated K
                              still overrides the DLT-estimated K in the saved Parquet).
        """
        if model_type not in {"pi3", "pi3x"}:
            raise ValueError(f"model_type must be 'pi3' or 'pi3x', got {model_type!r}")

        self.config_path = config_path
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Autocast dtype for model inference. Selected once: bfloat16 forces the model's
        # FlashAttention SDPA path, which is not built into every torch distribution.
        self.amp_dtype = self._select_amp_dtype()
        self.model_type = model_type
        ckpt_path = os.path.join(model_dir, model_type)
        self.model = (
            self._load_pretrained_model(ckpt_path, model_type)
            .to(self.device)
            .eval()
        )
        print(
            f"Loaded {'Pi3X (multimodal)' if use_multimodal else 'Pi3 (RGB-only)'} from {ckpt_path}"
        )

        self.free_label = 0
        self.pcd = None
        self.camera_intric = np.array(
            [[168.0498, 0.0, 240.0], [0.0, 192.79999, 135.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )  # Intrinsic matrix
        self.camera_intric_rs = None
        self.camera_pose = None  # Extrinsic matrix
        self.camera_trace = None
        self.norm_cam_ray = (
            None  # Normalized camera ray directions (default in camera coordinate)
        )

        with open(config_path, "r", encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)

        self.fps = self.config["fps"]
        self.voxel_size = self.config["voxel_size"]
        self.pc_range = self.config["pc_range"]
        self.occ_size = self.config["occ_size"]
        self.ray_cast_step_size = self.config["ray_cast_step_size"]
        self.interval = self.config["interval"]
        self.voxel_size_scale = self.config["voxel_size_scale"]
        self.history_len = self.config["history_len"]
        self.history_step = self.config["history_step"]
        self.occ_history_buffer = deque(
            maxlen=self.history_len
        )  # Fixed-length queue for sliding window
        self.save_path = self.save_dir

    def _load_pretrained_model(self, ckpt_path, model_type):
        """Load Pi3X / Pi3 weights via the standard ``from_pretrained`` (returns CPU model;
        caller moves it to device).

        The model class is imported lazily here (not at module top) for two reasons:
        (1) keep import side effects out of ``import L3ROcc.base``; and (2) on Windows,
        ``from_pretrained`` segfaults unless an early importer has pre-loaded the safetensors
        weights to CPU *before* the heavy imports (open3d / the model module) and monkeypatched
        ``from_pretrained`` -- see ``tools/exp_scale/exp_scale_compare.py``. Because the import
        is lazy, that monkeypatch (done before this module's deps are constructed) takes effect.
        On Linux/server this is just the native fast path.
        """
        if model_type == "pi3x":
            from third_party.pi3.pi3.models.pi3x import Pi3X

            return Pi3X.from_pretrained(ckpt_path)
        from third_party.pi3.pi3.models.pi3 import Pi3

        return Pi3.from_pretrained(ckpt_path)

    def _select_amp_dtype(self):
        """Choose the autocast dtype for model inference.

        The Pi3 / Pi3X attention layers force the FlashAttention SDPA backend whenever the
        tensors are bfloat16 (see third_party/pi3 .../layers/attention.py). That backend is
        not compiled into every PyTorch build (notably the Windows CUDA wheels), which
        raises ``RuntimeError: No available kernel``. We therefore use bfloat16 only when
        the Flash kernel is actually runnable; otherwise float16, which the same layers
        route through the mem-efficient / math SDPA kernels.
        """
        if self.device != "cuda":
            return torch.float16
        try:
            if torch.cuda.get_device_capability()[0] < 8:
                return torch.float16
            import torch.nn.functional as F
            from torch.nn.attention import SDPBackend, sdpa_kernel

            probe = torch.zeros(1, 1, 8, 16, device="cuda", dtype=torch.bfloat16)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                F.scaled_dot_product_attention(probe, probe, probe)
            return torch.bfloat16
        except Exception:
            print(
                "[Info] FlashAttention SDPA kernel unavailable; using float16 for inference."
            )
            return torch.float16

    def pcd_reconstruction(
        self, input_path, condit_depth_path=None, intrinsics_np=None
    ):
        """
        Reconstructs the 3D point cloud and camera trajectory from video frames using the Pi3 model.

        Args:
            input_path (str): Path to the input video file.
            condit_depth_path (str): Path to the conditional depth map (used by Pi3X only).
            intrinsics_np (np.ndarray): 3x3 intrinsic matrix at the ORIGINAL video resolution.
                - When model_type='pi3x', it is rescaled to model input size and fed to Pi3X.
                - In both modes, the rescaled K replaces the DLT-estimated K in ``camera_intric_rs``
                  so the downstream Parquet stores the calibrated K instead of the model's estimate.

        Returns:
            tuple:
                - pcd : The reconstructed point cloud (N, 3).
                - camera_pose : The estimated camera extrinsics (T, 4, 4).
                - norm_cam_ray ): Normalized camera ray directions (H*W, 3).
        """
        # Load all frames, resize to uniform size, and convert to Tensor [N, 3, H, W].
        # When model_type='pi3x', also build the depth / intrinsic conditions for Pi3X.
        imgs, traj_len, conditions = load_images_as_tensor(
            input_path,
            interval=self.interval,
            condit_depth_path=condit_depth_path,
            intrinsics_np=intrinsics_np,
            device=self.device,
        )  # imgs: [N, 3, H, W] on self.device

        # ``K_rescaled`` is metadata (numpy K at resized resolution), not a Pi3X kwarg —
        # pop it before splatting so ``model(**conditions)`` does not see an unknown argument.
        K_rescaled = conditions.pop("K_rescaled", None)

        # Run model inference to get point clouds and camera poses
        print("Running model inference...")
        dtype = self.amp_dtype
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                if self.model_type == "pi3x":
                    res = self.model(imgs[None], **conditions)  # Add batch dimension [1, N, 3, H, W]
                else:
                    res = self.model(imgs[None])  # Add batch dimension [1, N, 3, H, W]

        # # Filter noise using masks
        # masks = (
        #     torch.sigmoid(res["conf"][..., 0]) > 0.1
        # )  # Retain high-confidence points
        # non_edge = ~depth_edge(
        #     res["local_points"][..., 2], rtol=0.03
        # )  # Filter depth edges to sharpen point cloud boundaries

        # Filter noise using masks
        if self.model_type == "pi3x" and (condit_depth_path is not None):
            # 适度放宽带有真实传感器深度的检测：适当降低置信度，增加深度边缘容忍度
            masks = torch.sigmoid(res["conf"][..., 0]) > 0.05
            non_edge = ~depth_edge(
                res["local_points"][..., 2], rtol=0.15
            )  # 容忍更大范围的深度突变
        else:
            masks = torch.sigmoid(res["conf"][..., 0]) > 0.1
            non_edge = ~depth_edge(res["local_points"][..., 2], rtol=0.03)

        masks = torch.logical_and(masks, non_edge)[
            0
        ]  # Keep points that are both confident and non-edge

        # Extract points and colors
        pcd = res["points"][0][masks]  # [N, H, W, 3]
        pcd_color = imgs.permute(0, 2, 3, 1)[masks]
        camera_pose = res["camera_poses"][0].cpu().numpy()

        # Interpolate camera poses to match the full trajectory length
        camera_pose = interpolate_extrinsics(
            camera_pose,
            np.arange(camera_pose.shape[0]) * self.interval,
            np.arange(traj_len),
        )

        # Get normalized camera rays in camera coordinates
        ref_cam_index = 0
        norm_cam_ray_cam_coords = res["local_points"][0][ref_cam_index] / res[
            "local_points"
        ][0][ref_cam_index].norm(dim=2, keepdim=True)

        # Interpolate norm_cam_ray_cam_coords to 2× density
        norm_cam_ray_cam_coords = norm_cam_ray_cam_coords.unsqueeze(0).permute(
            0, 3, 1, 2
        )
        norm_cam_ray_cam_coords = torch.nn.functional.interpolate(
            norm_cam_ray_cam_coords,
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        ).permute(0, 2, 3, 1)

        norm_cam_ray_cam_coords = norm_cam_ray_cam_coords[0]
        # Camera intrinsics at the resized model resolution.
        # Prefer the calibrated K (rescaled to model input size) when an external intrinsic
        # was supplied — the MVP experiment (tools/exp_intrinsic) showed RGB-only Pi3X
        # back-calculates fx/fy with a systematic ~2-3% bias, while the calibrated K is
        # known to be more accurate. Fall back to DLT estimation only when no real K is given.
        if K_rescaled is not None:
            self.camera_intric_rs = K_rescaled.astype(np.float32)
            print(f"[intrinsics] using calibrated K (rescaled to model input):\n{self.camera_intric_rs}")
        else:
            self.camera_intric_rs = estimate_intrinsics(
                res["local_points"][0][ref_cam_index]
            ).cpu().numpy()
            print(f"[intrinsics] no real K provided; using DLT-estimated K:\n{self.camera_intric_rs}")

        if torch.isnan(pcd).any() or torch.isinf(pcd).any():
            print("[Reconstruction] NaN/Inf detected in Model Output! Cleaning...")
            valid_mask = ~torch.isnan(pcd).any(dim=1) & ~torch.isinf(pcd).any(dim=1)
            pcd = pcd[valid_mask]
            pcd_color = pcd_color[valid_mask]

        pcd = pcd.cpu().numpy()
        pcd_color = pcd_color.cpu().numpy()
        pcd_ocd = o3d.geometry.PointCloud()
        pcd_ocd.points = o3d.utility.Vector3dVector(pcd)
        pcd_ocd.colors = o3d.utility.Vector3dVector(pcd_color)

        # 1. 执行统计学离群点剔除 (Statistical Outlier Removal)
        # 能够有效铲除由真实深度图带来的漫天“飞点”，防止它们撑大场景包围盒
        pcd_ocd, ind = pcd_ocd.remove_statistical_outlier(
            nb_neighbors=30, std_ratio=2.0
        )
        pcd = np.asarray(pcd_ocd.points)
        pcd_color = np.asarray(pcd_ocd.colors)

        # Calculate scene bounds robustly using 1% and 99% percentiles to avoid outliers inflating the volume
        if pcd.shape[0] > 100:
            q1 = np.percentile(pcd, 1, axis=0)
            q99 = np.percentile(pcd, 99, axis=0)
            loc_range = np.clip(q99 - q1, 1e-3, None)
        else:
            loc_range = pcd.max(0) - pcd.min(0)

        loc_vol = np.prod(loc_range)  # Robust Total volume
        pcd_num = pcd.shape[0]
        frame_num = imgs.shape[0]

        # Dynamic voxel size calculation, with safeguards against extreme values
        voxel_size = loc_vol / max(pcd_num, 1) * frame_num * self.voxel_size_scale
        # 限制 voxel_size 上限，防止异常大的 volume 产生过大的网格导致点云剧烈坍缩
        voxel_size = np.clip(voxel_size, 0.01, 0.2)

        # Voxel downsampling
        pcd_ocd = pcd_ocd.voxel_down_sample(voxel_size=voxel_size)
        pcd = np.asarray(pcd_ocd.points)  # [M, 3] after downsampling
        pcd_color = np.asarray(pcd_ocd.colors)
        self.pcd_color = pcd_color
        self.pcd = pcd  # Point cloud in World Coordinate System

        print(f"downsample the pcd from {pcd_num} to {pcd.shape[0]}")

        return pcd, camera_pose, norm_cam_ray_cam_coords.reshape(-1, 3)

    def pcd_to_occ(self, pcd):
        """
        Converts a point cloud into an occupancy representation.
        Steps: Mesh Reconstruction -> Vertex Sampling -> Spatial Filtering -> Voxelization -> World Coord Recovery.

        Args:
            pcd : The input point cloud (N, 3).

        Returns:
            occ_pcd: Occupancy point cloud (M, 3).
        """
        if not isinstance(pcd, torch.Tensor):
            device = self.device
            pcd = torch.from_numpy(pcd).float().to(device)
        else:
            device = pcd.device

        pc_range_min = torch.tensor(self.pc_range[:3], device=device)
        pc_range_max = torch.tensor(self.pc_range[3:], device=device)
        voxel_size = self.voxel_size

        mask = (
            (pcd[:, 0] > pc_range_min[0])
            & (pcd[:, 0] < pc_range_max[0])
            & (pcd[:, 1] > pc_range_min[1])
            & (pcd[:, 1] < pc_range_max[1])
            & (pcd[:, 2] > pc_range_min[2])
            & (pcd[:, 2] < pc_range_max[2])
        )
        pcd = pcd[mask]

        voxel_indices = torch.floor((pcd - pc_range_min) / voxel_size).long()

        unique_voxel_indices = torch.unique(voxel_indices, dim=0)
        occ_pcd = (
            unique_voxel_indices.float() * voxel_size
            + pc_range_min
            + (voxel_size * 0.5)
        )

        return occ_pcd

    def pcd_to_points(self, pcd):
        """
        Converts a point cloud into a set of points.

        Args:
            pcd : The input point cloud (N, 3).

        Returns:
            points: Point cloud in world coordinates (N, 3).
        """
        # Convert to Open3D PointCloud object
        pcd = pcd.cpu().numpy() if isinstance(pcd, torch.Tensor) else pcd
        pcd = pcd.astype(np.float64)
        valid_mask = np.isfinite(pcd).all(axis=1)
        if np.sum(valid_mask) < len(pcd):
            print(f"[Warning] Removed {len(pcd) - np.sum(valid_mask)} NaN/Inf points.")
            pcd = pcd[valid_mask]
        if len(pcd) < 100:
            print("[Error] Too few points for reconstruction! Returning raw points.")
            return pcd

        try:
            point_cloud_original = o3d.geometry.PointCloud()
            point_cloud_original.points = o3d.utility.Vector3dVector(pcd)

            with_normal = preprocess(point_cloud_original, self.config, normals=True)

            if with_normal.has_normals():
                normals = np.asarray(with_normal.normals)
                if np.isnan(normals).any():
                    # print("[Warning] NaN detected in normals! Cleaning...")
                    valid_normal_mask = np.isfinite(normals).all(axis=1)
                    clean_points = np.asarray(with_normal.points)[valid_normal_mask]
                    clean_normals = normals[valid_normal_mask]

                    with_normal2 = o3d.geometry.PointCloud()
                    with_normal2.points = o3d.utility.Vector3dVector(clean_points)
                    with_normal2.normals = o3d.utility.Vector3dVector(clean_normals)
                else:
                    with_normal2 = with_normal
            else:
                return pcd

            mesh, _ = create_mesh_from_map(
                None,
                self.config["depth"],
                self.config["n_threads"],
                self.config["min_density"],
                with_normal2,
            )
            scene_points = np.asarray(mesh.vertices, dtype=float)

            if len(scene_points) == 0:
                return pcd

            return scene_points

        except Exception as e:
            print(
                f"[Error] Mesh reconstruction failed: {e}. Returning original points."
            )
            import pdb

            pdb.set_trace()
            return pcd

    def check_visual_occ(self, occ_pcd, T_cam2base=None):
        """
        Performs Ray Casting to check which occupancy voxels are visible from the current camera pose.

        Args:
            occ_pcd : Occupancy point cloud in Base Coordinates (N, 3).
            T_cam2base : Camera to Base Transformation Matrix (4, 4).

        Returns:
            tuple:
                - occ_voxels : Visible occupied voxels in Base Coordinates (K, 3).
                - camera_visible_mask : All voxels traversed by rays (Free + Occupied) in Base Coordinates (M, 3).
        """
        # Transform Camera Coords to Voxel Indices
        occ_voxels = pcd_to_voxels(
            occ_pcd, self.voxel_size, self.pc_range
        )  # Shape: (-1, 3) in grid indices

        occ_voxels = torch.tensor(occ_voxels, device=self.norm_cam_ray.device)
        occ_size_tensor = torch.tensor(
            self.config["occ_size"], device=self.norm_cam_ray.device
        )
        zero_size_tensor = torch.tensor([0, 0, 0], device=self.norm_cam_ray.device)

        # Filter voxels strictly within the defined map size
        mask_in_occ_range_max = (occ_voxels < occ_size_tensor).all(1)
        mask_in_occ_range_min = (occ_voxels >= zero_size_tensor).all(1)
        mask_in_occ_range = mask_in_occ_range_max * mask_in_occ_range_min
        occ_voxels = occ_voxels[mask_in_occ_range]

        # Ray Casting Setup
        max_distance = (
            int(
                np.sqrt(
                    (self.pc_range[3] - self.pc_range[0]) ** 2
                    + (self.pc_range[5] - self.pc_range[2]) ** 2
                    + (self.pc_range[4] - self.pc_range[1]) ** 2  # 加入 Y 轴 (前后深度)
                )
            )
            / self.voxel_size
            + 1
        )

        ray_cast_step_size = 1.0
        if T_cam2base is not None:
            if not isinstance(T_cam2base, torch.Tensor):
                T_cam2base = torch.tensor(
                    T_cam2base, device=self.norm_cam_ray.device, dtype=torch.float32
                )

            # Ray origin remains at (0,0,0); only rotate the camera coordinate system
            ray_position = torch.zeros(1, 3, device=self.norm_cam_ray.device)

            # Ray direction must be multiplied by rotation matrix R to convert from camera to base view
            R_c2b = T_cam2base[:3, :3]
            ray_direction_norm = torch.matmul(self.norm_cam_ray.reshape(-1, 3), R_c2b.T)
        else:
            ray_position = torch.zeros(
                1, 3, device=self.norm_cam_ray.device
            )  # Default to (0,0,0)
            ray_direction_norm = self.norm_cam_ray.reshape(-1, 3)

        pc_range_tensor = torch.tensor(
            self.pc_range[:3], device=self.norm_cam_ray.device
        )  # World origin offset

        # Convert ray origin to voxel coordinates
        ray_position = (ray_position - pc_range_tensor) / self.voxel_size

        # Initialize 3D Grid for Visibility
        camera_visible_mask_3d = torch.zeros(
            self.config["occ_size"], dtype=torch.bool, device=ray_direction_norm.device
        )
        occ_voxels_3d = camera_visible_mask_3d.clone()

        # Mark occupied voxels in the 3D grid
        D, H, W = self.config["occ_size"]
        idx_1d = occ_voxels[:, 0] * (H * W) + occ_voxels[:, 1] * W + occ_voxels[:, 2]
        idx_1d = idx_1d.long()
        occ_voxels_3d.view(-1).index_fill_(0, idx_1d, 1)

        # Begin Ray Marching
        steps = torch.arange(
            0, max_distance, ray_cast_step_size, device=self.norm_cam_ray.device
        )
        ray_positions = ray_position.unsqueeze(0) + ray_direction_norm.unsqueeze(
            1
        ) * steps.unsqueeze(0).unsqueeze(-1)
        voxel_coords = torch.floor(ray_positions).long()
        D, H, W = self.config["occ_size"]
        valid_mask = (
            (voxel_coords[..., 0] >= 0)
            & (voxel_coords[..., 0] < D)
            & (voxel_coords[..., 1] >= 0)
            & (voxel_coords[..., 1] < H)
            & (voxel_coords[..., 2] >= 0)
            & (voxel_coords[..., 2] < W)
        )

        flat_coords = (
            voxel_coords[..., 0] * (H * W)
            + voxel_coords[..., 1] * W
            + voxel_coords[..., 2]
        )
        occ_flat = occ_voxels_3d.view(-1)
        sampled_occ = torch.where(
            valid_mask,
            occ_flat[flat_coords.clamp(0, occ_flat.size(0) - 1)],
            torch.tensor(0, device=occ_flat.device, dtype=torch.bool),
        )  # 不补0没法组matrixs

        # The first occurrence of 1 on each ray indicates the starting position where the voxel becomes occluded, excluding the voxel itself.
        hit_mask = (sampled_occ > 0).cumsum(dim=1) > 0
        hit_mask[:, 1:] = hit_mask[:, :-1]

        visible_indices = flat_coords[valid_mask & ~hit_mask]
        camera_visible_mask_3d.view(-1).index_fill_(0, visible_indices, 1)
        occ_voxels_3d = occ_voxels_3d * camera_visible_mask_3d

        # Convert back to coordinates (N, 3)
        occ_voxels = torch.nonzero(occ_voxels_3d)
        values = occ_voxels_3d[occ_voxels[:, 0], occ_voxels[:, 1], occ_voxels[:, 2]]
        occ_voxels = torch.cat([occ_voxels, values.unsqueeze(1)], dim=1)

        camera_visible_mask = torch.nonzero(camera_visible_mask_3d)
        camera_visible_mask = torch.cat(
            [camera_visible_mask, torch.ones_like(camera_visible_mask[:, :1])], dim=1
        )

        return occ_voxels, camera_visible_mask

    def convert_pointcloud_world_to_camera(self, points_world, T_cw):
        """
        Transforms point cloud from World to Camera frame.
        Supports both Numpy and PyTorch Tensor (GPU).
        """
        # 1. Tensor Mode (GPU Optimized)
        if isinstance(points_world, torch.Tensor):
            if not isinstance(T_cw, torch.Tensor):
                T_cw = torch.tensor(
                    T_cw, device=points_world.device, dtype=points_world.dtype
                )

            R_cw = T_cw[:3, :3]
            t_cw = T_cw[:3, 3]

            # Logic: P_cam = (P_world - t_cw) @ R_cw
            # Note: R_wc = R_cw.T. The formula is P_cam = (R_wc @ (P_world - t_cw).T).T
            # Which simplifies to: (P_world - t_cw) @ R_wc.T => (P_world - t_cw) @ R_cw
            points_camera = (points_world - t_cw) @ R_cw
            return points_camera

        # 2. Numpy Mode (Legacy)
        else:
            if len(points_world) == 0:
                return np.zeros((0, 3), dtype=np.float32)
            R_cw = T_cw[:3, :3]
            t_cw = T_cw[:3, 3]
            R_wc = R_cw.T
            points_camera = (R_wc @ (points_world - t_cw).T).T
            return points_camera.astype(np.float32)

    def convert_pointcloud_camera_to_world(self, points_camera, T_cw):
        """
        Transforms point cloud from Camera Coordinate System to World Coordinate System.

        Args:
            points_camera : Points in camera frame (N, 3).
            T_cw : Camera extrinsic matrix (4, 4).

        Returns:
            points_world: Points in world frame (N, 3).
        """
        if points_camera is None or len(points_camera) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        if isinstance(points_camera, torch.Tensor):
            points_camera = points_camera.detach().cpu().numpy()

        if points_camera.ndim == 1:
            points_camera = points_camera.reshape(1, -1)

        R_cw = T_cw[:3, :3]
        t_cw = T_cw[:3, 3]

        # Formula: P_world = R * P_cam + t
        points_world = (R_cw @ points_camera.T).T + t_cw

        return points_world.astype(np.float32)

    def convert_pointcloud_camera_to_base(self, points_camera, T_cam2base):
        """
        Transforms point cloud from Camera Coordinate System to Robot Base Coordinate System (Rotation ONLY).

        Args:
            points_camera : Points in camera frame (N, 3).
            T_cam2base : Camera to Base  extrinsic matrix (4, 4).

        Returns:
            points_base: Points in base frame (N, 3).
        """
        if points_camera is None or len(points_camera) == 0:
            if isinstance(points_camera, torch.Tensor):
                return torch.zeros(
                    (0, 3), device=points_camera.device, dtype=points_camera.dtype
                )
            return np.zeros((0, 3), dtype=np.float32)

        # 1. Tensor Mode
        if isinstance(points_camera, torch.Tensor):
            if not isinstance(T_cam2base, torch.Tensor):
                T_cam2base = torch.tensor(
                    T_cam2base, device=points_camera.device, dtype=points_camera.dtype
                )

            R_c2b = T_cam2base[:3, :3]
            # col：P_base= P_cam@ R_c2b
            # row：P_base.T = P_cam.T @ R_c2b.T
            points_base = torch.matmul(points_camera, R_c2b.T)
            return points_base

        # 2. Numpy Mode
        else:
            if points_camera.ndim == 1:
                points_camera = points_camera.reshape(1, -1)

            R_c2b = T_cam2base[:3, :3]

            # Formula: P_base = P_cam @ R_b2c
            points_base = points_camera @ R_c2b.T
            return points_base.astype(np.float32)

    def get_temporal_occ(
        self, new_occ_world, current_pose_matrix, save_to_history=False
    ):
        """
        Accumulates OCC data over a sliding window and transforms it to the current camera frame.

        Args:
            new_occ_world : New OCC points from current frame in World Coordinates (N, 3).
            current_pose_matrix : Current camera pose (4, 4) in World Coordinates.
            save_to_history : If True, appends current data to the sliding buffer.

        Returns:
            tuple:
                - merged_occ_cam : Accumulated OCC in current camera frame (M, 3).
                - merged_occ_world : Accumulated OCC in world frame (M, 3).
        """
        # Save current frame to history buffer if requested
        if save_to_history and len(new_occ_world) > 0:
            self.occ_history_buffer.append(new_occ_world)

        # Retrieve all historical points
        candidates = list(self.occ_history_buffer)

        # If current frame was not saved to history, add it temporarily for visualization
        if not save_to_history and len(new_occ_world) > 0:
            candidates.append(new_occ_world)

        if len(candidates) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros(
                (0, 3), dtype=np.float32
            )

        # Merge all points in World Frame
        merged_occ_world = np.concatenate(candidates, axis=0)

        # Voxel Downsampling to remove duplicates
        if len(merged_occ_world) > 0:
            pcd_tmp = o3d.geometry.PointCloud()
            pcd_tmp.points = o3d.utility.Vector3dVector(merged_occ_world)
            pcd_tmp = pcd_tmp.voxel_down_sample(voxel_size=self.voxel_size)
            merged_occ_world = np.asarray(pcd_tmp.points, dtype=np.float32)

        if len(merged_occ_world) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros(
                (0, 3), dtype=np.float32
            )

        # Transform to Current Camera Frame
        merged_occ_cam = self.convert_pointcloud_world_to_camera(
            merged_occ_world, current_pose_matrix
        )

        return merged_occ_cam, merged_occ_world

    def get_gt_poses(self, input_path):
        """
        Retrieves Ground Truth (GT) camera trajectories.
        Should be overridden by subclasses.

        Args:
            input_path (str): Path to the input data directory.

        Returns:
            np.ndarray or None: Array of shape (N, 4, 4) if GT exists, else None.
        """
        return None

    def compute_trajectory_scale(self, poses_gt, poses_pred):
        """
        Computes the scale ratio (GT / Pred) between predicted and ground truth trajectories.
        Uses the ratio of standard deviations (Sim3 scale estimation).

        Args:
            poses_gt : Ground truth poses (N, 4, 4).
            poses_pred : Predicted poses (N, 4, 4).

        Returns:
            scale or 1.0: The calculated scale factor. Returns 1.0 if calculation fails or input is invalid.
        """
        return 1.0

    def align_with_gt_scale(self, input_path, pcd):
        """
        Attempts to align the predicted point cloud scale with Ground Truth.
        Dependent on `get_gt_poses`.

        Args:
            input_path : Path to the input data.
            pcd : The predicted point cloud (N, 3).

        Returns:
            tuple:
                - pcd : The scaled point cloud.
                - scale : The applied scale factor.
        """
        return pcd, 1.0

    def get_io_paths(self, input_path):
        """
        Defines output file paths.
        Subclasses can override this for complex directory structures.

        Args:
            input_path : The input file or directory path.

        Returns:
            paths: A dictionary containing paths for 'ply', 'global_occ', 'occ_seq', and 'mask_seq'.
        """
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        return {
            "ply": os.path.join(self.save_path, f"{base_name}_global.ply"),
            "global_occ": os.path.join(self.save_path, f"{base_name}_global_occ.npz"),
            "occ_seq": os.path.join(self.save_path, f"{base_name}_occ_seq.npz"),
            "mask_seq": os.path.join(self.save_path, f"{base_name}_mask_seq.npz"),
        }

    def save_global_data(self, paths):
        """
        Saves global point cloud and global occupancy map.

        Args:
            paths : A dictionary of file paths (output of `get_io_paths`).

        Returns:
            None: Saves files to disk.
        """

        pcd_to_save = self.pcd
        if isinstance(pcd_to_save, torch.Tensor):
            pcd_to_save = pcd_to_save.detach().cpu().numpy()

        pcd_color_to_save = self.pcd_color
        if isinstance(pcd_color_to_save, torch.Tensor):
            pcd_color_to_save = pcd_color_to_save.detach().cpu().numpy()

        write_ply(pcd_to_save, pcd_color_to_save, paths["ply"])

        # write traj ply
        camera_pose_to_save = self.camera_pose
        if isinstance(camera_pose_to_save, torch.Tensor):
            camera_pose_to_save = camera_pose_to_save.detach().cpu().numpy()
        camera_pose_to_save = np.asarray(camera_pose_to_save)
        if camera_pose_to_save.ndim == 3 and camera_pose_to_save.shape[-2:] == (
            4,
            4,
        ):
            traj_points = camera_pose_to_save[:, :3, 3].astype(np.float32)
            traj_colors = np.zeros_like(traj_points, dtype=np.float32)
            traj_colors[:, 0] = 1.0  # Red camera trajectory points
            trajectory_ply_path = paths.get(
                "camera_trajectory_ply",
                os.path.join(
                    os.path.dirname(paths["ply"]), "camera_trajectory_orin.ply"
                ),
            )
            write_ply(traj_points, traj_colors, trajectory_ply_path)
            print(f"Saved Camera Trajectory to {trajectory_ply_path}")

        occ_pcd_to_save = self.occ_pcd

        if isinstance(occ_pcd_to_save, torch.Tensor):
            occ_pcd_to_save = occ_pcd_to_save.detach().cpu().numpy()

        occ_frame_pcd = getattr(self, "occ_frame_pointcloud", None)
        if occ_frame_pcd is not None:
            if isinstance(occ_frame_pcd, torch.Tensor):
                occ_frame_pcd = occ_frame_pcd.detach().cpu().numpy()
            occ_frame_pcd = np.asarray(occ_frame_pcd, dtype=np.float32)
            if occ_frame_pcd.ndim == 2 and occ_frame_pcd.shape[1] == 3:
                occ_frame_pcd_path = paths.get(
                    "occ_frame_pointcloud_ply",
                    os.path.join(
                        os.path.dirname(paths["global_occ"]),
                        "occ_frame_pointcloud.ply",
                    ),
                )
                occ_frame_colors = np.zeros_like(occ_frame_pcd, dtype=np.float32)
                occ_frame_colors[:, 1] = 1.0  # Green point cloud in final OCC frame
                write_ply(occ_frame_pcd, occ_frame_colors, occ_frame_pcd_path)
                print(f"Saved OCC-frame Point Cloud to {occ_frame_pcd_path}")

        occ_frame_traj = getattr(self, "occ_frame_camera_trajectory", None)
        if occ_frame_traj is not None:
            if isinstance(occ_frame_traj, torch.Tensor):
                occ_frame_traj = occ_frame_traj.detach().cpu().numpy()
            occ_frame_traj = np.asarray(occ_frame_traj, dtype=np.float32)
            if occ_frame_traj.ndim == 2 and occ_frame_traj.shape[1] == 3:
                traj_occ_ply_path = paths.get(
                    "camera_trajectory_occ_frame_ply",
                    os.path.join(
                        os.path.dirname(paths["global_occ"]),
                        "camera_trajectory_occ_frame.ply",
                    ),
                )
                traj_occ_colors = np.zeros_like(occ_frame_traj, dtype=np.float32)
                traj_occ_colors[:, 0] = 1.0  # Red trajectory points in final OCC frame
                write_ply(occ_frame_traj, traj_occ_colors, traj_occ_ply_path)
                print(f"Saved OCC-frame Camera Trajectory to {traj_occ_ply_path}")

        np.savez_compressed(
            paths["global_occ"], data=occ_pcd_to_save.astype(np.float32)
        )
        print(f"Saved Global Data to {paths['global_occ']}")

    def save_sequence_data(self, paths, sparse_occ_indices, packed_mask_data):
        """
        Saves sequence data to disk.
        - OCC: Stored as Sparse CSR Matrix in .npz format.
        - Mask: Stored as Packed Bit Array in .npz format.

        Args:
            paths : Dictionary containing 'occ_seq' and 'mask_seq' paths.
            sparse_occ_indices : Array of sparse indices (Frame, X, Y, Z).
            packed_mask_data : Compressed bitmask array.

        Returns:
            None: Saves files to disk.
        """
        import scipy.sparse as sparse

        N = len(self.camera_pose)
        grid_size = self.config["occ_size"]
        H, W, D = grid_size
        flat_dim = H * W * D

        # --- Save OCC ---
        if "occ_seq" in paths:
            t_start = time.time()
            if len(sparse_occ_indices) == 0:
                sparse_mat = sparse.csr_matrix((N, flat_dim), dtype=np.uint8)
            else:
                times = sparse_occ_indices[:, 0]
                xs, ys, zs = (
                    sparse_occ_indices[:, 1],
                    sparse_occ_indices[:, 2],
                    sparse_occ_indices[:, 3],
                )
                flat_indices = (
                    xs.astype(np.int64) * (W * D)
                    + ys.astype(np.int64) * D
                    + zs.astype(np.int64)
                )
                data = np.ones(len(flat_indices), dtype=np.uint8)
                sparse_mat = sparse.csr_matrix(
                    (data, (times, flat_indices)), shape=(N, flat_dim)
                )

            sparse.save_npz(paths["occ_seq"], sparse_mat)
            print(f"Saved OCC in {time.time() - t_start:.2f}s")

        # --- Save Mask ---
        if "mask_seq" in paths:
            t_start = time.time()
            np.savez_compressed(
                paths["mask_seq"], data=packed_mask_data, shape=grid_size, mode="packed"
            )
            print(f"Saved Mask in {time.time() - t_start:.2f}s")

    def compute_sequence_data(self, pcd, mesh=True, T_cam2base=None, scale=1.0):
        """
        Computes sequential data for the entire trajectory, including sparse OCC indices
        and compressed visibility masks.

        Args:
            None

        Returns:
            tuple:
                - final_occ : Sparse OCC indices (N, 4) -> [Frame, X, Y, Z].
                - final_mask_packed : Compressed mask data.
                - all_camera_poses : List of camera poses.
                - all_camera_intrinsics : List of camera intrinsics.
        """
        total_frames = len(self.camera_pose)
        grid_dims = self.config["occ_size"]  # (H, W, D)
        device = self.device

        # Lists for storage
        all_sparse_indices_occ = []
        all_packed_masks = []
        all_camera_poses = []

        # Prepare Intrinsics
        current_intrinsic = self.camera_intric_rs.astype(np.float32)
        all_camera_intrinsics = [[row for row in current_intrinsic]] * total_frames

        # Convert pcd to points
        if mesh:
            print(f"Using mesh")
            pcd_points_world_np = self.pcd_to_points(pcd)
        else:
            print(f"Using origin point cloud")
            pcd_points_world_np = pcd
        pcd_points_world = torch.from_numpy(pcd_points_world_np).float().to(device)

        print(f"Processing {total_frames} frames (Simple Packed Mode)...")
        occ_start = time.time()

        # collect camera poses
        self.camera_pose = self.camera_pose.astype(np.float32)
        all_camera_poses = [[row for row in pose] for pose in self.camera_pose]
        camera_poses = torch.from_numpy(self.camera_pose).to(device).float()
        self.occ_frame_pointcloud = None
        self.occ_frame_camera_trajectory = None
        for i in range(total_frames):
            current_pose = camera_poses[i]

            # Transform global OCC to current Camera Coordinates
            pcd_points_cam = self.convert_pointcloud_world_to_camera(
                pcd_points_world, current_pose
            )  # Shape: (-1, 3) in meters

            pcd_points_cam *= scale

            if T_cam2base is not None:
                pcd_points_base = self.convert_pointcloud_camera_to_base(
                    pcd_points_cam, T_cam2base
                )  # Shape: (-1, 3) in meters
            else:
                pcd_points_base = pcd_points_cam

            if i == total_frames - 1:
                if isinstance(pcd_points_base, torch.Tensor):
                    self.occ_frame_pointcloud = pcd_points_base.detach().cpu().numpy()
                else:
                    self.occ_frame_pointcloud = np.asarray(pcd_points_base)
                traj_points_world = camera_poses[:, :3, 3]
                traj_points_cam = self.convert_pointcloud_world_to_camera(
                    traj_points_world, current_pose
                )
                traj_points_cam *= scale
                if T_cam2base is not None:
                    traj_points_base = self.convert_pointcloud_camera_to_base(
                        traj_points_cam, T_cam2base
                    )
                else:
                    traj_points_base = traj_points_cam
                if isinstance(traj_points_base, torch.Tensor):
                    self.occ_frame_camera_trajectory = (
                        traj_points_base.detach().cpu().numpy()
                    )
                else:
                    self.occ_frame_camera_trajectory = np.asarray(traj_points_base)

            # Convert to occupancy (pcd is maintained at aligned scale)
            self.occ_pcd = self.pcd_to_occ(pcd_points_base)

            # Check visibility
            valid_voxels_occ, cam_visible_mask = self.check_visual_occ(
                self.occ_pcd, T_cam2base
            )

            if len(valid_voxels_occ) > 0:
                time_col = torch.full(
                    (len(valid_voxels_occ), 1), i, device=device, dtype=torch.int16
                )
                frame_indices = torch.cat([time_col, valid_voxels_occ.short()], dim=1)

                all_sparse_indices_occ.append(frame_indices)

            # Construct single frame Bool Grid (memory intensive momentarily)
            frame_grid = torch.zeros(grid_dims, dtype=torch.bool, device=device)
            if len(cam_visible_mask) > 0:
                frame_grid[
                    cam_visible_mask[:, 0],
                    cam_visible_mask[:, 1],
                    cam_visible_mask[:, 2],
                ] = True

            # Compress using packbits
            all_packed_masks.append(frame_grid)
            if i % 50 == 0:
                print(f"  Frame {i}/{total_frames} packed.")
        occ_end = time.time()
        print(f"GPU OCC Sequence cost: {occ_end - occ_start:.4f}s")

        final_occ = (
            torch.concat(all_sparse_indices_occ, dim=0).cpu().numpy()
            if all_sparse_indices_occ
            else np.zeros((0, 4), dtype=np.int16)
        )
        final_mask_packed = torch.stack(all_packed_masks, dim=0)
        final_mask_packed = (
            final_mask_packed.reshape(final_mask_packed.shape[0], -1).cpu().numpy()
        )
        final_mask_packed = np.packbits(final_mask_packed, axis=1)
        return final_occ, final_mask_packed, all_camera_poses, all_camera_intrinsics

    def update_metadata(
        self, paths, all_camera_poses, all_camera_intrinsics, input_path
    ):
        """
        Update Parquet metadata (to be implemented by subclass).

        Args:
            paths : Output paths.
            all_camera_poses : List of camera poses.
            all_camera_intrinsics : List of camera intrinsics.
            input_path : Input video path.

        Returns:
            None
        """
        pass

    def update_meta_episodes_jsonl(self, scale):
        """
        Update episodes.jsonl metadata (to be implemented by subclass).

        Args:
            scale : The calculated scale factor.

        Returns:
            None
        """
        pass

    # Single-frame OCC pipeline
    def single_frame_pipeline(
        self,
        input_path,
        condit_depth_path=None,
        intrinsics_np=None,
        pcd_save=False,
        mesh=False,
    ):
        """
        Generates estimated OCC map and camera trajectory from a full video episode.

        Args:
            input_path : Path to the input video.
            condit_depth_path : Path to the conditional depth map (Pi3X only).
            intrinsics_np : 3x3 intrinsic matrix (Pi3X only).
            pcd_save : If True, saves visualization files (occ.ply, etc.).

        Returns:
            None: Sets self.camera_pose, and optionally saves files.
        """
        self.camera_intric = np.array(
            [[168.0498, 0.0, 240.0], [0.0, 192.79999, 135.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )  # Temporary hardcoded intrinsics

        # Reconstruct PCD and Trajectory
        pcd, self.camera_pose, self.norm_cam_ray = self.pcd_reconstruction(
            input_path, condit_depth_path, intrinsics_np
        )

        if mesh:
            pcd_points_world_np_0 = self.pcd_to_points(pcd)
        else:
            pcd_points_world_np_0 = pcd

        # Transform global OCC to current Camera Coordinates
        pcd_cam_0 = self.convert_pointcloud_world_to_camera(
            pcd_points_world_np_0, self.camera_pose[0]
        )  # Shape: (-1, 3) in meters

        # Convert to occupancy (pcd is maintained at aligned scale)
        occ_pcd_cam_0 = self.pcd_to_occ(pcd_cam_0)

        # Check visibility for Frame 0
        occ_voxels_visual_0, visual_mask_0 = self.check_visual_occ(occ_pcd_cam_0)

        if pcd_save:
            save_path = input_path.split("videos/")[0]
            write_ply(
                occ_pcd_cam_0[:, :3], path=os.path.join(self.save_path, "occ.ply")
            )
            occ_pcd_cam_0 = np.concatenate(
                [occ_pcd_cam_0, np.ones((occ_pcd_cam_0.shape[0], 1))], axis=1
            )
            np.save(os.path.join(self.save_path, "occ_pcd_cam0.npy"), occ_pcd_cam_0)

            # Convert Frame 0 visible voxels back to world coords for viz
            occ_pcd_visual_0 = voxels_to_pcd(
                occ_voxels_visual_0, self.voxel_size, self.pc_range
            )
            write_ply(
                occ_pcd_visual_0[:, :3],
                path=os.path.join(self.save_path, "occ_visual.ply"),
            )
            if isinstance(occ_pcd_visual_0, torch.Tensor):
                occ_pcd_visual_0 = occ_pcd_visual_0.cpu().numpy()
            occ_pcd_visual_0 = np.concatenate(
                [occ_pcd_visual_0, np.ones((occ_pcd_visual_0.shape[0], 1))], axis=1
            )
            np.save(
                os.path.join(self.save_path, "occ_pcd_visual_cam0.npy"),
                occ_pcd_visual_0,
            )

    # Visualization pipeline with historical accumulation
    def visual_pipeline(
        self,
        input_path,
        condit_depth_path=None,
        intrinsics_np=None,
        pcd_save=False,
        mesh=False,
        max_frames=None,
    ):
        """
        Executes the full visualization pipeline with sliding window accumulation.
        Generates PLY/NPY files for merged views, solo OCC, and sequence data.

        Args:
            input_path : Path to the input video.
            condit_depth_path : Path to the conditional depth map (Pi3X only).
            intrinsics_np : 3x3 intrinsic matrix (Pi3X only).
            pcd_save : Must be True to trigger the visualization logic.
            max_frames : If set, cap the trajectory to this many frames (quick verification).

        Returns:
            None: Output files are saved to self.save_path.
        """

        # Reconstruct global map and trajectory
        pcd, self.camera_pose, self.norm_cam_ray = self.pcd_reconstruction(
            input_path, condit_depth_path, intrinsics_np
        )

        # Quick verification: cap the trajectory so the full data path runs on only a few frames.
        if (
            max_frames is not None
            and max_frames > 0
            and len(self.camera_pose) > max_frames
        ):
            print(
                f"[Quick] Truncating trajectory from {len(self.camera_pose)} to "
                f"{max_frames} frames for fast pipeline verification."
            )
            self.camera_pose = self.camera_pose[:max_frames]

        if isinstance(pcd, torch.Tensor):
            pcd = pcd.detach().cpu().numpy()

        # Save visualization data if requested
        if pcd_save:
            print("Start processing sequence frames...")

            # --- 1. Create Folders for Merged Visualization ---
            merge_cam_ply_dir = os.path.join(self.save_path, "merge_ply_sequence_cam")
            if not os.path.exists(merge_cam_ply_dir):
                os.makedirs(merge_cam_ply_dir)

            merge_cam_npy_dir = os.path.join(self.save_path, "merge_npy_sequence_cam")
            if not os.path.exists(merge_cam_npy_dir):
                os.makedirs(merge_cam_npy_dir)

            merge_world_ply_dir = os.path.join(
                self.save_path, "merge_ply_sequence_world"
            )
            if not os.path.exists(merge_world_ply_dir):
                os.makedirs(merge_world_ply_dir)

            merge_world_npy_dir = os.path.join(
                self.save_path, "merge_npy_sequence_world"
            )
            if not os.path.exists(merge_world_npy_dir):
                os.makedirs(merge_world_npy_dir)

            # --- 2. Create Folders for Solo Occ ---
            occ_only_cam_ply_dir = os.path.join(self.save_path, "occ_only_cam_ply")
            if not os.path.exists(occ_only_cam_ply_dir):
                os.makedirs(occ_only_cam_ply_dir)

            occ_only_cam_npy_dir = os.path.join(self.save_path, "occ_only_cam_npy")
            if not os.path.exists(occ_only_cam_npy_dir):
                os.makedirs(occ_only_cam_npy_dir)

            total_frames = len(self.camera_pose)

            # --- Handle Frame 0 (Safe Conversion) ---
            pcd_cam_0 = self.convert_pointcloud_world_to_camera(
                pcd, self.camera_pose[0]
            )
            if isinstance(pcd_cam_0, torch.Tensor):
                pcd_cam_0 = pcd_cam_0.cpu().numpy()

            if mesh:
                occ_pcd_cam_points_0 = self.pcd_to_points(pcd_cam_0)
            else:
                occ_pcd_cam_points_0 = pcd_cam_0

            occ_pcd_cam_0 = self.pcd_to_occ(occ_pcd_cam_points_0)

            if isinstance(occ_pcd_cam_0, torch.Tensor):
                occ_pcd_cam_0 = occ_pcd_cam_0.cpu().numpy()

            write_ply(
                occ_pcd_cam_0[:, :3],
                path=os.path.join(self.save_path, "all_occ_cam.ply"),
            )

            # Clear history buffer before processing new video
            self.occ_history_buffer.clear()

            # Convert global pcd to points
            if mesh:
                pcd_points = self.pcd_to_points(pcd)
            else:
                pcd_points = pcd

            # Process each frame in the sequence
            occ_start = time.time()
            for i in range(total_frames):
                current_pose = self.camera_pose[i]

                # ================= A. Compute Data =================

                # Transform global pcd to current Camera Coordinates
                # World -> Camera Coordinates
                pcd_cam = self.convert_pointcloud_world_to_camera(
                    pcd_points, current_pose
                )  # Shape: (-1, 3) in meters

                if isinstance(pcd_cam, torch.Tensor):
                    pcd_cam = pcd_cam.detach().cpu().numpy()

                # Convert to occupancy (pcd is maintained at aligned scale)
                occ_pcd_cam = self.pcd_to_occ(pcd_cam)
                if isinstance(occ_pcd_cam, torch.Tensor):
                    occ_pcd_cam = occ_pcd_cam.detach().cpu().numpy()

                # Calculate visible Occ for current frame
                occ_indices, _ = self.check_visual_occ(occ_pcd_cam)
                if isinstance(occ_indices, torch.Tensor):
                    occ_indices = occ_indices.detach().cpu().numpy()

                single_frame_occ_cam = voxels_to_pcd(
                    occ_indices, self.voxel_size, self.pc_range
                )
                if isinstance(single_frame_occ_cam, torch.Tensor):
                    single_frame_occ_cam = single_frame_occ_cam.detach().cpu().numpy()
                if single_frame_occ_cam.shape[1] == 4:
                    single_frame_occ_cam = single_frame_occ_cam[:, :3]

                # Convert local visible Occ to World Frame
                single_frame_occ_world = self.convert_pointcloud_camera_to_world(
                    single_frame_occ_cam, current_pose
                )
                if isinstance(single_frame_occ_world, torch.Tensor):
                    single_frame_occ_world = (
                        single_frame_occ_world.detach().cpu().numpy()
                    )

                # Get accumulated result from sliding window
                save_flag = i % self.history_step == 0
                local_occ_cam, local_occ_world = self.get_temporal_occ(
                    single_frame_occ_world, current_pose, save_to_history=save_flag
                )
                if isinstance(local_occ_world, torch.Tensor):
                    local_occ_world = local_occ_world.detach().cpu().numpy()

                # Calculate Background and Trajectory
                bg_cam = self.convert_pointcloud_world_to_camera(
                    occ_pcd_cam, current_pose
                )
                if isinstance(bg_cam, torch.Tensor):
                    bg_cam = bg_cam.detach().cpu().numpy()
                if bg_cam.shape[1] == 4:
                    bg_cam = bg_cam[:, :3]

                traj_world = self.camera_pose[:, :3, 3]
                traj_cam = self.convert_pointcloud_world_to_camera(
                    traj_world, current_pose
                )
                if isinstance(traj_cam, torch.Tensor):
                    traj_cam = traj_cam.detach().cpu().numpy()
                if traj_cam.shape[1] == 4:
                    traj_cam = traj_cam[:, :3]

                # Using World Frame Background and Trajectory
                # Note: background is the initial dense point cloud
                bg_world = self.pcd
                if isinstance(bg_world, torch.Tensor):
                    bg_world = bg_world.detach().cpu().numpy()

                traj_current_world = self.camera_pose[0 : i + 1, :3, 3]
                if isinstance(traj_current_world, torch.Tensor):
                    traj_current_world = traj_current_world.detach().cpu().numpy()
                if bg_world.shape[1] == 4:
                    bg_world = bg_world[:, :3]
                if traj_current_world.shape[1] == 4:
                    traj_current_world = traj_current_world[:, :3]

                # ================= B. Save Accumulated Occ (Solo) =================

                # Save PLY
                if len(local_occ_cam) > 0:
                    pure_occ_color = np.zeros_like(local_occ_cam)
                    pure_occ_color[:, 1] = 1.0  # Green color
                    write_ply(
                        local_occ_cam,
                        pure_occ_color,
                        os.path.join(occ_only_cam_ply_dir, f"occ_{i:04d}.ply"),
                    )
                else:
                    pass

                # Save NPY
                if len(local_occ_cam) > 0:
                    # Format: [X, Y, Z, Label=2]
                    occ_npy_single = np.concatenate(
                        [local_occ_cam, np.full((local_occ_cam.shape[0], 1), 2)], axis=1
                    )
                    np.save(
                        os.path.join(occ_only_cam_npy_dir, f"occ_{i:04d}.npy"),
                        occ_npy_single.astype(np.float32),
                    )
                else:
                    np.save(
                        os.path.join(occ_only_cam_npy_dir, f"occ_{i:04d}.npy"),
                        np.zeros((0, 4), dtype=np.float32),
                    )

                # ================= C. Save Mixed Data (Camera Coords) =================

                # Colors
                bg_color = np.ones_like(bg_cam) * 0.7
                traj_color = np.zeros_like(traj_cam)
                traj_color[:, 0] = 1.0

                occ_color = np.zeros_like(local_occ_cam)
                if len(occ_color) > 0:
                    occ_color[:, 1] = 1.0

                # Assemble
                points_list = [bg_cam, traj_cam]
                colors_list = [bg_color, traj_color]
                if len(local_occ_cam) > 0:
                    points_list.append(local_occ_cam)
                    colors_list.append(occ_color)

                final_points = np.concatenate(points_list, axis=0)
                final_colors = np.concatenate(colors_list, axis=0)

                # Save PLY
                write_ply(
                    final_points,
                    final_colors,
                    os.path.join(merge_cam_ply_dir, f"frame_{i:04d}_cam.ply"),
                )

                # Save NPY (With Labels)
                bg_npy = np.concatenate(
                    [bg_cam, np.zeros((bg_cam.shape[0], 1))], axis=1
                )  # Label 0
                traj_npy = np.concatenate(
                    [traj_cam, np.ones((traj_cam.shape[0], 1))], axis=1
                )  # Label 1

                if len(local_occ_cam) > 0:
                    occ_npy = np.concatenate(
                        [local_occ_cam, np.full((local_occ_cam.shape[0], 1), 2)], axis=1
                    )  # Label 2
                    final_npy_data = np.concatenate([bg_npy, traj_npy, occ_npy], axis=0)
                else:
                    final_npy_data = np.concatenate([bg_npy, traj_npy], axis=0)

                np.save(
                    os.path.join(merge_cam_npy_dir, f"frame_{i:04d}_cam.npy"),
                    final_npy_data.astype(np.float32),
                )

                # ================= D. Save Mixed Data (World Coords + True Color) =================

                bg_world_dense = bg_world  # (N, 3)

                # Get Background Color
                if hasattr(self, "pcd_color"):
                    bg_color_dense = self.pcd_color  # (N, 3)
                else:
                    bg_color_dense = np.ones_like(bg_world_dense) * 0.7

                # Align dimensions
                min_len = min(len(bg_world_dense), len(bg_color_dense))
                bg_world_dense = bg_world_dense[:min_len]
                bg_color_dense = bg_color_dense[:min_len]

                # Construct Background NPY: [x, y, z, r, g, b, 0]
                bg_label = np.zeros((min_len, 1))  # Label 0
                bg_npy = np.concatenate(
                    [bg_world_dense, bg_color_dense, bg_label], axis=1
                )

                # Construct Trajectory NPY: [x, y, z, 0, 0, 1, 1]
                traj_len = len(traj_current_world)
                if traj_len > 0:
                    traj_rgb = np.tile([0.0, 0.0, 1.0], (traj_len, 1))  # Blue
                    traj_label = np.ones((traj_len, 1))  # Label 1
                    traj_npy = np.concatenate(
                        [traj_current_world, traj_rgb, traj_label], axis=1
                    )
                else:
                    traj_npy = np.zeros((0, 7))

                # Construct OCC NPY: [x, y, z, 0.5, 0.5, 0.5, 2]
                occ_len = len(local_occ_world)
                if occ_len > 0:
                    occ_rgb = np.tile([0.5, 0.5, 0.5], (occ_len, 1))  # Gray
                    occ_label = np.full((occ_len, 1), 2)  # Label 2
                    occ_npy = np.concatenate(
                        [local_occ_world, occ_rgb, occ_label], axis=1
                    )
                else:
                    occ_npy = np.zeros((0, 7))

                final_npy_data = np.concatenate([bg_npy, traj_npy, occ_npy], axis=0)

                # Save as (N, 7) NPY
                np.save(
                    os.path.join(merge_world_npy_dir, f"frame_{i:04d}_world.npy"),
                    final_npy_data.astype(np.float32),
                )

                # Save PLY (points + colors only)
                write_ply(
                    final_npy_data[:, :3],
                    final_npy_data[:, 3:6],
                    os.path.join(merge_world_ply_dir, f"frame_{i:04d}_world.ply"),
                )

                if i % 10 == 0:
                    print(f"Processed frame {i}/{total_frames}")
            occ_end = time.time()
            print(f"GPU OCC gen and save cost: {occ_end - occ_start}s")

    # Standard Pipeline for Occ Data Generation
    def run_pipeline(
        self, input_path, condit_depth_path=None, intrinsics_np=None, pcd_save=True
    ):
        """
        Executes the full data generation pipeline:
        Reconstruction -> Global Storage -> Sequence Calculation

        Args:
            input_path (str): Path to the input video file.
            condit_depth_path (str): Path to the conditional depth map (Pi3X only).
            intrinsics_np : 3x3 intrinsic matrix (Pi3X only).
            pcd_save (bool, optional): Whether to save 3D artifacts (point cloud, etc.). Defaults to True.

        Returns:
            None
        """

        # 3D Reconstruction
        pcd, self.camera_pose, self.norm_cam_ray = self.pcd_reconstruction(
            input_path, condit_depth_path, intrinsics_np
        )

        if not pcd_save:
            return

        print("Start processing sequence frames...")

        paths = self.get_io_paths(input_path)

        # Execute core computation
        arr_4d_occ, arr_4d_mask, all_camera_poses, all_camera_intrinsics = (
            self.compute_sequence_data(pcd)
        )

        # Save global data
        self.save_global_data(paths)

        # Save sequence data
        print("Saving 4D Sequence Arrays...")
        self.save_sequence_data(paths, arr_4d_occ, arr_4d_mask)


if __name__ == "__main__":
    occ_pcd_cam = np.load("./tmp.npy")
    occ_pcd_cam = torch.tensor(occ_pcd_cam, dtype=torch.float32, device="cuda")
    generetor = DataGenerator()
    generetor.norm_cam_ray = torch.tensor(
        np.load("./cam_ray.npy"), dtype=torch.float32, device="cuda"
    )
    st = time.time()

    for i in range(100):
        _, _ = generetor.check_visual_occ(occ_pcd_cam)
    et = time.time()

    def fill_time():
        occ_3d = torch.rand((400, 400, 400)).cuda()
        occ_3d[occ_3d >= 0.8] = 1
        occ_3d[occ_3d < 0.8] = 0

        occ_points = voxel2points(occ_3d)
        print(occ_points)
        occ_points = occ_points.int()
        tpl = torch.zeros_like(occ_3d)
        import time

        loop_number = 500
        st = time.time()
        for i in range(loop_number):
            tpl[occ_points[:, 0], occ_points[:, 1], occ_points[:, 2]] = 1
        et = time.time()
        print(f"index: {et - st}s")
        # occ_points = occ_points.contiguous()
        tpl = tpl.to(torch.uint8)
        for i in range(loop_number):
            # indices = (occ_points[:, 0], occ_points[:, 1], occ_points[:, 2])
            # values = torch.tensor(1.0, device=tpl.device)
            # tpl.index_put_(indices, values)

            D, H, W = tpl.shape
            idx_1d = (
                occ_points[:, 0] * (H * W) + occ_points[:, 1] * W + occ_points[:, 2]
            )
            idx_1d = idx_1d.long()
            tpl.view(-1).index_fill_(0, idx_1d, 1)
            # tpl.view(-1)[idx_1d] = True
        et2 = time.time()
        print(f"lat: {et2 - et}s")
        print("tpl type: ", tpl.dtype)
