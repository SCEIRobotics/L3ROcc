import fcntl
import glob
import json
import os
import time

import numpy as np
import pandas as pd

from L3ROcc.base import DataGenerator
from L3ROcc.utils import compute_similarity_transform


def _quat_wxyz_to_R(q):
    """(T, 4) (w, x, y, z) -> (T, 3, 3) rotation matrices. Vectorized."""
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


class InternNavDataGenerator(DataGenerator):
    """
    Data generator designed for the InternNav dataset.

    This class handles the pipeline of 3D reconstruction, scale alignment against
    ground truth, occupancy generation, and safe metadata updates using file locking.
    """

    def __init__(
        self,
        config_path,
        save_dir,
        model_dir,
        model_type="pi3x",
    ):
        """
        Initialize the InternNavDataGenerator.

        Args:
            config_path (str): Path to the YAML configuration file.
            save_dir (str): Root directory where outputs will be saved.
            model_dir (str): Directory containing model checkpoints.
            model_type (str): Which backbone to load. One of {"pi3", "pi3x"}.
                Pi3X consumes depth/intrinsic kwargs at the model layer; Pi3 is RGB-only
                at the forward pass but still honors any calibrated K at post-processing
                (the rescaled K overrides the saved Parquet intrinsic).
        """
        super().__init__(config_path, save_dir, model_dir, model_type=model_type)

    def check_processing_status(self, input_path, overwrite=False):
        """
        Check if the data needs to be processed.
        It verifies the existence of final target files, Parquet columns & lengths,
        and scale values in the episodes.jsonl.
        """
        if overwrite:
            print(
                f"[Status] Force Overwrite enabled. Running '{os.path.basename(input_path)}'."
            )
            return True

        paths = self.get_io_paths(input_path)
        mask_final_target = paths["mask_seq"]
        occ_final_target = paths["occ_seq"]

        if not os.path.exists(mask_final_target):
            return True
        if not os.path.exists(occ_final_target):
            return True

        parquet_path = paths.get("parquet")
        if not parquet_path or not os.path.exists(parquet_path):
            return True

        try:
            df = pd.read_parquet(parquet_path, engine="pyarrow")

            if (
                "observation.camera_extrinsic_occ" not in df.columns
                or "observation.camera_intrinsic_occ" not in df.columns
            ):
                print(
                    f"[Status] Missing OCC camera columns in parquet. Needs generation."
                )
                return True

            if "observation.camera_extrinsic" not in df.columns:
                print(
                    f"[Status] Base camera_extrinsic missing in parquet. Needs generation."
                )
                return True

            valid_base_ext = df["observation.camera_extrinsic"].dropna()
            valid_occ_ext = df["observation.camera_extrinsic_occ"].dropna()
            valid_occ_int = df["observation.camera_intrinsic_occ"].dropna()

            if len(valid_occ_ext) != len(valid_base_ext) or len(valid_occ_int) != len(
                valid_base_ext
            ):
                print(
                    f"[Status] Valid length mismatch between base extrinsic and OCC camera data. Needs generation."
                )
                return True

        except Exception as e:
            print(
                f"[Status] Error reading parquet for status check ({e}). Needs generation."
            )
            return True

        meta_dir = os.path.join(self.save_path, "meta")
        jsonl_path = os.path.join(meta_dir, "episodes.jsonl")

        if not os.path.exists(jsonl_path):
            return True

        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                entries = [json.loads(line) for line in lines if line.strip()]

                if not entries or any("scale" not in entry for entry in entries):
                    print(
                        f"[Status] Missing 'scale' key in episodes.jsonl. Needs generation."
                    )
                    return True
        except Exception as e:
            print(
                f"[Status] Error reading jsonl for status check ({e}). Needs generation."
            )
            return True

        print(
            f"[Status] Found '{os.path.basename(mask_final_target)}' '{os.path.basename(occ_final_target)}' and all metadata is valid. Skipping."
        )
        return False

    def get_io_paths(self, input_path):
        """
        Generates the directory structure and file paths required for dataset outputs.

        Args:
            input_path (str): Absolute path to the input video or data source.

        Returns:
            dict: A dictionary containing output file paths. Keys include:
                - 'ply': Path to the original point cloud (.ply).
                - 'global_occ': Path to the global occupancy file (.npz).
                - 'parquet': Path to the episode metadata (.parquet).
                - 'occ_seq': Path to the occupancy sequence (.npz).
                - 'mask_seq': Path to the mask sequence (.npz).
        """

        # 1. Construct directories
        data_chunk_dir = os.path.join(self.save_path, "data", "chunk-000")
        video_chunk_dir = os.path.join(self.save_path, "videos", "chunk-000")
        occ_view_dir = os.path.join(video_chunk_dir, "observation.occ.view")
        occ_mask_dir = os.path.join(video_chunk_dir, "observation.occ.mask")

        for d in [data_chunk_dir, video_chunk_dir, occ_view_dir, occ_mask_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

        # 2. Define file paths
        paths = {
            "ply": os.path.join(data_chunk_dir, "origin_pcd.ply"),
            "global_occ": os.path.join(data_chunk_dir, "all_occ.npz"),
            "parquet": os.path.join(data_chunk_dir, "episode_000000.parquet"),
            "occ_seq": os.path.join(occ_view_dir, "occ_sequence.npz"),
            "mask_seq": os.path.join(occ_mask_dir, "mask_sequence.npz"),
        }
        return paths

    def _locate_traj_parquet(self, input_path):
        """Locate the GT parquet file matching ``input_path``.

        Tries (in order): episode_<id>.parquet under traj_root (lerobot rosbag uses
        episode_000/001/...; InternData-N1 uses episode_000000), the same path under
        ``self.save_path``, any glob match of ``episode_*.parquet``, finally the legacy
        episode_000000.parquet. Returns the resolved path or None.
        """
        traj_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(input_path)))
        )
        episode_id = os.path.splitext(os.path.basename(input_path))[0]

        candidates = [
            os.path.join(traj_root, "data", "chunk-000", f"{episode_id}.parquet"),
            os.path.join(self.save_path, "data", "chunk-000", f"{episode_id}.parquet"),
            os.path.join(traj_root, "data", "chunk-000", "episode_000000.parquet"),
            os.path.join(self.save_path, "data", "chunk-000", "episode_000000.parquet"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c

        for root in (traj_root, self.save_path):
            hits = sorted(glob.glob(os.path.join(root, "data", "chunk-000", "episode_*.parquet")))
            if hits:
                return hits[0]
        return None

    def get_gt_poses(self, input_path):
        """
        Parses Ground Truth (GT) camera trajectories. Two parquet schemas are supported:

        A. InternData-N1: ``action`` column where each row stacks into a 4x4 SE(3) matrix —
           used directly as the camera pose in the world frame.

        B. lerobot rosbag: ``observation.state`` column (14-dim body odometry with quaternion);
           combined with the full hand-eye calibration (rotation ``R_cam2gripper`` and
           translation ``t_cam2gripper``) from ``meta/info.json``, the camera pose is
           reconstructed as ``R_world_cam = R_world_body @ R_cam2gripper`` and
           ``p_world_cam = R_world_body @ t_cam2gripper + p_body``. Including the hand-eye
           rotation is what gives the downstream alignment enough constraint to recover the
           camera mount tilt — without it, straight-line trajectories leave the roll/pitch
           around the trajectory direction unconstrained.

        Returns:
            np.ndarray or None: (N, 4, 4) camera poses in the GT world frame, else None.
        """
        try:
            parquet_path = self._locate_traj_parquet(input_path)
            if parquet_path is None:
                return None

            df = pd.read_parquet(parquet_path, engine="pyarrow")

            # --- Schema A: InternData-N1 action column = (4, 4) SE(3) ---
            if "action" in df.columns:
                gt_raw = df["action"].tolist()
                gt_poses_np = []
                for p in gt_raw:
                    if p is None:
                        continue
                    try:
                        mat = np.stack(p)
                    except Exception:
                        mat = np.asarray(p)
                    if mat.shape == (4, 4):
                        gt_poses_np.append(mat.astype(np.float64))
                if len(gt_poses_np) >= 1:
                    return np.array(gt_poses_np)

            # --- Schema B: lerobot rosbag observation.state (14-dim) + hand-eye ---
            if "observation.state" in df.columns:
                state = np.stack(df["observation.state"].values).astype(np.float64)
                if state.ndim != 2 or state.shape[1] < 10:
                    return None
                p_body = state[:, 0:3]
                quat_wxyz = state[:, 6:10]

                traj_root = os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.dirname(input_path)))
                )
                info_json_path = os.path.join(traj_root, "meta", "info.json")
                t_cam2body = np.zeros(3, dtype=np.float64)
                R_cam2body = np.eye(3, dtype=np.float64)
                if os.path.isfile(info_json_path):
                    with open(info_json_path, "r", encoding="utf-8") as f:
                        info = json.load(f)
                    ext = info.get("head_camera_extrinsic", {})
                    t_raw = ext.get("t_cam2gripper", ext.get("t_cam2robot", None))
                    if t_raw is not None:
                        t_cam2body = np.asarray(t_raw, dtype=np.float64).reshape(3)
                    else:
                        print(
                            f"[GT] info.json missing head_camera_extrinsic.t_cam2gripper "
                            f"({info_json_path}); GT degenerates to body-center trajectory."
                        )
                    R_raw = ext.get("R_cam2gripper", ext.get("R_cam2robot", None))
                    if R_raw is not None:
                        R_arr = np.asarray(R_raw, dtype=np.float64)
                        if R_arr.shape == (3, 3):
                            R_cam2body = R_arr
                        elif R_arr.size == 9:
                            R_cam2body = R_arr.reshape(3, 3)
                        else:
                            print(
                                f"[GT] Unexpected R_cam2gripper shape {R_arr.shape}; "
                                f"GT rotation degenerates to body orientation."
                            )
                    else:
                        print(
                            f"[GT] info.json missing head_camera_extrinsic.R_cam2gripper "
                            f"({info_json_path}); GT rotation degenerates to body orientation. "
                            f"Camera mount tilt cannot be recovered."
                        )
                else:
                    print(
                        f"[GT] meta/info.json not found ({info_json_path}); GT degenerates "
                        f"to body-center trajectory."
                    )

                from scipy.spatial.transform import Rotation as _Rot
                eu_c2b = _Rot.from_matrix(R_cam2body).as_euler("xyz", degrees=True)
                # Physical down-tilt: camera z (OpenCV forward) projected into body frame.
                # XYZ-Euler 'pitch' is misleading when |roll|>10° (the down-tilt hides in
                # the roll/yaw composition). The geometrically correct number is the angle
                # between camera z and the body horizontal plane.
                _z_cam_in_body = R_cam2body @ np.array([0.0, 0.0, 1.0])
                _horiz = float(np.sqrt(_z_cam_in_body[0] ** 2 + _z_cam_in_body[1] ** 2))
                tilt_down_deg = float(np.degrees(np.arctan2(-_z_cam_in_body[2], _horiz)))
                print(
                    f"[GT-diag] R_cam2gripper euler XYZ raw (deg): "
                    f"roll={eu_c2b[0]:.2f} pitch={eu_c2b[1]:.2f} yaw={eu_c2b[2]:.2f} "
                    f"(note: XYZ pitch ≠ physical tilt when |roll|>10°)"
                )
                print(
                    f"[GT-diag] camera forward (z) in body = {_z_cam_in_body.tolist()}; "
                    f"physical down-tilt = {tilt_down_deg:.2f} deg "
                    f"(expected ~20° for Go2 head-mount; <5° ⇒ install tilt missed; "
                    f"t_cam2body (m)={t_cam2body.tolist()})"
                )

                R_wb = _quat_wxyz_to_R(quat_wxyz)
                cam_pos = np.einsum("tij,j->ti", R_wb, t_cam2body) + p_body
                R_world_cam = np.einsum("tij,jk->tik", R_wb, R_cam2body)
                T = cam_pos.shape[0]
                gt_poses = np.zeros((T, 4, 4), dtype=np.float64)
                gt_poses[:, :3, :3] = R_world_cam
                gt_poses[:, :3, 3] = cam_pos
                gt_poses[:, 3, 3] = 1.0
                return gt_poses

            return None

        except Exception as e:
            print(f"[Subclass Error] Failed to load GT poses: {e}")
            return None

    def compute_trajectory_scale(self, poses_gt, poses_pred):
        """
        Computes the scale ratio (GT / Pred) between predicted and ground truth trajectories.
        Uses the ratio of standard deviations (Sim3 scale estimation).

        Args:
            poses_gt : Ground truth poses (N, 4, 4).
            poses_pred : Predicted poses (N, 4, 4).

        Returns:
            scale: The calculated scale factor. Returns 1.0 if calculation fails or input is invalid.
        """

        # def to_mat4x4(p):
        #     p = np.array(p)
        #     if p.ndim == 1:
        #         if p.size == 16:
        #             return p.reshape(4, 4)
        #         if p.size == 12:
        #             return np.vstack([p.reshape(3, 4), [0, 0, 0, 1]])
        #     return p

        traj_gt = np.array([p[:3, 3] for p in poses_gt])
        traj_pred = np.array([p[:3, 3] for p in poses_pred])

        # Ensure frame counts match
        n_frames = min(len(traj_gt), len(traj_pred))
        traj_gt = traj_gt[:n_frames]
        traj_pred = traj_pred[:n_frames]

        if n_frames < 5:
            print("Warning: Trajectory too short for scale estimation. Using scale=1.0")
            return 1.0

        # Sim3 Scale: Ratio of standard deviations
        gt_centered = traj_gt - np.mean(traj_gt, axis=0)
        pred_centered = traj_pred - np.mean(traj_pred, axis=0)

        std_gt = np.sqrt(np.mean(np.sum(gt_centered**2, axis=1)))
        std_pred = np.sqrt(np.mean(np.sum(pred_centered**2, axis=1)))

        # Avoid division by zero
        if std_pred < 1e-6 or np.isnan(std_pred) or np.isnan(std_gt):
            print(
                f"[Scale Warning] Invalid std detected (GT:{std_gt}, Pred:{std_pred}). Using scale=1.0"
            )
            return 1.0

        scale = std_gt / std_pred

        if np.isnan(scale) or np.isinf(scale):
            print(f"[Scale Warning] Calculated scale is NaN/Inf. Using 1.0")
            return 1.0

        print(
            f"[Scale Info] GT std: {std_gt:.4f}, Pred std: {std_pred:.4f} -> Scale: {scale:.4f}"
        )
        return scale

    def align_with_gt_scale(self, input_path, pcd):
        """
        Aligns the Pi3X reconstruction to the GT world frame via a two-stage Sim3 transform:

        1. **Rotation** (orthogonal Procrustes on the per-frame camera rotation columns):
           solve ``R = argmin Σ ||R · R_pred[t] - R_gt[t]||`` via SVD of
           ``M = Σ R_gt[t] · R_pred[t]^T``. This fully constrains all 3 rotation DOF and is
           independent of the trajectory geometry — crucial because real robot trajectories
           are often near-collinear, which leaves the rotation around the trajectory axis
           underdetermined when only translation columns are used.

        2. **Scale + translation** (on the translation columns under the already-known R):
           with ``pred_xyz_rot = pred_xyz @ R^T``, fit ``s, t`` so that
           ``s · pred_xyz_rot + t ≈ gt_xyz``.

        The composed (s, R, t) Sim3 is then applied in-place to ``self.pcd`` and
        ``self.camera_pose`` so downstream consumers (``save_global_data``,
        ``compute_sequence_data``) see already-aligned data. ``compute_sequence_data`` must
        therefore be called with ``scale=1.0`` (otherwise the OCC voxels would be scaled
        twice).

        Args:
            input_path : Path to the input data.
            pcd : The predicted point cloud (N, 3) in the Pi3X world frame.

        Returns:
            tuple:
                - pcd_aligned : (N, 3) point cloud transformed into the GT world frame.
                - scale       : The Sim3 scale factor applied (1.0 if alignment skipped).
        """
        try:
            gt_poses_np = self.get_gt_poses(input_path)

            if gt_poses_np is None or len(gt_poses_np) == 0:
                print(
                    "[Scale Info] No GT poses provided by subclass. Skipping alignment."
                )
                return pcd, 1.0

            cp_full = np.asarray(self.camera_pose, dtype=np.float64)
            gt_full = np.asarray(gt_poses_np, dtype=np.float64)
            n = min(len(cp_full), len(gt_full))
            if n < 5:
                print(
                    f"[Scale Info] Too few frames for Sim3 alignment ({n} < 5). "
                    f"Skipping alignment."
                )
                return pcd, 1.0

            pred_R = cp_full[:n, :3, :3]
            gt_R = gt_full[:n, :3, :3]
            pred_xyz = cp_full[:n, :3, 3]
            gt_xyz = gt_full[:n, :3, 3]

            # Stage 1: orthogonal Procrustes on per-frame rotation columns.
            M = np.einsum("tij,tkj->ik", gt_R, pred_R)
            U, _, Vt = np.linalg.svd(M)
            det_sign = np.sign(np.linalg.det(U @ Vt))
            D = np.diag([1.0, 1.0, det_sign if det_sign != 0 else 1.0])
            R = U @ D @ Vt

            from scipy.spatial.transform import Rotation as _Rot
            R_diff_0 = gt_R[0] @ pred_R[0].T
            eu_R = _Rot.from_matrix(R).as_euler("xyz", degrees=True)
            eu_diff0 = _Rot.from_matrix(R_diff_0).as_euler("xyz", degrees=True)
            frob_R_minus_diff0 = float(np.linalg.norm(R - R_diff_0))
            sv_M = np.linalg.svd(M, compute_uv=False)
            cond_M = float(sv_M[0] / max(sv_M[-1], 1e-12))
            print(
                f"[Align-diag] solved R euler XYZ (deg): "
                f"roll={eu_R[0]:.2f} pitch={eu_R[1]:.2f} yaw={eu_R[2]:.2f}"
            )
            print(
                f"[Align-diag] frame-0 (R_gt @ R_pred.T) euler XYZ (deg): "
                f"roll={eu_diff0[0]:.2f} pitch={eu_diff0[1]:.2f} yaw={eu_diff0[2]:.2f}"
            )
            print(
                f"[Align-diag] |R - R_diff_0|_F = {frob_R_minus_diff0:.4f} "
                f"(small ⇒ c2w convention consistent; >0.5 ⇒ likely transpose mismatch); "
                f"M singular values = {sv_M.tolist()}, cond(M) = {cond_M:.2e}"
            )

            # Stage 2: scale + translation on translation columns under the known R.
            pred_rot = pred_xyz @ R.T
            pred_c = pred_rot.mean(axis=0)
            gt_c = gt_xyz.mean(axis=0)
            num = float(((gt_xyz - gt_c) ** 2).sum())
            den = float(((pred_rot - pred_c) ** 2).sum())
            if den < 1e-12 or not np.isfinite(num) or not np.isfinite(den):
                print(
                    f"[Scale Warning] Degenerate translation spread (den={den:.3e}). "
                    f"Skipping alignment."
                )
                return pcd, 1.0
            s = float(np.sqrt(num / den))
            if not np.isfinite(s) or s <= 0:
                print(f"[Scale Warning] Invalid scale ({s}). Skipping alignment.")
                return pcd, 1.0
            t = gt_c - s * pred_c

            transformed = s * pred_rot + t
            rmse = float(np.sqrt(((transformed - gt_xyz) ** 2).sum(axis=1).mean()))
            # Rotation residual: median geodesic angle (deg) between R·R_pred and R_gt.
            R_resid = np.einsum("ij,tjk->tik", R, pred_R)
            cos_ang = np.clip(
                (np.einsum("tii->t", np.einsum("tij,tkj->tik", R_resid, gt_R)) - 1.0) / 2.0,
                -1.0, 1.0,
            )
            rot_err_deg = float(np.median(np.degrees(np.arccos(cos_ang))))

            print(
                f"[Scale Info] Sim3 alignment: scale={s:.4f}, RMSE={rmse:.4f} m, "
                f"rot residual median={rot_err_deg:.2f} deg (n={n} frames)"
            )

            pcd_np = np.asarray(pcd, dtype=np.float64)
            pcd_aligned = (s * (pcd_np @ R.T) + t).astype(np.float32)

            new_cp = cp_full.copy()
            new_cp[:, :3, 3] = s * (cp_full[:, :3, 3] @ R.T) + t
            new_cp[:, :3, :3] = np.einsum("ij,tjk->tik", R, cp_full[:, :3, :3])
            new_cp = new_cp.astype(np.float32)

            # P6: ground-plane RANSAC on aligned pcd, report tilt vs world Z.
            # P7: when tilt > threshold, apply Rodrigues gravity correction so the
            # dominant ground plane normal aligns with +Z (compensates the Unitree odom
            # Z-axis vs absolute gravity offset).
            tilt_pcd_deg = float("nan")
            try:
                import open3d as _o3d
                _pcd_o3d = _o3d.geometry.PointCloud()
                _pcd_o3d.points = _o3d.utility.Vector3dVector(pcd_aligned.astype(np.float64))
                _plane, _inliers = _pcd_o3d.segment_plane(
                    distance_threshold=0.05, ransac_n=3, num_iterations=300
                )
                n_ground = np.asarray(_plane[:3], dtype=np.float64)
                n_ground /= max(np.linalg.norm(n_ground), 1e-12)
                if n_ground[2] < 0:
                    n_ground = -n_ground
                tilt_pcd_deg = float(np.degrees(np.arccos(np.clip(n_ground[2], -1.0, 1.0))))
                inlier_frac = len(_inliers) / max(len(pcd_aligned), 1)
                print(
                    f"[Pcd-diag] aligned-pcd ground plane: normal={n_ground.tolist()}, "
                    f"inlier_frac={inlier_frac:.3f}, tilt vs world Z = {tilt_pcd_deg:.2f} deg"
                )

                # Threshold for triggering gravity correction.
                # Empirical: real lerobot rosbags show tilt_pcd_deg ≈ 4-5°
                # (Unitree odom Z is initialized from startup pose, drifts ~3-4°
                # from absolute gravity; combined with calibration + Procrustes
                # residuals gives ~4-5° total). N1 synthetic data is gravity-aligned
                # by construction so tilt_pcd_deg ≈ 0-1° and P7 is automatically
                # skipped. 2.0° leaves 1° margin above noise floor for lerobot
                # while still skipping N1.
                GRAV_TILT_THRESH_DEG = 2.0
                if tilt_pcd_deg > GRAV_TILT_THRESH_DEG:
                    target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                    axis = np.cross(n_ground, target)
                    s_axis = float(np.linalg.norm(axis))
                    c_axis = float(np.dot(n_ground, target))
                    if s_axis > 1e-6:
                        k_unit = axis / s_axis
                        K_skew = np.array(
                            [[0.0, -k_unit[2], k_unit[1]],
                             [k_unit[2], 0.0, -k_unit[0]],
                             [-k_unit[1], k_unit[0], 0.0]],
                            dtype=np.float64,
                        )
                        R_grav = np.eye(3) + s_axis * K_skew + (1.0 - c_axis) * (K_skew @ K_skew)
                        pivot = new_cp[0, :3, 3].astype(np.float64)

                        pcd_g = (R_grav @ (pcd_aligned.astype(np.float64) - pivot).T).T + pivot
                        pcd_aligned = pcd_g.astype(np.float32)

                        cp_t = (R_grav @ (new_cp[:, :3, 3].astype(np.float64) - pivot).T).T + pivot
                        cp_R = np.einsum(
                            "ij,tjk->tik", R_grav, new_cp[:, :3, :3].astype(np.float64)
                        )
                        new_cp[:, :3, 3] = cp_t.astype(np.float32)
                        new_cp[:, :3, :3] = cp_R.astype(np.float32)

                        # Re-fit ground for verification.
                        tilt_after = float("nan")
                        try:
                            _pcd_o3d2 = _o3d.geometry.PointCloud()
                            _pcd_o3d2.points = _o3d.utility.Vector3dVector(
                                pcd_aligned.astype(np.float64)
                            )
                            _pm2, _ = _pcd_o3d2.segment_plane(
                                distance_threshold=0.05, ransac_n=3, num_iterations=300
                            )
                            n2 = np.asarray(_pm2[:3], dtype=np.float64)
                            n2 /= max(np.linalg.norm(n2), 1e-12)
                            if n2[2] < 0:
                                n2 = -n2
                            tilt_after = float(np.degrees(np.arccos(np.clip(n2[2], -1.0, 1.0))))
                        except Exception as _e2:
                            print(f"[Pcd-fix] post-fix RANSAC failed: {_e2}")
                        print(
                            f"[Pcd-fix] gravity correction: rotated pcd+cam by "
                            f"{tilt_pcd_deg:.2f} deg around pivot={pivot.tolist()} "
                            f"(axis={k_unit.tolist()}); post-fix ground tilt vs Z = "
                            f"{tilt_after:.2f} deg."
                        )
                    else:
                        print(
                            f"[Pcd-fix] skipped: ground normal {n_ground.tolist()} already ≈ +Z "
                            f"(|axis|={s_axis:.4f} below floating-point threshold)."
                        )
                else:
                    print(
                        f"[Pcd-fix] skipped: tilt={tilt_pcd_deg:.2f}° ≤ "
                        f"{GRAV_TILT_THRESH_DEG}° threshold (GT Z already gravity-aligned)."
                    )
            except Exception as _e:
                print(f"[Pcd-diag] ground plane RANSAC / gravity fix failed: {_e}")

            self.camera_pose = new_cp
            self.pcd = pcd_aligned

            return pcd_aligned, s

        except Exception as e:
            print(f"[Scale Error] Exception during alignment: {e}")
            return pcd, 1.0

    def _gravity_R_to_z(self, pcd, cam_centers):
        """地面 RANSAC 估法向，用"相机在地面之上"消歧符号，返回把地面法向转到 +Z 的 Rodrigues
        旋转 R_grav 与纠前 tilt（度）。被 ``_gravity_align_to_z``（world 系）与
        ``_fold_gravity_into_tcam2base``（base 系）复用。

        Args:
            pcd : (N, 3) 点云
            cam_centers : (T, 3) 相机中心（用于法向符号消歧）

        Returns:
            (R_grav (3,3) f64, tilt_before_deg)
        """
        import open3d as o3d

        pcd64 = np.asarray(pcd, dtype=np.float64)
        if pcd64.shape[0] < 100:
            raise ValueError(
                f"[gravity] too few points for ground RANSAC: {pcd64.shape[0]}"
            )

        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pcd64)
        plane, inliers = o3d_pcd.segment_plane(
            distance_threshold=0.05, ransac_n=3, num_iterations=300
        )
        # 法向符号用"相机在地面之上"消歧（物理上相机挂在机器人本体、在地面之上）。
        # 不能用 n[2]>0：Pi3X/OpenCV 帧0 系里 +Z 是相机前向(近水平)，该方向符号任意/含噪，
        # 会把整个场景旋成上下颠倒。改为：法向应指向相机一侧。
        a, b, c, d = float(plane[0]), float(plane[1]), float(plane[2]), float(plane[3])
        n_ground = np.array([a, b, c], dtype=np.float64)
        cc = np.asarray(cam_centers, dtype=np.float64).reshape(-1, 3)
        mean_signed = float(np.mean(cc @ n_ground + d))  # 相机相对平面平均有符号距离
        if mean_signed < 0:
            n_ground = -n_ground
        n_ground /= max(np.linalg.norm(n_ground), 1e-12)
        tilt_before = float(np.degrees(np.arccos(np.clip(n_ground[2], -1.0, 1.0))))
        inlier_frac = len(inliers) / max(pcd64.shape[0], 1)
        print(
            f"[gravity] ground plane normal={n_ground.tolist()} "
            f"inlier_frac={inlier_frac:.3f} tilt vs +Z = {tilt_before:.2f} deg"
        )
        if inlier_frac < 0.05:
            raise ValueError(
                f"[gravity] ground RANSAC inlier_frac too low ({inlier_frac:.3f}); "
                f"cannot estimate gravity reliably"
            )

        target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        axis = np.cross(n_ground, target)
        s_axis = float(np.linalg.norm(axis))
        c_axis = float(np.dot(n_ground, target))
        if s_axis <= 1e-6:
            # 已与 +Z 对齐（反向情形上面已翻正）；无需旋转
            R_grav = np.eye(3, dtype=np.float64)
        else:
            k_unit = axis / s_axis
            K_skew = np.array(
                [[0.0, -k_unit[2], k_unit[1]],
                 [k_unit[2], 0.0, -k_unit[0]],
                 [-k_unit[1], k_unit[0], 0.0]],
                dtype=np.float64,
            )
            R_grav = np.eye(3) + s_axis * K_skew + (1.0 - c_axis) * (K_skew @ K_skew)
        return R_grav, tilt_before

    def _fold_gravity_into_tcam2base(self, pcd, T_cam2base):
        """Lerobot(实采, OpenCV 外参) 专用：复现实验的 base 系 Z 倾斜重力纠偏并**折进外参旋转**。

        在帧0 base 系（world->cam0->base，Lerobot 约定 C=identity）估计把地面法向转到 +Z 的 R_deskew，
        折进 ``T_cam2base`` 旋转（R_eff = R_deskew @ R_c2b），从而对**所有帧**一致地纠正 OCC/base 的轻微下倾，
        与 exp_coord_align 帧0 结果逐点一致。绕 base 原点（=帧0相机）旋转，保持轨迹起点不动。
        失败（点太少 / RANSAC 不可靠）则告警并原样返回，跳过纠偏。
        """
        try:
            cam0 = self.camera_pose[0]
            p_base0 = self.convert_pointcloud_camera_to_base(
                self.convert_pointcloud_world_to_camera(pcd, cam0), T_cam2base
            )
            cc_base0 = self.convert_pointcloud_camera_to_base(
                self.convert_pointcloud_world_to_camera(
                    np.asarray(self.camera_pose)[:, :3, 3], cam0
                ),
                T_cam2base,
            )
            R_deskew, tilt_before = self._gravity_R_to_z(p_base0, cc_base0)
        except Exception as e:
            print(f"[gravity-base] lerobot Z-tilt deskew skipped: {e}")
            return T_cam2base
        T_eff = np.asarray(T_cam2base, dtype=np.float32).copy()
        T_eff[:3, :3] = (R_deskew @ T_eff[:3, :3].astype(np.float64)).astype(np.float32)
        print(
            f"[gravity-base] lerobot Z-tilt deskew folded into T_cam2base: "
            f"tilt {tilt_before:.2f} deg -> ~0"
        )
        return T_eff

    def _gravity_align_to_z(self, pcd, camera_pose, pivot):
        """用地面 RANSAC 估重力方向，构造把地面法向转到 +Z 的 Rodrigues 旋转，绕 pivot
        同步旋转点云与相机位姿。

        Args:
            pcd : (N, 3) 点云（将被旋转）
            camera_pose : (T, 4, 4) 相机位姿（将被旋转）
            pivot : (3,) 旋转中心（通常帧 0 相机位置）

        Returns:
            (pcd_rot (N,3) f32, camera_pose_rot (T,4,4) f32, tilt_before_deg,
             tilt_after_deg, ground_z)  —— ground_z 为旋转后地面在世界系的 z 电平
        """
        import open3d as o3d

        pcd64 = np.asarray(pcd, dtype=np.float64)
        cam_centers = np.asarray(camera_pose, dtype=np.float64)[:, :3, 3]
        R_grav, tilt_before = self._gravity_R_to_z(pcd64, cam_centers)
        pivot = np.asarray(pivot, dtype=np.float64).reshape(3)

        cp = np.asarray(camera_pose, dtype=np.float64).copy()
        pcd_rot = (R_grav @ (pcd64 - pivot).T).T + pivot
        cp[:, :3, 3] = (R_grav @ (cp[:, :3, 3] - pivot).T).T + pivot
        cp[:, :3, :3] = np.einsum("ij,tjk->tik", R_grav, cp[:, :3, :3])

        # 旋转后复测地面：验证 tilt，并取地面 z 电平 ground_z（地面水平后处处同 z）。
        o3d_pcd2 = o3d.geometry.PointCloud()
        o3d_pcd2.points = o3d.utility.Vector3dVector(pcd_rot)
        plane2, _ = o3d_pcd2.segment_plane(
            distance_threshold=0.05, ransac_n=3, num_iterations=300
        )
        c2 = float(plane2[2])
        if abs(c2) < 1e-6:
            raise ValueError(
                f"[gravity] post-correction ground plane not horizontal (c2={c2:.3e}); "
                f"gravity alignment failed"
            )
        ground_z = float(-float(plane2[3]) / c2)  # 平面 a x+b y+c z+d=0 在水平时的 z 电平
        n2 = np.asarray(plane2[:3], dtype=np.float64)
        n2 /= max(np.linalg.norm(n2), 1e-12)
        if n2[2] < 0:
            n2 = -n2
        tilt_after = float(np.degrees(np.arccos(np.clip(n2[2], -1.0, 1.0))))
        print(
            f"[gravity] post-correction ground tilt vs +Z = {tilt_after:.2f} deg, "
            f"ground_z = {ground_z:.3f}"
        )
        return (
            pcd_rot.astype(np.float32),
            cp.astype(np.float32),
            tilt_before,
            tilt_after,
            ground_z,
        )

    def align_to_world(self, pcd):
        """GT-free 把 Pi3X 重建对齐到真实世界系（z 朝上、帧 0 规范原点/朝向）。

        与 ``align_with_gt_scale`` 不同：不读任何 GT odom。依据：
          - 尺度：直接信任 Pi3X metric_head（已实验验证），外加 config 可选修正系数
            ``self.metric_scale_correction``（默认 1.0）。
          - 重力(roll/pitch)：地面 RANSAC 估法向 -> +Z（_gravity_align_to_z）。
            注意 Pi3X 世界系锚定帧 0 相机(OpenCV)，相机已在该坐标系，故地面法向
            即可恢复全局重力，无需GT。
          - 原点/朝向(yaw)：帧 0 规范——原点取**帧 0 相机正下方的地面**（x,y=帧0相机、z=地面电平），
            使地面 z=0、相机在 +高度(~0.6)、重建点云多为正值（符合"本体在地面上"）；
            绕 +Z 旋转使帧 0 前向投影到 +X。OCC 为 ego 系、对全局 yaw/平移不变，此步仅为
            全局可视化与落盘位姿的确定性。

        就地更新 self.pcd / self.camera_pose；返回 (pcd_aligned, scale)。
        """
        if self.camera_pose is None or len(self.camera_pose) == 0:
            raise ValueError("[align_to_world] camera_pose is empty")

        pcd_np = np.asarray(pcd, dtype=np.float64)
        cp = np.asarray(self.camera_pose, dtype=np.float64).copy()

        # 1) 尺度（config 可选修正系数；默认 1.0 = 直接信任 metric_head）
        s = float(self.metric_scale_correction)
        if not np.isfinite(s) or s <= 0:
            raise ValueError(f"[align_to_world] invalid metric_scale_correction: {s}")
        pcd_np = pcd_np * s
        cp[:, :3, 3] = cp[:, :3, 3] * s

        # 2) 重力对齐（绕帧 0）
        pivot = cp[0, :3, 3].copy()
        pcd_g, cp_g, tilt_b, tilt_a, ground_z = self._gravity_align_to_z(
            pcd_np, cp, pivot
        )
        pcd_np = np.asarray(pcd_g, dtype=np.float64)
        cp = np.asarray(cp_g, dtype=np.float64)

        # 3) yaw 规范：帧 0 前向(OpenCV z 轴在世界系)投影到 XY，旋到 +X
        pivot = cp[0, :3, 3].copy()
        fwd = cp[0, :3, 2].copy()
        fwd[2] = 0.0
        fwd_norm = float(np.linalg.norm(fwd))
        if fwd_norm > 1e-6:
            fwd /= fwd_norm
            yaw = float(np.arctan2(fwd[1], fwd[0]))
            c, sN = np.cos(-yaw), np.sin(-yaw)
            Rz = np.array(
                [[c, -sN, 0.0], [sN, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
            )
            pcd_np = (Rz @ (pcd_np - pivot).T).T + pivot
            cp[:, :3, 3] = (Rz @ (cp[:, :3, 3] - pivot).T).T + pivot
            cp[:, :3, :3] = np.einsum("ij,tjk->tik", Rz, cp[:, :3, :3])

        # 4) 原点：x,y 取帧 0 相机、z 取地面电平 -> 地面 z=0、相机在 +高度、点云多为正
        origin = np.array(
            [cp[0, 0, 3], cp[0, 1, 3], ground_z], dtype=np.float64
        )
        pcd_np = pcd_np - origin
        cp[:, :3, 3] = cp[:, :3, 3] - origin

        pcd_aligned = pcd_np.astype(np.float32)
        self.pcd = pcd_aligned
        self.camera_pose = cp.astype(np.float32)
        cam0_h = float(cp[0, 2, 3])
        print(
            f"[align_to_world] GT-free: scale={s:.4f}, ground tilt "
            f"{tilt_b:.2f}->{tilt_a:.2f} deg, origin->frame-0 foot on ground "
            f"(ground z=0, cam0 height={cam0_h:.3f} m), yaw canonicalized."
        )
        return pcd_aligned, s

    def update_metadata(
        self, paths, all_camera_poses, all_camera_intrinsics, input_path
    ):
        """
        Updates Parquet and JSON metadata files with generated camera parameters.

        Args:
            paths (dict): Dictionary of file paths (output of `get_io_paths`).
            all_camera_poses (list or np.ndarray): Generated camera extrinsic matrices (N, 4, 4).
            all_camera_intrinsics (list or np.ndarray): Camera intrinsic matrices (N, 3, 3).
            input_path (str): Path to the input source, used to locate the root JSON info.

        Returns:
            None: This method modifies files on disk (Side Effect).
        """

        # --- Update Parquet (Per trajectory) ---
        parquet_path = paths.get("parquet")
        if parquet_path and os.path.exists(parquet_path):
            print(f"Updating Parquet: {parquet_path}")
            try:
                df = pd.read_parquet(parquet_path, engine="pyarrow")
                curr_len = len(df)
                gen_len = len(all_camera_poses)

                # Validate data consistency
                if gen_len != curr_len:
                    raise ValueError(
                        f"[Length Mismatch] Parquet has {curr_len} frames, "
                        f"but generated poses have {gen_len} frames."
                    )

                df["observation.camera_extrinsic_occ"] = all_camera_poses
                df["observation.camera_intrinsic_occ"] = all_camera_intrinsics

                df.to_parquet(parquet_path, engine="pyarrow")
                print("Parquet updated.")

            except Exception as e:
                print(f"Parquet update failed: {e}")
                raise e
        else:
            print(f"Parquet file not found at {parquet_path}")

        # --- Update JSON ---
        traj_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(input_path)))
        )
        json_path = os.path.join(traj_root, "meta", "info.json")

        def update_info_logic(meta):
            feat_ext = {
                "dtype": "float32",
                "shape": [4, 4],
                "names": [f"extrinsic_{i}_{j}" for i in range(4) for j in range(4)],
            }
            feat_int = {
                "dtype": "float32",
                "shape": [3, 3],
                "names": [f"intrinsic_{i}_{j}" for i in range(3) for j in range(3)],
            }

            if "features" in meta:
                meta["features"]["observation.camera_extrinsic_occ"] = feat_ext
                meta["features"]["observation.camera_intrinsic_occ"] = feat_int
                return meta
            return None

        if os.path.exists(json_path):
            print(f"Updating JSON Safely: {json_path}")
            self._update_json_safely(json_path, update_info_logic)
        else:
            print(f"JSON path does not exist: {json_path}")

    def update_meta_episodes_jsonl(self, scale):
        """
        Updates `meta/episodes.jsonl` with the calculated scale value.
        Uses file locking (fcntl) to ensure safe concurrent writes.

        Args:
            scale (float): The calculated scale factor to be saved.

        Returns:
            None: Modifies the jsonl file on disk.
        """
        import json

        meta_dir = os.path.join(self.save_path, "meta")
        jsonl_path = os.path.join(meta_dir, "episodes.jsonl")

        if not os.path.exists(jsonl_path):
            print(f"[Meta Warning] episodes.jsonl not found at: {jsonl_path}")
            return

        # Retry mechanism for acquiring lock
        for _ in range(5):
            try:
                with open(jsonl_path, "r+", encoding="utf-8") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)

                    try:
                        lines = f.readlines()
                        entries = [json.loads(line) for line in lines if line.strip()]

                        scale_val = float(scale)
                        for entry in entries:
                            entry["scale"] = scale_val

                        f.seek(0)
                        f.truncate()
                        for entry in entries:
                            f.write(json.dumps(entry) + "\n")
                        f.flush()
                        os.fsync(f.fileno())

                        print(
                            f"[Meta Updated] Saved scale ({scale_val:.4f}) to {jsonl_path}"
                        )

                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)

                break

            except BlockingIOError:
                time.sleep(0.1)
            except Exception as e:
                print(f"[Meta Error] Failed to update episodes.jsonl: {e}")
                raise e

    def _update_json_safely(self, file_path, update_func):
        """
        Generic helper for safe JSON updates using file locking.
        Prevents race conditions in multi-process environments.

        Args:
            file_path (str): Path to the JSON file to update.
            update_func (callable): A function that takes the current dict data
                                    and returns the modified dict. If it returns None,
                                    no write occurs.

        Returns:
            None
        """
        if not os.path.exists(file_path):
            return

        for _ in range(5):
            try:
                with open(file_path, "r+", encoding="utf-8") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)

                    try:
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError:
                            data = {}

                        new_data = update_func(data)

                        if new_data is not None:
                            f.seek(0)
                            f.truncate()
                            json.dump(new_data, f, indent=4)
                            f.flush()
                            os.fsync(f.fileno())

                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)

                break

            except BlockingIOError:
                time.sleep(0.1)
            except Exception as e:
                print(f"Error updating {file_path}: {e}")
                break

    def run_pipeline(
        self,
        input_path,
        condit_depth_path=None,
        intrinsics_np=None,
        pcd_save=True,
        overwrite=False,
        mesh=False,
        T_cam2base=None,
        extrinsic_convention=None,
    ):
        """
        Executes the full data generation pipeline:
        Reconstruction -> GT Scale Alignment -> Global Storage -> Sequence Calculation -> Metadata Update

        Args:
            input_path (str): Path to the input video file.
            condit_depth_path (str, optional): Path to a depth video (Pi3X conditioning input).
                Ignored by the model when ``model_type='pi3'``. Defaults to None.
            intrinsics_np (np.ndarray, optional): 3x3 intrinsic matrix at the ORIGINAL video
                resolution. When provided, it is rescaled to the model input size and used
                both for Pi3X conditioning (if multimodal) and to override the DLT-estimated K
                in the saved ``observation.camera_intrinsic_occ`` column. Defaults to None.
            pcd_save (bool, optional): Whether to save 3D artifacts (point cloud, etc.). Defaults to True.
            overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
            mesh (bool, optional): Whether to use mesh instead of origin point cloud. Defaults to False.
            T_cam2base (np.ndarray, optional): 4x4 transformation matrix from camera to base coordinate system. Defaults to None.
            extrinsic_convention (str, optional): ``T_cam2base`` 的相机约定，决定 OpenCV(Pi3X)->外参约定
                的换基：``"opengl"``(N1 渲染外参) 施加 C=diag(1,-1,-1)；``"opencv"``(lerobot 实采手眼) / None
                不翻转。由 ``InternNavSequenceLoader.get_trajectory_info`` 产出并透传。Defaults to None.

        Returns:
            None
        """
        # Check Status
        if not self.check_processing_status(input_path, overwrite=overwrite):
            return

        # 3D Reconstruction
        pcd, self.camera_pose, self.norm_cam_ray = self.pcd_reconstruction(
            input_path, condit_depth_path, intrinsics_np
        )

        # 不再用 align_to_world 做坐标对齐（与 exp_coord_align 一致）：坐标对齐由 compute_sequence_data
        # 的相机约定换基 C（按 extrinsic_convention：N1=OpenGL 折 diag(1,-1,-1)、lerobot=OpenCV 不翻转）完成。
        # 尺度：信任 metric_head + config 修正系数（沿用 align_to_world 的尺度处理，就地缩放 pcd 与相机平移）。
        # 旧 align_to_world / align_with_gt_scale / get_gt_poses 保留供离线 GT 诊断，pipeline 不再调用。
        s = float(self.metric_scale_correction)
        pcd = pcd * s
        self.pcd = pcd
        self.camera_pose[:, :3, 3] = self.camera_pose[:, :3, 3] * s
        print(f"[Scale Info] no align_to_world; scale={s:.4f} (metric_scale_correction)")

        # Lerobot(实采, OpenCV 外参)：仅对 Z 轴轻微下倾做重力纠偏，把地面法向->+Z 的 R_deskew
        # 折进 T_cam2base(base 系)，对所有帧一致纠偏。N1(OpenGL) 不纠偏。
        if extrinsic_convention == "opencv" and T_cam2base is not None:
            T_cam2base = self._fold_gravity_into_tcam2base(pcd, T_cam2base)

        self.update_meta_episodes_jsonl(s)

        print("Start processing sequence frames...")

        paths = self.get_io_paths(input_path)

        # Execute core computation. 尺度已就地施加到 pcd / camera_pose，故 OCC/voxel 阶段不再缩放(scale=1.0)。
        arr_4d_occ, arr_4d_mask, all_camera_poses, all_camera_intrinsics = (
            self.compute_sequence_data(
                pcd,
                mesh=mesh,
                T_cam2base=T_cam2base,
                scale=1.0,
                extrinsic_convention=extrinsic_convention,
            )
        )

        # Save sequence data
        print("Saving 4D Sequence Arrays...")
        self.save_sequence_data(paths, arr_4d_occ, arr_4d_mask)

        # Update metadata
        self.update_metadata(paths, all_camera_poses, all_camera_intrinsics, input_path)

        if pcd_save:
            # Save global data
            self.save_global_data(paths)
