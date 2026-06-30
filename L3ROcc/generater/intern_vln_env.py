import fcntl
import glob
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import open3d as o3d

from L3ROcc.base import DataGenerator
from L3ROcc.utils import (
    align_reconstruction_to_world,
    gravity_R_to_z,
    gravity_align_to_z,
    voxels_to_pcd,
)


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

    This class handles the pipeline of 3D reconstruction, metric-scale handling, optional
    Lerobot z-axis deskew, occupancy generation, and safe metadata updates using file locking.
    The OCC data chain is GT-free (ego/base frame, invariant to the global world frame). The saved
    ``camera_extrinsic_occ`` is ALWAYS in the **OpenGL camera convention** (C=diag(1,-1,-1), the
    downstream model input requirement); only the **anchoring** differs per dataset (see
    ``_align_camera_poses_to_gt_world``):
      - N1 (opengl): **frame-0 anchored into the dataset GT world** (Method A) — the frame-0 GT pose
        ``A0`` (from ``action``, Schema A) pins origin/orientation; Pi3X's relative motion + scale
        carry later frames, ``aligned[0]==A0``. (verified correct.)
      - Lerobot (opencv): **frame-0 CAMERA anchored** (``A0=I`` -> ``C @ inv(P0) @ P[i] @ C``, OpenGL
        axes, ``aligned[0]==I`` since C@C=I), NOT the odom GT world. The odom+hand-eye GT (Schema B)
        is reconstructed for diagnostics only.
    When GT is unavailable (or opencv) the saved poses use frame-0 normalization (model frame-0 ->
    identity, ego-canonical), still in OpenGL axes.
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

    def _check_processing_status(self, input_path, overwrite=False):
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
                    "[Status] Missing OCC camera columns in parquet. Needs generation."
                )
                return True

            if "observation.camera_extrinsic" not in df.columns:
                print(
                    "[Status] Base camera_extrinsic missing in parquet. Needs generation."
                )
                return True

            valid_base_ext = df["observation.camera_extrinsic"].dropna()
            valid_occ_ext = df["observation.camera_extrinsic_occ"].dropna()
            valid_occ_int = df["observation.camera_intrinsic_occ"].dropna()

            if len(valid_occ_ext) != len(valid_base_ext) or len(valid_occ_int) != len(
                valid_base_ext
            ):
                print(
                    "[Status] Valid length mismatch between base extrinsic and OCC camera data. Needs generation."
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
                        "[Status] Missing 'scale' key in episodes.jsonl. Needs generation."
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
                - 'ply': Path to the downsampled reconstructed point cloud (.ply).
                - 'global_occ': Path to the last-frame occupancy point cloud (.npz).
                - 'parquet': Path to the episode metadata (.parquet).
                - 'occ_seq': Path to the occupancy sequence (.npz).
                - 'mask_seq': Path to the mask sequence (.npz).
                - 'meta_dir': Path to the output meta/ directory.
                - 'meta_info_json': Path to the output meta/info.json.
                - 'meta_episodes_jsonl': Path to the output meta/episodes.jsonl.
        """

        # 1. Construct directories
        data_chunk_dir = os.path.join(self.save_path, "data", "chunk-000")
        video_chunk_dir = os.path.join(self.save_path, "videos", "chunk-000")
        occ_view_dir = os.path.join(video_chunk_dir, "observation.occ.view")
        occ_mask_dir = os.path.join(video_chunk_dir, "observation.occ.mask")
        meta_dir = os.path.join(self.save_path, "meta")

        for d in (data_chunk_dir, video_chunk_dir, occ_view_dir, occ_mask_dir, meta_dir):
            os.makedirs(d, exist_ok=True)

        # 2. Define file paths
        paths = {
            "ply": os.path.join(data_chunk_dir, "downsampled_pcd.ply"),
            "global_occ": os.path.join(data_chunk_dir, "last_frame_occ.npz"),
            "parquet": os.path.join(data_chunk_dir, "episode_000000.parquet"),
            "occ_seq": os.path.join(occ_view_dir, "occ_sequence.npz"),
            "mask_seq": os.path.join(occ_mask_dir, "mask_sequence.npz"),
            "meta_dir": meta_dir,
            "meta_info_json": os.path.join(meta_dir, "info.json"),
            "meta_episodes_jsonl": os.path.join(meta_dir, "episodes.jsonl"),
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

    def _seed_source_metadata(self, input_path, paths, overwrite=False):
        """Seed the (separate) output dir with the source dataset's records so the
        in-place ``update_*`` steps have something to augment.

        Copies the source trajectory's ``meta/`` folder and its base
        ``episode_*.parquet`` into the output, renaming the parquet to the canonical
        ``episode_000000.parquet``. The output keeps the lerobot/N1 originals
        (tasks.jsonl, episodes_stats.jsonl, info.json, base camera columns), onto which
        ``update_metadata`` / ``update_meta_episodes_jsonl`` later add OCC columns,
        ``scale`` and OCC features. Missing sources warn but do not raise.

        Args:
            input_path (str): Input video path (same basis as ``_locate_traj_parquet``).
            paths (dict): Output paths from ``get_io_paths`` (uses ``meta_dir`` / ``parquet``).
            overwrite (bool): When False, an already-seeded output file/dir is left as is.
        """
        traj_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(input_path)))
        )

        # --- meta/ folder ---
        src_meta = os.path.join(traj_root, "meta")
        dst_meta = paths["meta_dir"]
        if os.path.isdir(src_meta) and os.path.abspath(src_meta) != os.path.abspath(
            dst_meta
        ):
            try:
                shutil.copytree(src_meta, dst_meta, dirs_exist_ok=True)
                print(f"[seed] copied source meta/ -> {dst_meta}")
            except Exception as e:
                print(f"[seed] failed to copy source meta/ ({e}); continuing.")
        elif not os.path.isdir(src_meta):
            print(f"[seed] source meta/ not found at {src_meta}; skipping meta seed.")

        # --- base episode parquet ---
        src_parquet = self._locate_traj_parquet(input_path)
        dst_parquet = paths["parquet"]
        if (
            src_parquet
            and os.path.abspath(src_parquet) != os.path.abspath(dst_parquet)
            and (overwrite or not os.path.exists(dst_parquet))
        ):
            try:
                shutil.copy2(src_parquet, dst_parquet)
                print(f"[seed] copied source parquet -> {dst_parquet}")
            except Exception as e:
                print(f"[seed] failed to copy source parquet ({e}); continuing.")
        elif not src_parquet:
            print("[seed] source episode parquet not found; skipping parquet seed.")

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
                gt_poses_np = []
                for p in df["action"]:
                    if p is None:
                        continue
                    # N1 parquet stores each action cell as an object array of 4 row-vectors
                    # (dtype=object, shape (4,)), not a regular (4, 4) array, so a direct
                    # np.asarray(p, dtype=float64) raises "setting an array element with a
                    # sequence". Parse row-by-row (same idea as adapter's _reshape_matrix).
                    try:
                        mat = np.array([np.asarray(r, dtype=np.float64) for r in p])
                    except Exception:
                        continue
                    if mat.shape != (4, 4) and mat.size == 16:
                        mat = mat.reshape(4, 4)
                    if mat.shape == (4, 4):
                        gt_poses_np.append(mat)
                if gt_poses_np:
                    return np.array(gt_poses_np)

            # --- Schema B: lerobot rosbag observation.state (14-dim) + hand-eye ---
            if "observation.state" in df.columns:
                # Be robust to object-array cells (each row a sequence) like Schema A above.
                try:
                    state = np.array(
                        [np.asarray(r, dtype=np.float64) for r in df["observation.state"]]
                    )
                except Exception:
                    return None
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
                        if R_arr.size == 9:
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
            raise

    def _camera_opengl_bridge(self):
        """Camera-axis basis change M (4x4) OpenCV->OpenGL ``C=diag(1,-1,-1)`` in the top-left,
        **dataset-independent**. Pi3X always outputs OpenCV camera poses, and the saved
        ``camera_extrinsic_occ`` (the downstream model input) is required in the OpenGL camera
        convention for BOTH datasets, so the bridge is the same regardless of
        ``extrinsic_convention``. The per-dataset difference lives in the ANCHORING (N1 -> GT
        world, Lerobot -> frame-0 camera; see ``run_pipeline``), not in this bridge. Used by
        ``_align_camera_poses_to_gt_world`` (both inside the frame-0 map ``A0 @ C @ inv(P0)`` and
        as the per-pose right factor) and the ``infer_cam_traj_R.ply`` diagnostic, so the single
        matrix mapping is not duplicated.
        """
        C4 = np.eye(4, dtype=np.float64)
        C4[:3, :3] = self.R_opencv_to_opengl
        return C4

    def _align_camera_poses_to_gt_world(self, camera_pose, gt_poses):
        """Place the Pi3X camera poses into the saved frame by **frame-0 anchoring** (Method A),
        always emitting the **OpenGL camera convention** (``C=diag(1,-1,-1)``, the downstream model
        input requirement for both datasets; see ``_camera_opengl_bridge``).

        For each frame ``i`` (``P=camera_pose``, ``C``=OpenCV->OpenGL bridge)::

            M_left   = A0 @ C @ inv(P0)               # model-world -> saved-frame rigid (frame-0)
            aligned[i] = M_left @ P[i] @ C = A0 @ C @ inv(P0) @ P[i] @ C

        ``inv(P0) @ P[i]`` is Pi3X's relative motion (survives for i!=0, so the trajectory shape is
        preserved); only frame 0 is pinned. The ``C @ ... @ C`` conjugation re-expresses that
        relative motion in OpenGL axes.

        **Anchoring is the only per-dataset difference**, selected by the caller via ``gt_poses``:

        - N1 (``"opengl"``): caller passes the GT poses; ``A0`` from ``action`` (Schema A) anchors
          into the dataset GT world. ``aligned[0] == A0``. (verified correct.)
        - Lerobot (``"opencv"``): caller passes ``gt_poses=None``; ``A0=I`` anchors to the frame-0
          CAMERA, i.e. ``aligned[i] = C @ inv(P0) @ P[i] @ C`` (OpenGL axes, ``aligned[0]==I`` since
          ``C@C=I``), NOT the odom+hand-eye GT world. The odom+hand-eye GT is still reconstructed
          (Schema B) for diagnostics only.

        **GT-free fallback**: when ``gt_poses`` is None/empty (no GT, or opencv as above), set
        ``A0 = I`` -> ``M_left = C @ inv(P0)``, so ``aligned[0] == I`` (ego-canonical: frame-0
        camera at origin/identity, trajectory relative to it; camera-tilted, NOT gravity-aligned,
        and NOT the dataset's real world placement — that needs GT).

        Args:
            camera_pose (np.ndarray): (N, 4, 4) model camera poses (cam->world, scaled).
            gt_poses (np.ndarray or None): (M, 4, 4) dataset GT camera->world poses; only ``[0]``
                is used as the anchor. None/empty -> GT-free (frame-0 camera) fallback.

        Returns:
            tuple ``(aligned, M_left)``:
                - aligned (np.ndarray or None): (N, 4, 4) OpenGL-convention poses in the GT world
                  frame (or the frame-0 camera frame on fallback, aligned[0]==I); None when
                  ``camera_pose`` empty.
                - M_left (np.ndarray or None): 4x4 model-world -> saved-frame rigid for world-fusion
                  reuse; ``C @ inv(P0)`` on fallback.
        """
        if camera_pose is None or len(camera_pose) == 0:
            return None, None

        P = np.asarray(camera_pose, dtype=np.float64)
        C4 = self._camera_opengl_bridge()

        if gt_poses is not None and len(gt_poses) > 0:
            A0 = np.asarray(gt_poses[0], dtype=np.float64)
            M_left = A0 @ C4 @ np.linalg.inv(P[0])  # model-world -> GT-world (frame-0 anchored to GT)
        else:
            # GT-free fallback: frame-0 normalization = the general M_left with A0 = I.
            # aligned[0] = C4 @ inv(P0) @ P0 @ C4 = C4 @ C4 = I (ego-canonical: frame-0 cam = origin).
            M_left = C4 @ np.linalg.inv(P[0])

        # aligned[i] = M_left @ P[i] @ C4
        aligned = np.einsum("ij,njk,kl->nil", M_left, P, C4)
        return aligned.astype(np.float32), M_left

    def compute_trajectory_scale(self, poses_gt, poses_pred):
        """
        Computes the scale ratio (GT / Pred) between predicted and ground truth trajectories.
        Uses the ratio of standard deviations (Sim3 scale estimation).

        NOTE: not used by the data-processing pipeline; kept for verification only.

        Args:
            poses_gt : Ground truth poses (N, 4, 4).
            poses_pred : Predicted poses (N, 4, 4).

        Returns:
            scale: The calculated scale factor. Returns 1.0 if calculation fails or input is invalid.
        """

        # Ensure frame counts match, then extract translation columns.
        n_frames = min(len(poses_gt), len(poses_pred))
        traj_gt = np.asarray(poses_gt)[:n_frames, :3, 3]
        traj_pred = np.asarray(poses_pred)[:n_frames, :3, 3]

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
            print("[Scale Warning] Calculated scale is NaN/Inf. Using 1.0")
            return 1.0

        print(
            f"[Scale Info] GT std: {std_gt:.4f}, Pred std: {std_pred:.4f} -> Scale: {scale:.4f}"
        )
        return scale

    def align_with_gt_scale(self, input_path, pcd):
        """
        Aligns the Pi3X reconstruction to the GT world frame via a two-stage Sim3 transform.

        NOTE: not used by the data-processing pipeline; kept for verification only.

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
                _pcdo3d = o3d.geometry.PointCloud()
                _pcdo3d.points = o3d.utility.Vector3dVector(pcd_aligned.astype(np.float64))
                _plane, _inliers = _pcdo3d.segment_plane(
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
                            _pcdo3d2 = o3d.geometry.PointCloud()
                            _pcdo3d2.points = o3d.utility.Vector3dVector(
                                pcd_aligned.astype(np.float64)
                            )
                            _pm2, _ = _pcdo3d2.segment_plane(
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
            raise

    def _fold_gravity_into_tcam2base(self, pcd, T_cam2base):
        """Lerobot z-deskew stage ① (frame-0 gravity fold): fold the base-frame Z-tilt deskew
        into the extrinsic rotation.

        Estimate R_deskew (ground normal -> +Z) in the frame-0 base frame (Lerobot convention
        C=identity) and fold it as R_eff = R_deskew @ R_c2b — a single uniform deskew applied to
        every frame (matching exp_coord_align's frame-0 result). The residual per-frame drift on
        later frames is then handled by stage ② (per-frame leveling in ``compute_sequence_data``).
        On RANSAC failure, warn and return T_cam2base unchanged. Called only when z_deskew is on.
        """
        try:
            cam0 = self.camera_pose[0]
            p_base0 = self.convert_pointcloud_camera_to_base(
                self.convert_pointcloud_world_to_camera(pcd, cam0), T_cam2base
            )
            cc_base0 = self.convert_pointcloud_camera_to_base(
                self.convert_pointcloud_world_to_camera(self.camera_pose[:, :3, 3], cam0),
                T_cam2base,
            )
            R_deskew, tilt_before = gravity_R_to_z(p_base0, cc_base0)
        except ValueError as e:
            print(f"[gravity-base] lerobot Z-tilt deskew skipped: {e}")
            return T_cam2base
        T_eff = np.array(T_cam2base, dtype=np.float32)
        T_eff[:3, :3] = R_deskew @ T_eff[:3, :3]
        print(f"[gravity-base] lerobot Z-tilt deskew folded into T_cam2base: tilt {tilt_before:.2f} deg -> ~0")
        return T_eff

    def align_to_world(self, pcd):
        """GT-free alignment of the Pi3X reconstruction to the real-world frame (z up,
        frame-0 canonical origin/orientation). Thin wrapper around the pure transform
        ``align_reconstruction_to_world`` (utils); this method only does the instance I/O.

        NOTE: not used by the data-processing pipeline; kept for verification only.

        Reads ``self.metric_scale_correction``; updates ``self.pcd`` / ``self.camera_pose``
        in place; returns ``(pcd_aligned, scale)``.
        """
        if self.camera_pose is None or len(self.camera_pose) == 0:
            raise ValueError("[align_to_world] camera_pose is empty")

        pcd_aligned, cp_aligned, s = align_reconstruction_to_world(
            pcd, self.camera_pose, self.metric_scale_correction
        )
        self.pcd = pcd_aligned
        self.camera_pose = cp_aligned
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
        # Target the OUTPUT meta/info.json (seeded from source), not the input dataset,
        # so OCC features are written to the output and the source stays untouched.
        json_path = paths.get(
            "meta_info_json", os.path.join(self.save_path, "meta", "info.json")
        )

        def _update_info_logic(meta):
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
            self._update_json_safely(json_path, _update_info_logic)
        else:
            print(f"JSON path does not exist: {json_path}")

    def _save_transforms_parquet(self, paths, per_frame_T_cam2base, T_cam2base_raw):
        """Lerobot z-deskew only: append the per-frame deskew rotation as an ``R_deskew``
        column to ``data/chunk-000/episode_000000.parquet`` (already seeded with the base
        columns and augmented with the OCC camera columns by ``update_metadata``).

        Each row is the 3x3 applied deskew ``D_i = R_corrected_i @ R_raw.T`` (the frame-0 fold
        composed with the per-frame leveling, a pure rotation about the base origin). It maps an
        uncorrected base point to the corrected one (``P_corrected = D_i @ P_uncorrected``), so an
        already-deskewed OCC is restored with ``P_uncorrected = D_i.T @ P_corrected``. Stored as
        list-of-rows (like the N1 ``action`` column) so ``np.asarray(cell)`` -> (3, 3).

        Args:
            paths (dict): Output paths from ``get_io_paths`` (uses ``paths['parquet']``).
            per_frame_T_cam2base (np.ndarray or None): (N, 4, 4) corrected cam->base transforms.
            T_cam2base_raw (np.ndarray or None): 4x4 raw (pre-deskew) hand-eye T_cam2base.
        """
        parquet_path = paths.get("parquet")
        if parquet_path is None:
            print("[parquet] no parquet path resolved; skipping deskew save.")
            return
        if per_frame_T_cam2base is None or len(per_frame_T_cam2base) == 0:
            print(
                "[parquet] no per-frame cam->base transforms available "
                "(T_cam2base missing?); skipping deskew save."
            )
            return
        if T_cam2base_raw is None:
            print(
                "[parquet] raw hand-eye T_cam2base unavailable; cannot derive deskew; "
                "skipping deskew save."
            )
            return

        corrected = np.asarray(per_frame_T_cam2base, dtype=np.float64)[:, :3, :3]
        R_raw = np.asarray(T_cam2base_raw, dtype=np.float64)[:3, :3]
        # D_i = R_corrected_i @ R_raw.T; SVD re-orthonormalize to keep a clean rotation.
        deskew = corrected @ R_raw.T
        U, _, Vt = np.linalg.svd(deskew)
        deskew = U @ Vt
        deskew = deskew.astype(np.float32)

        n = deskew.shape[0]
        r_deskew = [[row for row in d] for d in deskew]
        try:
            # Augment the existing parquet (base + OCC columns) with the deskew column,
            # keeping a single complete episode_000000.parquet. Fall back to a fresh
            # table only if the parquet was never seeded.
            if os.path.exists(parquet_path):
                df = pd.read_parquet(parquet_path, engine="pyarrow")
                if len(df) != n:
                    raise ValueError(
                        f"[Length Mismatch] Parquet has {len(df)} frames, "
                        f"but deskew rotations have {n} frames."
                    )
                df["R_deskew"] = r_deskew
            else:
                print(
                    f"[parquet] {parquet_path} not seeded; writing a fresh "
                    f"deskew-only table."
                )
                df = pd.DataFrame(
                    {
                        "index": np.arange(n, dtype=np.int64),
                        "frame_index": np.arange(n, dtype=np.int64),
                        "R_deskew": r_deskew,
                    }
                )
            df.to_parquet(parquet_path, engine="pyarrow")
            print(
                f"[parquet] saved per-frame z-deskew rotation: {parquet_path} "
                f"({n} frames)"
            )
        except Exception as e:
            print(f"[parquet] failed to save z-deskew rotation: {e}")
            raise e

    def update_meta_episodes_jsonl(self, scale):
        """
        Updates `meta/episodes.jsonl` with the calculated scale value.
        Uses file locking (fcntl) to ensure safe concurrent writes.

        Args:
            scale (float): The calculated scale factor to be saved.

        Returns:
            None: Modifies the jsonl file on disk.
        """
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
                raise

    def _save_world_fusion_sequence(self, final_occ, save_dir, world_transform=None):
        """Save per-frame world-fused ``(N, 7)`` npy for ``tools/visual/npy_to_world_video.py``.

        Reuses the data pipeline's already-computed canonical-ego OCC (``final_occ`` from
        ``compute_sequence_data``) instead of recomputing OCC in the raw camera frame — the
        latter (as ``visual_pipeline`` does) would crop along the wrong axis because
        ``pc_range`` is defined in the canonical base frame (forward=+y), not the OpenCV
        camera frame. Each frame fuses three labeled blocks ``[x, y, z, r, g, b, label]``:

          - label 0: background dense point cloud (``self.pcd`` + ``self.pcd_color``)
          - label 1: camera trajectory up to frame ``i`` (blue)
          - label 2: temporally accumulated visible OCC (gray)

        All points are first expressed in the model-world frame, then rigid-transformed by
        ``world_transform`` (M) into the dataset GT world frame — the same frame as the saved
        ``camera_extrinsic_occ`` — so the visualization overlays the GT trajectory. Writes
        only ``merge_npy_sequence_world/frame_XXXX_world.npy`` (no cam/ply/solo-occ outputs).

        Args:
            final_occ (np.ndarray): (M, 4) sparse ``[frame, vx, vy, vz]`` visible OCC voxel
                indices in the canonical ego/base frame (``compute_sequence_data`` return).
            save_dir (str): Trajectory output dir (usually ``self.save_path``).
            world_transform (np.ndarray or None): 4x4 model-world -> GT-world rigid. None keeps
                the model-world frame (GT unavailable).
        """
        out_dir = os.path.join(save_dir, "merge_npy_sequence_world")
        os.makedirs(out_dir, exist_ok=True)

        total_frames = len(self.camera_pose)
        final_occ = np.asarray(final_occ)
        frame_col = (
            final_occ[:, 0].astype(np.int64)
            if final_occ.size
            else np.zeros((0,), dtype=np.int64)
        )

        def _apply_M(xyz):
            """Apply the 4x4 model-world -> GT-world rigid to (K, 3) points."""
            xyz = np.asarray(xyz, dtype=np.float32)
            if world_transform is None or len(xyz) == 0:
                return xyz
            M = np.asarray(world_transform, dtype=np.float64)
            h = np.concatenate([xyz.astype(np.float64), np.ones((len(xyz), 1))], axis=1)
            return (h @ M.T)[:, :3].astype(np.float32)

        # Background block is constant across frames: build it once (mapped to GT world).
        bg_world = np.asarray(self.pcd, dtype=np.float32)
        if bg_world.ndim == 2 and bg_world.shape[1] == 4:
            bg_world = bg_world[:, :3]
        if getattr(self, "pcd_color", None) is not None:
            bg_color = np.asarray(self.pcd_color, dtype=np.float32)
        else:
            bg_color = np.ones_like(bg_world) * 0.7
        min_len = min(len(bg_world), len(bg_color))
        bg_block = np.concatenate(
            [
                _apply_M(bg_world[:min_len]),
                bg_color[:min_len],
                np.zeros((min_len, 1), dtype=np.float32),  # label 0
            ],
            axis=1,
        ).astype(np.float32)

        # Reset temporal accumulation buffer (same semantics as visual_pipeline).
        self.occ_history_buffer.clear()

        for i in range(total_frames):
            current_pose = self.camera_pose[i]

            # --- OCC (label 2): canonical ego voxels -> cam -> model world ---
            sel = final_occ[frame_col == i, 1:4] if final_occ.size else np.zeros((0, 3))
            if len(sel) > 0:
                ego_pts = voxels_to_pcd(
                    sel.astype(np.float32), self.voxel_size, self.pc_range
                )
                # base -> cam: P_cam = R_eff.T @ P_base  (row form: P_cam = P_base @ R_eff)
                if getattr(self, "occ_per_frame_T_cam2base", None) is not None:
                    R_eff = np.asarray(
                        self.occ_per_frame_T_cam2base[i], dtype=np.float32
                    )[:3, :3]
                    cam_pts = ego_pts @ R_eff
                else:
                    cam_pts = ego_pts
                occ_world = self.convert_pointcloud_camera_to_world(cam_pts, current_pose)
            else:
                occ_world = np.zeros((0, 3), dtype=np.float32)

            save_flag = i % self.history_step == 0
            _, local_occ_world = self.get_temporal_occ(
                occ_world, current_pose, save_to_history=save_flag
            )
            local_occ_world = np.asarray(local_occ_world, dtype=np.float32)
            if local_occ_world.ndim == 2 and local_occ_world.shape[1] == 4:
                local_occ_world = local_occ_world[:, :3]

            # --- Trajectory (label 1) ---
            traj_world = np.asarray(self.camera_pose[: i + 1, :3, 3], dtype=np.float32)
            if len(traj_world) > 0:
                traj_block = np.concatenate(
                    [
                        _apply_M(traj_world),
                        np.tile([0.0, 0.0, 1.0], (len(traj_world), 1)),  # blue
                        np.ones((len(traj_world), 1), dtype=np.float32),  # label 1
                    ],
                    axis=1,
                ).astype(np.float32)
            else:
                traj_block = np.zeros((0, 7), dtype=np.float32)

            # --- OCC block ---
            if len(local_occ_world) > 0:
                occ_block = np.concatenate(
                    [
                        _apply_M(local_occ_world),
                        np.tile([0.5, 0.5, 0.5], (len(local_occ_world), 1)),  # gray
                        np.full((len(local_occ_world), 1), 2, dtype=np.float32),  # label 2
                    ],
                    axis=1,
                ).astype(np.float32)
            else:
                occ_block = np.zeros((0, 7), dtype=np.float32)

            final_npy = np.concatenate(
                [bg_block, traj_block, occ_block], axis=0
            ).astype(np.float32)
            np.save(os.path.join(out_dir, f"frame_{i:04d}_world.npy"), final_npy)

            if i % 10 == 0:
                print(f"[world-fusion] frame {i}/{total_frames}")

        print(f"[world-fusion] saved {total_frames} frames -> {out_dir}")

    def _export_lerobot_source_media(self, input_path, depth_video_path):
        """Lerobot (opencv) only: copy the source RGB/depth VIDEOS into the output trajectory.

        Writes, under ``<save_path>/videos/chunk-000/``:
          - ``observation.video.rgb/episode_000000.mp4`` — verbatim copy of the source RGB mp4.
          - ``observation.video.depth/episode_000000<ext>`` — verbatim copy of the source depth
            video (native ext, typically ``.mkv``), only when ``depth_video_path`` exists.

        ``depth_video_path`` is the loader-resolved depth video for THIS trajectory (independent
        of ``--use_depth`` / model conditioning), so depth is copied whenever it exists. No
        frame extraction / no resize — the source videos are copied byte-for-byte.
        """
        video_chunk_dir = os.path.join(self.save_path, "videos", "chunk-000")

        # Clean up legacy products from older runs (per-frame dirs / trajectory copy) so a
        # re-export yields only the new video-only structure.
        for legacy in (
            "observation.images.rgb",
            "observation.images.depth",
            "observation.video.trajectory",
        ):
            stale = os.path.join(video_chunk_dir, legacy)
            if os.path.isdir(stale):
                shutil.rmtree(stale, ignore_errors=True)

        # RGB: copy the source mp4.
        rgb_dir = os.path.join(video_chunk_dir, "observation.video.rgb")
        os.makedirs(rgb_dir, exist_ok=True)
        rgb_out = os.path.join(rgb_dir, "episode_000000.mp4")
        shutil.copy2(input_path, rgb_out)
        print(f"   [export] copied RGB video -> {rgb_out}")

        # Depth: copy the source video (keep native ext), only when present.
        if depth_video_path and os.path.isfile(depth_video_path):
            depth_dir = os.path.join(video_chunk_dir, "observation.video.depth")
            os.makedirs(depth_dir, exist_ok=True)
            ext = os.path.splitext(depth_video_path)[1] or ".mkv"
            depth_out = os.path.join(depth_dir, f"episode_000000{ext}")
            shutil.copy2(depth_video_path, depth_out)
            print(f"   [export] copied depth video -> {depth_out}")
        else:
            print("   [export] no depth video for this trajectory; skipping depth video copy.")

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
        z_deskew=False,
        save_world_fusion=False,
        export_source_video=False,
        export_depth_path=None,
    ):
        """
        Executes the full data generation pipeline:
        Reconstruction -> metric scale (no GT alignment) -> optional Lerobot z-deskew fold ->
        OCC sequence computation -> save sequence -> metadata update -> optional per-frame
        deskew parquet -> optional global point cloud save.

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
            extrinsic_convention (str, optional): Camera convention of ``T_cam2base``, selecting the
                OpenCV(Pi3X)->extrinsic basis change: ``"opengl"`` (N1 rendered extrinsics) applies
                C=diag(1,-1,-1); ``"opencv"`` (Lerobot hand-eye) / None applies no flip. Produced and
                passed through by ``InternNavSequenceLoader.get_trajectory_info``. Defaults to None.
            z_deskew (bool, optional): Lerobot (opencv) only — enable the z-axis tilt deskew
                (frame-0 gravity fold + per-frame leveling) and write the per-frame deskew
                rotation to ``data/chunk-000/episode_000000.parquet``. Default False. N1
                (opengl) is unaffected.
            save_world_fusion (bool, optional): Also write per-frame world-fused ``(N, 7)`` npy
                to ``merge_npy_sequence_world/`` for ``tools/visual/npy_to_world_video.py``.
                Reuses the in-memory OCC/pcd/poses (no second inference) and places the fusion
                in the dataset GT world frame. Default False.
            export_source_video (bool, optional): Lerobot (opencv) only — also copy the source
                videos verbatim into ``videos/chunk-000/``: the RGB mp4 -> ``observation.video.rgb/
                episode_000000.mp4`` and the depth video -> ``observation.video.depth/
                episode_000000<ext>`` (when present). No frame extraction. N1 (opengl) is
                unaffected. Default False.
            export_depth_path (str, optional): Loader-resolved depth video for THIS trajectory,
                used for the depth video copy above. Independent of ``condit_depth_path`` /
                ``--use_depth`` so depth is exported whenever it exists. Defaults to None.

        Returns:
            None
        """
        # Check Status
        if not self._check_processing_status(input_path, overwrite=overwrite):
            return

        # 3D Reconstruction
        pcd, self.camera_pose, self.norm_cam_ray = self.pcd_reconstruction(
            input_path, condit_depth_path, intrinsics_np
        )

        # No align_to_world (matches exp_coord_align): coordinate alignment is done by the
        # camera-convention basis change C in compute_sequence_data (per extrinsic_convention).
        # Scale trusts metric_head plus the config correction factor; apply it in place to pcd and
        # camera translations. align_to_world / align_with_gt_scale / get_gt_poses are kept for
        # offline GT diagnosis only — the saved camera extrinsic below is GT-free (convention change).
        s = float(self.metric_scale_correction)
        pcd = self.pcd = pcd * s
        self.camera_pose[:, :3, 3] *= s
        print(f"[Scale Info] no align_to_world; scale={s:.4f} (metric_scale_correction)")

        # Lerobot (OpenCV) z-deskew, only when enabled: fold ground-normal->+Z into T_cam2base
        # to fix the slight Z down-tilt. Off -> raw hand-eye. N1 (OpenGL) skipped.
        # Keep the raw (pre-deskew) hand-eye so the pure per-frame deskew D_i can be recovered.
        T_cam2base_raw = None
        if z_deskew and extrinsic_convention == "opencv" and T_cam2base is not None:
            T_cam2base_raw = np.array(T_cam2base, dtype=np.float32)
            T_cam2base = self._fold_gravity_into_tcam2base(pcd, T_cam2base)

        paths = self.get_io_paths(input_path)

        # Lerobot only: copy the source RGB/depth videos verbatim (no frame extraction, no
        # trajectory mp4) so each output trajectory is self-contained. Like save_world_fusion,
        # this only runs together with (re)generation, so use --overwrite true to re-copy.
        if export_source_video and extrinsic_convention == "opencv":
            self._export_lerobot_source_media(input_path, export_depth_path)

        # Output is a separate dir: seed the source meta/ + base parquet so the in-place
        # update_* steps below have records to augment (otherwise they silently no-op).
        self._seed_source_metadata(input_path, paths, overwrite=overwrite)

        self.update_meta_episodes_jsonl(s)

        print("Start processing sequence frames...")

        # Scale already applied in place to pcd / camera_pose, so the OCC/voxel stage uses scale=1.0.
        arr_4d_occ, arr_4d_mask, all_camera_poses, all_camera_intrinsics = (
            self.compute_sequence_data(
                pcd,
                mesh=mesh,
                T_cam2base=T_cam2base,
                scale=1.0,
                extrinsic_convention=extrinsic_convention,
                z_deskew=z_deskew,
            )
        )

        # Save sequence data
        print("Saving 4D Sequence Arrays...")
        self.save_sequence_data(paths, arr_4d_occ, arr_4d_mask)

        # Camera extrinsic anchoring — saved extrinsic is ALWAYS OpenGL axes (downstream model
        # input convention; C=diag(1,-1,-1) applied in _align_camera_poses_to_gt_world). Only the
        # ANCHORING differs per dataset:
        #   - N1 (opengl): frame-0 anchoring into the dataset GT world (Method A). The frame-0 GT
        #     pose A0 pins the global origin/orientation; Pi3X's relative motion + metric scale
        #     carry the later frames. (verified correct — left unchanged.)
        #   - Lerobot (opencv): anchor the SAVED extrinsic to the frame-0 CAMERA (A0=I), i.e.
        #     C @ inv(P0) @ P[i] @ C (OpenGL axes, robot-camera frame, aligned[0]=I since C@C=I),
        #     NOT the odom GT world (whose A0 reorientation broke the downstream camera convention).
        # self.camera_pose itself stays untouched because the OCC stage above (world->camera per
        # frame) depends on it. ``world_M`` is the model-world->saved-frame rigid, reused for
        # world-fusion so it lands in the SAME frame as camera_extrinsic_occ. GT is still read here
        # (used for the traj_gt.ply / world overlays) but, for opencv, NOT used to anchor the save.
        self.gt_camera_pose_world = self.get_gt_poses(input_path)
        gt_for_anchor = (
            None if extrinsic_convention == "opencv" else self.gt_camera_pose_world
        )
        aligned_poses, world_M = self._align_camera_poses_to_gt_world(
            self.camera_pose, gt_for_anchor
        )
        self.aligned_camera_pose_world = aligned_poses  # stashed for save_global_data overlay
        # GT-free convention-ONLY trajectory (C @ P @ C, un-anchored, OpenGL axes) for the
        # infer_cam_traj_R.ply diagnostic — the pure axis-convention change that still sits in the
        # model world frame. Computed inline (NOT via the method's no-GT branch, which frame-0 pins).
        if self.camera_pose is not None and len(self.camera_pose) > 0:
            _C4 = self._camera_opengl_bridge()
            _P = np.asarray(self.camera_pose, dtype=np.float64)
            self.gtfree_camera_pose_world = np.einsum(
                "ij,njk,kl->nil", _C4, _P, _C4
            ).astype(np.float32)
        else:
            self.gtfree_camera_pose_world = None
        _anchored = self.gt_camera_pose_world is not None and len(self.gt_camera_pose_world) > 0
        if aligned_poses is not None:
            all_camera_poses = [[row for row in pose] for pose in aligned_poses]
            if extrinsic_convention == "opencv":
                _msg = "frame-0 camera anchored (OpenGL, aligned[0]=I); GT used for diagnostics only."
            elif _anchored:
                _msg = "frame-0 anchored to GT world."
            else:
                _msg = "GT-free fallback (no GT; frame-0 normalized, aligned[0]=I)."
            print("[extrinsic] camera_extrinsic_occ: " + _msg)
        else:
            print(
                "[extrinsic] no camera poses to transform; keeping model-world camera poses (scaled)."
            )

        # Update metadata
        self.update_metadata(paths, all_camera_poses, all_camera_intrinsics, input_path)

        # Lerobot z-deskew: persist the per-frame deskew rotation D_i to episode_000000.parquet.
        # After update_metadata so its "Parquet not found" path stays harmless.
        if z_deskew and extrinsic_convention == "opencv":
            self._save_transforms_parquet(
                paths, self.occ_per_frame_T_cam2base, T_cam2base_raw
            )

        if pcd_save:
            # Save global data
            self.save_global_data(paths)

        # Optional: per-frame world-fused (N, 7) npy for npy_to_world_video.py. Reuses the
        # in-memory OCC (arr_4d_occ) / pcd / poses — no second inference. Map the fusion with the
        # SAME ``world_M`` (model-world -> GT-world rigid = A0 @ C4 @ inv(P0), or C4 on GT-free
        # fallback) used for camera_extrinsic_occ, so the fusion overlays the saved trajectory.
        if save_world_fusion and world_M is not None:
            self._save_world_fusion_sequence(
                arr_4d_occ, self.save_path, world_transform=world_M
            )
