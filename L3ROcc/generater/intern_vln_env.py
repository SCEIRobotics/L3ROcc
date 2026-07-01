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
    """Pipeline for InternNav: 3D reconstruction, metric scale, optional Lerobot z-deskew,
    occupancy generation, and lock-safe metadata updates.

    The OCC chain is GT-free (ego/base frame). The saved ``camera_extrinsic_occ`` is ALWAYS in
    the OpenGL camera convention (C=diag(1,-1,-1)); only the anchoring differs per dataset (see
    ``_align_camera_poses_to_gt_world``):
      - N1 (opengl): frame-0 anchored into the dataset GT world (``aligned[0]==A0``).
      - Lerobot (opencv): frame-0 CAMERA anchored (``aligned[0]==I``); odom+hand-eye GT is
        reconstructed for diagnostics only.
    """

    def __init__(
        self,
        config_path,
        save_dir,
        model_dir,
        model_type="pi3x",
    ):
        """Init the generator. ``model_type`` is "pi3" (RGB-only forward) or "pi3x"
        (consumes depth/intrinsic kwargs); both honor a calibrated K at post-processing."""
        super().__init__(config_path, save_dir, model_dir, model_type=model_type)

    def _check_processing_status(self, input_path, overwrite=False):
        """Return True if processing is needed: checks target files, parquet OCC
        columns/lengths, and the ``scale`` key in episodes.jsonl."""
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
        """Create the output directory tree and return a dict of output file paths
        (ply, global_occ, parquet, occ_seq, mask_seq, meta_dir, meta_info_json,
        meta_episodes_jsonl)."""

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
        """Locate the GT parquet matching ``input_path`` (episode_<id>.parquet under
        traj_root or save_path, else any episode_*.parquet glob). Returns path or None."""
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
        """Copy the source ``meta/`` folder and base ``episode_*.parquet`` into the output
        (parquet renamed to ``episode_000000.parquet``) so the in-place ``update_*`` steps
        have records to augment. Missing sources warn but do not raise."""
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
        """Parse GT camera poses -> (N, 4, 4) in the GT world frame, else None.

        Two parquet schemas:
        A. InternData-N1: ``action`` column, each row a 4x4 SE(3) camera pose.
        B. lerobot rosbag: ``observation.state`` (body odometry) + hand-eye
           ``R/t_cam2gripper`` from info.json -> ``R_world_cam = R_world_body @ R_cam2gripper``,
           ``p_world_cam = R_world_body @ t_cam2gripper + p_body``.
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
                    # N1 action cell is an object array of 4 row-vectors; parse row-by-row.
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
                # Robust to object-array cells, like Schema A.
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
                # Physical down-tilt = angle between camera z and the body horizontal plane
                # (XYZ-Euler pitch is misleading when |roll|>10°).
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
        """4x4 OpenCV->OpenGL camera-axis basis change (``C=diag(1,-1,-1)`` top-left),
        dataset-independent. Pi3X outputs OpenCV poses; the saved extrinsic must be OpenGL."""
        C4 = np.eye(4, dtype=np.float64)
        C4[:3, :3] = self.R_opencv_to_opengl
        return C4

    def _align_camera_poses_to_gt_world(self, camera_pose, gt_poses):
        """Frame-0 anchor the Pi3X poses, emitting OpenGL convention. Returns
        ``(aligned (N,4,4), M_left)`` where (``C``=OpenCV->OpenGL bridge)::

            M_left     = A0 @ C @ inv(P0)
            aligned[i] = M_left @ P[i] @ C

        Anchoring is the only per-dataset difference, via ``gt_poses``:
        - N1: pass GT poses; ``A0`` anchors into the GT world (``aligned[0]==A0``).
        - Lerobot / no GT: pass None; ``A0=I`` anchors to the frame-0 camera (``aligned[0]==I``).
        """
        if camera_pose is None or len(camera_pose) == 0:
            return None, None

        P = np.asarray(camera_pose, dtype=np.float64)
        C4 = self._camera_opengl_bridge()

        if gt_poses is not None and len(gt_poses) > 0:
            A0 = np.asarray(gt_poses[0], dtype=np.float64)
            M_left = A0 @ C4 @ np.linalg.inv(P[0])  # frame-0 anchored to GT world
        else:
            # GT-free fallback: A0 = I -> aligned[0] = I (frame-0 camera = origin).
            M_left = C4 @ np.linalg.inv(P[0])

        # aligned[i] = M_left @ P[i] @ C4
        aligned = np.einsum("ij,njk,kl->nil", M_left, P, C4)
        return aligned.astype(np.float32), M_left

    def compute_trajectory_scale(self, poses_gt, poses_pred):
        """Sim3 scale ratio (GT / Pred) from the ratio of translation std devs; 1.0 on failure.

        NOTE: not used by the pipeline; kept for verification only.
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
        """Align the Pi3X reconstruction to the GT world frame via a two-stage Sim3:
        (1) rotation by orthogonal Procrustes on per-frame rotation columns (robust to
        near-collinear trajectories), (2) scale + translation on translation columns.
        Applies (s, R, t) in place to ``self.pcd`` / ``self.camera_pose`` and returns
        ``(pcd_aligned, scale)``.

        NOTE: not used by the pipeline; kept for verification only.
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

            # Ground-plane RANSAC -> tilt vs world Z; if over threshold, Rodrigues gravity
            # correction aligns the ground normal to +Z.
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

                # Threshold: ~2° triggers correction for lerobot (tilt ≈ 4-5°) while
                # skipping gravity-aligned N1 (tilt ≈ 0-1°).
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
        """Lerobot z-deskew stage 1: estimate R_deskew (ground normal -> +Z) in the frame-0
        base frame and fold it as ``R_eff = R_deskew @ R_c2b`` (uniform deskew for all frames;
        per-frame drift handled later in ``compute_sequence_data``). Returns T_cam2base
        unchanged on RANSAC failure."""
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
        """GT-free alignment to the real-world frame (z up, frame-0 canonical). Thin wrapper
        over ``align_reconstruction_to_world`` (utils); updates ``self.pcd`` /
        ``self.camera_pose`` in place and returns ``(pcd_aligned, scale)``.

        NOTE: not used by the pipeline; kept for verification only.
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
        """Write the generated camera extrinsic/intrinsic OCC columns to the output parquet
        and the OCC features to meta/info.json."""

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
        # Target the OUTPUT meta/info.json (seeded from source), leaving the source untouched.
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
        """Lerobot z-deskew only: append per-frame deskew rotation ``D_i = R_corrected_i @ R_raw.T``
        as an ``R_deskew`` column to the output parquet (maps uncorrected base point ->
        corrected; restore with ``D_i.T``). Stored as list-of-rows -> (3, 3) per cell."""
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
        # D_i = R_corrected_i @ R_raw.T; SVD re-orthonormalize.
        deskew = corrected @ R_raw.T
        U, _, Vt = np.linalg.svd(deskew)
        deskew = U @ Vt
        deskew = deskew.astype(np.float32)

        n = deskew.shape[0]
        r_deskew = [[row for row in d] for d in deskew]
        try:
            # Augment the existing parquet with the deskew column; fresh table only if unseeded.
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
        """Write ``scale`` into every entry of meta/episodes.jsonl (fcntl-locked)."""
        meta_dir = os.path.join(self.save_path, "meta")
        jsonl_path = os.path.join(meta_dir, "episodes.jsonl")

        if not os.path.exists(jsonl_path):
            print(f"[Meta Warning] episodes.jsonl not found at: {jsonl_path}")
            return

        # Retry while acquiring the lock.
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
        """Lock-safe JSON update: apply ``update_func(data)`` and write back its result
        (no write when it returns None)."""
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
        """Save per-frame world-fused ``(N, 7)`` npy for ``tools/visual/npy_to_world_video.py``,
        reusing the canonical-ego OCC (``final_occ``). Each frame fuses three labeled blocks
        ``[x, y, z, r, g, b, label]``: label 0 background pcd, label 1 trajectory, label 2
        accumulated visible OCC. Points are mapped by ``world_transform`` into the GT world
        frame (None keeps model-world). Writes only ``merge_npy_sequence_world/``."""
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

        # Reset temporal accumulation buffer.
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
        """Lerobot (opencv) only: copy the source RGB/depth videos verbatim into
        ``videos/chunk-000/`` (``observation.video.rgb/episode_000000.mp4`` and, when present,
        ``observation.video.depth/episode_000000<ext>``). No frame extraction / resize."""
        video_chunk_dir = os.path.join(self.save_path, "videos", "chunk-000")

        # Remove legacy per-frame / trajectory products from older runs.
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
        """Run the full pipeline: reconstruction -> metric scale (no GT alignment) ->
        optional Lerobot z-deskew fold -> OCC sequence -> save -> metadata update ->
        optional per-frame deskew parquet -> optional global pcd save.

        Key args: ``condit_depth_path`` / ``intrinsics_np`` are Pi3X conditioning (intrinsic
        also overrides the saved K); ``T_cam2base`` + ``extrinsic_convention`` ("opengl" N1 ->
        C=diag(1,-1,-1); "opencv"/None Lerobot -> no flip) set the camera convention;
        ``z_deskew`` (Lerobot only) enables z-tilt deskew; ``save_world_fusion`` /
        ``export_source_video`` / ``export_depth_path`` control optional outputs.
        """
        # Check Status
        if not self._check_processing_status(input_path, overwrite=overwrite):
            return

        # 3D Reconstruction
        pcd, self.camera_pose, self.norm_cam_ray = self.pcd_reconstruction(
            input_path, condit_depth_path, intrinsics_np
        )

        # No align_to_world: coordinate alignment is the camera-convention basis change C in
        # compute_sequence_data. Scale = metric_head * config correction, applied in place to
        # pcd and camera translations. (align_* / get_gt_poses are GT-diagnosis only.)
        s = float(self.metric_scale_correction)
        pcd = self.pcd = pcd * s
        self.camera_pose[:, :3, 3] *= s
        print(f"[Scale Info] no align_to_world; scale={s:.4f} (metric_scale_correction)")

        # Lerobot (OpenCV) z-deskew when enabled: fold ground-normal->+Z into T_cam2base.
        # Keep the raw hand-eye so the per-frame deskew D_i can be recovered. N1 skipped.
        T_cam2base_raw = None
        if z_deskew and extrinsic_convention == "opencv" and T_cam2base is not None:
            T_cam2base_raw = np.array(T_cam2base, dtype=np.float32)
            T_cam2base = self._fold_gravity_into_tcam2base(pcd, T_cam2base)

        paths = self.get_io_paths(input_path)

        # Lerobot only: copy the source RGB/depth videos verbatim (runs only on (re)generation).
        if export_source_video and extrinsic_convention == "opencv":
            self._export_lerobot_source_media(input_path, export_depth_path)

        # Seed the source meta/ + base parquet so the in-place update_* steps have records.
        self._seed_source_metadata(input_path, paths, overwrite=overwrite)

        self.update_meta_episodes_jsonl(s)

        print("Start processing sequence frames...")

        # Scale already applied in place, so the OCC/voxel stage uses scale=1.0.
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

        # Camera extrinsic anchoring (saved extrinsic is always OpenGL axes). Anchoring differs
        # per dataset: N1 (opengl) -> frame-0 GT world; Lerobot (opencv) -> frame-0 camera (A0=I).
        # self.camera_pose stays untouched (the OCC stage depends on it). ``world_M`` (model-world
        # -> saved-frame rigid) is reused for world-fusion. GT read here is diagnostic for opencv.
        self.gt_camera_pose_world = self.get_gt_poses(input_path)
        gt_for_anchor = (
            None if extrinsic_convention == "opencv" else self.gt_camera_pose_world
        )
        aligned_poses, world_M = self._align_camera_poses_to_gt_world(
            self.camera_pose, gt_for_anchor
        )
        self.aligned_camera_pose_world = aligned_poses  # for save_global_data overlay
        # GT-free convention-only trajectory (C @ P @ C, un-anchored) for the
        # infer_cam_traj_R.ply diagnostic.
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

        # Lerobot z-deskew: persist the per-frame deskew rotation D_i (after update_metadata).
        if z_deskew and extrinsic_convention == "opencv":
            self._save_transforms_parquet(
                paths, self.occ_per_frame_T_cam2base, T_cam2base_raw
            )

        if pcd_save:
            # Save global data
            self.save_global_data(paths)

        # Optional: per-frame world-fused (N, 7) npy, reusing in-memory OCC/pcd/poses and the
        # same ``world_M`` as camera_extrinsic_occ so the fusion overlays the saved trajectory.
        if save_world_fusion and world_M is not None:
            self._save_world_fusion_sequence(
                arr_4d_occ, self.save_path, world_transform=world_M
            )
