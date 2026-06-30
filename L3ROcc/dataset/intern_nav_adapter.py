import json
import os

import numpy as np
import pandas as pd


class InternNavSequenceLoader:
    """Loads InternNav trajectories: indexes RGB / parquet / depth paths per episode."""

    # RGB trajectory video locations, probed in order; first hit wins.
    _RGB_CANDIDATE_DIRS = (
        "videos/chunk-000/observation.video.trajectory",
        "videos/chunk-000/observation.images.RGB",
        "videos/chunk-000/observation.images.rgb",
    )
    _RGB_CANDIDATE_EXTS = (".mp4",)

    # Depth video locations, probed in order; first hit wins.
    _DEPTH_CANDIDATE_DIRS = (
        "videos/chunk-000/observation.video.depth",
        "videos/chunk-000/observation.images.depth",
    )
    _DEPTH_CANDIDATE_EXTS = (".mkv", ".mp4")

    def __init__(self, root_dirs):
        self.root_dirs = root_dirs

        # One entry per (unit, episode) pair; trajectory_dirs[i] is the unit root.
        self.trajectory_dirs = []
        self.trajectory_data_paths = []   # episode_*.parquet
        self.trajectory_video_paths = []  # episode_*.mp4 (RGB)
        self.trajectory_depth_paths = []  # episode_*.{mkv,mp4} or None

        self._scan_dataset()

    @staticmethod
    def _is_unit_dir(path):
        """True if path has the canonical ``data/chunk-000`` + ``videos/chunk-000`` layout."""
        return os.path.isdir(os.path.join(path, "data", "chunk-000")) and os.path.isdir(
            os.path.join(path, "videos", "chunk-000")
        )

    def _iter_unit_dirs(self, root):
        """Yield unit directories under ``root``, auto-detecting layout:
        (1) root is a unit, (2) root's children are units (lerobot batch),
        (3) InternData-N1 nesting ``<group>/<scene>/<trajectory_*>``.
        """
        if self._is_unit_dir(root):
            yield root
            return

        try:
            first_level = sorted(
                d for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            )
        except OSError:
            return

        first_level_units = [
            os.path.join(root, d) for d in first_level
            if self._is_unit_dir(os.path.join(root, d))
        ]
        if first_level_units:
            for unit in first_level_units:
                yield unit
            return

        # InternData-N1: <group>/<scene>/<trajectory_*>
        for group_dir in first_level:
            group_path = os.path.join(root, group_dir)
            try:
                scenes = os.listdir(group_path)
            except OSError:
                continue
            for scene_dir in sorted(scenes):
                scene_path = os.path.join(group_path, scene_dir)
                if not os.path.isdir(scene_path):
                    continue
                try:
                    trajs = os.listdir(scene_path)
                except OSError:
                    continue
                for traj_dir in sorted(trajs):
                    traj_path = os.path.join(scene_path, traj_dir)
                    if self._is_unit_dir(traj_path):
                        yield traj_path

    def _find_rgb_videos(self, unit_dir):
        """Return ``[(video_path, episode_id), ...]`` for a unit (episode_id = mp4 stem)."""
        for rel_dir in self._RGB_CANDIDATE_DIRS:
            cand = os.path.join(unit_dir, rel_dir)
            if not os.path.exists(cand):
                continue
            if os.path.isfile(cand) and cand.lower().endswith(self._RGB_CANDIDATE_EXTS):
                stem = os.path.splitext(os.path.basename(cand))[0]
                return [(cand, stem)]
            if os.path.isdir(cand):
                hits = []
                for f in sorted(os.listdir(cand)):
                    if f.lower().endswith(self._RGB_CANDIDATE_EXTS):
                        stem = os.path.splitext(f)[0]
                        hits.append((os.path.join(cand, f), stem))
                if hits:
                    return hits
        return []

    def _find_depth_video(self, unit_dir, episode_id=None):
        """Find a depth video under a unit. Prefer the file matching ``episode_id``,
        else the first found. Returns None if absent (depth is optional)."""
        for rel_dir in self._DEPTH_CANDIDATE_DIRS:
            cand_dir = os.path.join(unit_dir, rel_dir)
            if not os.path.isdir(cand_dir):
                continue
            files = sorted(
                f for f in os.listdir(cand_dir)
                if f.lower().endswith(self._DEPTH_CANDIDATE_EXTS)
            )
            if not files:
                continue
            if episode_id is not None:
                for f in files:
                    if os.path.splitext(f)[0] == episode_id:
                        return os.path.join(cand_dir, f)
            return os.path.join(cand_dir, files[0])
        return None

    def _resolve_parquet(self, unit_dir, episode_id):
        """Pair an episode mp4 with its parquet: ``{episode_id}.parquet``,
        else legacy fixed name ``episode_000000.parquet``."""
        chunk_dir = os.path.join(unit_dir, "data", "chunk-000")
        primary = os.path.join(chunk_dir, f"{episode_id}.parquet")
        if os.path.exists(primary):
            return primary
        legacy = os.path.join(chunk_dir, "episode_000000.parquet")
        if os.path.exists(legacy):
            return legacy
        return None

    def _scan_dataset(self):
        """Index every ``(unit, episode)`` pair under ``root_dirs``."""
        print(f"Scanning dataset in {self.root_dirs}...")

        for unit_dir in self._iter_unit_dirs(self.root_dirs):
            for video_path, episode_id in self._find_rgb_videos(unit_dir):
                data_path = self._resolve_parquet(unit_dir, episode_id)
                if data_path is None:
                    continue
                depth_path = self._find_depth_video(unit_dir, episode_id)
                self.trajectory_dirs.append(unit_dir)
                self.trajectory_data_paths.append(data_path)
                self.trajectory_video_paths.append(video_path)
                self.trajectory_depth_paths.append(depth_path)

        print(f"Found {len(self.trajectory_dirs)} valid trajectories.")

    def __len__(self):
        return len(self.trajectory_dirs)

    def get_trajectory_info(self, index):
        """Return (video_path, depth_path, camera_intrinsic, camera_extrinsic,
        extrinsic_convention) for a trajectory.

        extrinsic_convention selects the OpenCV(Pi3X)->extrinsic basis change:
            * "opengl" -- InternData-N1 parquet observation.camera_extrinsic;
              downstream applies C=diag(1,-1,-1).
            * "opencv" -- Lerobot info.json hand-eye R_cam2gripper; no flip.
            * None -- no extrinsic.
        """

        def _reshape_matrix(value, shape):
            arr = np.array(value)
            if arr.shape == shape:
                return arr
            if arr.size == shape[0] * shape[1]:
                return arr.reshape(shape)
            if arr.size == shape[0]:
                return np.stack(value)
            return None

        # 1. Stored paths
        video_path = self.trajectory_video_paths[index]
        depth_path = self.trajectory_depth_paths[index]
        data_path = self.trajectory_data_paths[index]
        traj_root = self.trajectory_dirs[index]

        # 2. Parse parquet for camera intrinsics / extrinsics
        camera_intrinsic = None
        camera_extrinsic = None
        extrinsic_convention = None

        df = pd.read_parquet(data_path)

        if "observation.camera_intrinsic" in df.columns and len(df) > 0:
            camera_intrinsic = _reshape_matrix(
                df["observation.camera_intrinsic"].tolist()[0], (3, 3)
            )

        if "observation.camera_extrinsic" in df.columns and len(df) > 0:
            camera_extrinsic = _reshape_matrix(
                df["observation.camera_extrinsic"].tolist()[0], (4, 4)
            )
            if camera_extrinsic is not None:
                extrinsic_convention = "opengl"  # InternData-N1 render camera

        if camera_intrinsic is not None:
            print(
                f"Loaded camera intrinsic for trajectory {index}: \n{camera_intrinsic}"
            )

        # 3. Fallback to meta/info.json (lerobot path: intrinsic + hand-eye R_cam2gripper).
        if camera_intrinsic is None or camera_extrinsic is None:
            info_json_path = os.path.join(traj_root, "meta", "info.json")
            if os.path.exists(info_json_path):
                with open(info_json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if camera_intrinsic is None and "head_camera_intrinsic" in meta:
                    camera_intrinsic = _reshape_matrix(
                        meta["head_camera_intrinsic"], (3, 3)
                    )
                    print(
                        f"Loaded head_camera_intrinsic from info.json for trajectory {index}."
                    )
                # 手眼外参 -> T_cam2base(仅旋转参与下游)。
                ext = meta.get("head_camera_extrinsic", {})
                if camera_extrinsic is None and "R_cam2gripper" in ext:
                    R_c2b = _reshape_matrix(ext["R_cam2gripper"], (3, 3))
                    if R_c2b is None:
                        raise ValueError(
                            f"head_camera_extrinsic.R_cam2gripper in {info_json_path} "
                            f"is not a 3x3 matrix"
                        )
                    T = np.eye(4)
                    T[:3, :3] = R_c2b
                    t_raw = ext.get("t_cam2gripper", None)
                    if t_raw is not None:
                        T[:3, 3] = np.array(t_raw, dtype=float).reshape(3)
                    camera_extrinsic = T
                    extrinsic_convention = "opencv"  # real hand-eye camera, no flip
                    print(
                        f"Loaded head_camera_extrinsic.R_cam2gripper from info.json "
                        f"as T_cam2base for trajectory {index}."
                    )

        return (
            video_path,
            depth_path,
            camera_intrinsic,
            camera_extrinsic,
            extrinsic_convention,
        )
