import json
import os

import numpy as np
import pandas as pd


class InternNavSequenceLoader:
    """
    A data loader for the InternNav dataset.

    This class traverses the dataset directory structure to identify valid trajectory
    sequences and manages paths for RGB images, metadata (Parquet), and video files.
    """

    # Candidate sub-directories (relative to a unit root) under which the RGB
    # trajectory video may live. Probed in order; first hit wins.
    # - observation.video.trajectory: InternData-N1 layout (one mp4 per trajectory)
    # - observation.images.RGB / .rgb: lerobot rosbag layout (many episode_*.mp4 per rosbag)
    _RGB_CANDIDATE_DIRS = (
        "videos/chunk-000/observation.video.trajectory",
        "videos/chunk-000/observation.images.RGB",
        "videos/chunk-000/observation.images.rgb",
    )
    _RGB_CANDIDATE_EXTS = (".mp4",)

    # Candidate sub-directories (relative to a unit root) under which a depth
    # video may live. Probed in order; first hit wins.
    _DEPTH_CANDIDATE_DIRS = (
        "videos/chunk-000/observation.video.depth",
        "videos/chunk-000/observation.images.depth",
    )
    _DEPTH_CANDIDATE_EXTS = (".mkv", ".mp4")

    def __init__(self, root_dirs):
        self.root_dirs = root_dirs

        # Per-episode entries. One element per (unit, episode) pair.
        # `trajectory_dirs[i]` holds the unit root (rosbag_* or trajectory_*) the
        # episode lives in; multiple episodes from the same rosbag share that root.
        self.trajectory_dirs = []
        self.trajectory_data_paths = []   # episode_*.parquet
        self.trajectory_video_paths = []  # episode_*.mp4 (RGB)
        self.trajectory_depth_paths = []  # episode_*.{mkv,mp4} or None

        self._scan_dataset()

    @staticmethod
    def _is_unit_dir(path):
        """A unit holds the canonical chunk layout: ``data/chunk-000`` and
        ``videos/chunk-000`` siblings. Covers both a lerobot ``rosbag_*`` and an
        InternData-N1 ``trajectory_*``."""
        return os.path.isdir(os.path.join(path, "data", "chunk-000")) and os.path.isdir(
            os.path.join(path, "videos", "chunk-000")
        )

    def _iter_unit_dirs(self, root):
        """Yield every unit directory under ``root``, auto-detecting layout:

        1. ``root`` itself is a unit -> single rosbag / single trajectory passed directly.
        2. ``root``'s immediate children are units -> rosbag parent (lerobot batch).
        3. Otherwise fall back to InternData-N1 nesting: ``<group>/<scene>/<trajectory_*>``.
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
        """Return ``[(video_path, episode_id), ...]`` for a unit. Empty list if none.
        ``episode_id`` is the mp4 stem, used to pair parquet/depth by episode."""
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
        """Probe conventional depth-video locations under a unit.

        When ``episode_id`` is given, prefer a file whose stem equals it (lerobot
        rosbag uses one depth file per episode). Otherwise return the first depth
        file found, matching the single-trajectory InternData-N1 convention.
        Depth is optional; absence returns None.
        """
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
        """Pair an episode mp4 with its parquet.

        Tries ``data/chunk-000/{episode_id}.parquet`` first (matches the mp4 stem),
        then falls back to the legacy ``episode_000000.parquet`` fixed name used by
        early InternData-N1 trajectories.
        """
        chunk_dir = os.path.join(unit_dir, "data", "chunk-000")
        primary = os.path.join(chunk_dir, f"{episode_id}.parquet")
        if os.path.exists(primary):
            return primary
        legacy = os.path.join(chunk_dir, "episode_000000.parquet")
        if os.path.exists(legacy):
            return legacy
        return None

    def _scan_dataset(self):
        """Index every ``(unit, episode)`` pair under ``root_dirs``.

        Supports both InternData-N1 nesting and lerobot rosbag layouts; see
        ``_iter_unit_dirs`` for the detection rules.
        """
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
        """
        Retrieves information for a specific trajectory by index.

        Args:
            index (int): The index of the trajectory sequence.

        Returns:
            tuple: (video_path, depth_path, camera_intrinsic, camera_extrinsic, extrinsic_convention)
                - video_path (str): Absolute path to the RGB trajectory video file.
                - depth_path (str or None): Absolute path to the depth video file if available.
                - camera_intrinsic (np.ndarray or None): 3x3 camera intrinsic matrix.
                - camera_extrinsic (np.ndarray or None): 4x4 camera to base extrinsic matrix.
                - extrinsic_convention (str or None): 相机约定，决定 OpenCV(Pi3X)->外参约定 的换基:
                    * "opengl" -- 外参来自 InternData-N1 parquet observation.camera_extrinsic
                      (3D-Front 渲染相机, OpenGL Y-up/Z-back)。下游需对 Pi3X 施加 C=diag(1,-1,-1)。
                    * "opencv" -- 外参来自 lerobot meta/info.json的手眼标定 R_cam2gripper
                      (实采相机, OpenCV Y-down, 与 Pi3X 一致)。下游不翻转 (C=identity)。
                    * None -- 无外参。
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

        # 1. Retrieve stored paths
        video_path = self.trajectory_video_paths[index]
        depth_path = self.trajectory_depth_paths[index]
        data_path = self.trajectory_data_paths[index]
        traj_root = self.trajectory_dirs[index]

        # 2. Parse Parquet data to extract camera intrinsics / extrinsics
        camera_intrinsic = None
        camera_extrinsic = None
        # 外参约定: 由产出 camera_extrinsic 的分支决定 (见 Returns 文档)。
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
                # InternData-N1 渲染相机外参 = OpenGL 约定。
                extrinsic_convention = "opengl"

        if camera_intrinsic is not None:
            print(
                f"Loaded camera intrinsic for trajectory {index}: \n{camera_intrinsic}"
            )

        # 3. Fallback: try meta/info.json for intrinsics and/or hand-eye extrinsic.
        #    lerobot parquet 既无 observation.camera_intrinsic 也无 observation.camera_extrinsic,
        #    内参/外参都需从 meta/info.json 取（外参=手眼标定 R_cam2gripper，cam->base 旋转）。
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
                # 手眼外参 -> T_cam2base(仅旋转参与下游 convert_pointcloud_camera_to_base)。
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
                    # 实采手眼标定相机 = OpenCV 约定 (与 Pi3X 一致, 下游不翻转)。
                    extrinsic_convention = "opencv"
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
