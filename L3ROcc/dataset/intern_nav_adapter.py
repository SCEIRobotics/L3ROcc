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

    def __init__(self, root_dirs):
        self.root_dirs = root_dirs

        # Lists to store file paths for valid trajectories
        self.trajectory_dirs = []  # Root directory of the trajectory
        self.trajectory_rgb_paths = []  # Path to the RGB image folder
        self.trajectory_data_paths = []  # Path to the metadata .parquet file
        self.trajectory_video_paths = []  # Path to the RGB trajectory video file
        self.trajectory_depth_paths = []  # Path to the depth video file (.mkv/.mp4), if available

        self._scan_dataset()

    def _scan_dataset(self):
        """
        Traverses the dataset root directory to index all valid trajectory sequences.
        A trajectory is considered valid only if RGB, data, and video components exist.
        """
        print(f"Scanning dataset in {self.root_dirs}...")

        # 1. Iterate over scene groups (e.g., gibson_zed, 3dfront_d435i)
        group_dirs = [
            d
            for d in os.listdir(self.root_dirs)
            if os.path.isdir(os.path.join(self.root_dirs, d))
        ]

        for group_dir in group_dirs:
            group_path = os.path.join(self.root_dirs, group_dir)
            scene_dirs = os.listdir(group_path)

            # 2. Iterate over individual scenes (e.g., 00154c06...)
            for scene_dir in scene_dirs:
                scene_path = os.path.join(group_path, scene_dir)
                if not os.path.isdir(scene_path):
                    continue

                traj_dirs = os.listdir(scene_path)

                # 3. Iterate over trajectory folders (e.g., trajectory_1)
                for traj_dir in traj_dirs:
                    entire_task_dir = os.path.join(scene_path, traj_dir)

                    # Construct paths for critical components
                    data_path = os.path.join(
                        entire_task_dir, "data/chunk-000/episode_000000.parquet"
                    )

                    # Define the potential video location
                    video_folder_path = os.path.join(
                        entire_task_dir, "videos/chunk-000/observation.video.trajectory"
                    )

                    # Locate the specific RGB .mp4 video file
                    video_file_path = None
                    if os.path.exists(video_folder_path):
                        # Case A: The path is directly a file
                        if os.path.isfile(
                            video_folder_path
                        ) and video_folder_path.endswith(".mp4"):
                            video_file_path = video_folder_path
                        # Case B: The path is a directory containing the mp4
                        elif os.path.isdir(video_folder_path):
                            files = os.listdir(video_folder_path)
                            for f in files:
                                if f.endswith(".mp4"):
                                    video_file_path = os.path.join(video_folder_path, f)
                                    break

                    # Try locating the paired depth video for Pi3X multimodal reconstruction
                    depth_folder_path = os.path.join(
                        entire_task_dir, "videos/chunk-000/observation.video.depth"
                    )
                    depth_file_path = None
                    if os.path.exists(depth_folder_path):
                        if os.path.isfile(depth_folder_path) and (
                            depth_folder_path.endswith(".mkv")
                            or depth_folder_path.endswith(".mp4")
                        ):
                            depth_file_path = depth_folder_path
                        elif os.path.isdir(depth_folder_path):
                            files = os.listdir(depth_folder_path)
                            for f in files:
                                if f.endswith(".mkv") or f.endswith(".mp4"):
                                    depth_file_path = os.path.join(depth_folder_path, f)
                                    break

                    # Validate that all required components exist before registering
                    if os.path.exists(data_path) and video_file_path:
                        self.trajectory_dirs.append(entire_task_dir)
                        self.trajectory_data_paths.append(data_path)
                        self.trajectory_video_paths.append(video_file_path)
                        self.trajectory_depth_paths.append(depth_file_path)

        print(f"Found {len(self.trajectory_dirs)} valid trajectories.")

    def __len__(self):
        return len(self.trajectory_dirs)

    def get_trajectory_info(self, index):
        """
        Retrieves information for a specific trajectory by index.

        Args:
            index (int): The index of the trajectory sequence.

        Returns:
            tuple: (video_path, depth_path, camera_intrinsic, camera_extrinsic)
                - video_path (str): Absolute path to the RGB trajectory video file.
                - depth_path (str or None): Absolute path to the depth video file if available.
                - camera_intrinsic (np.ndarray or None): 3x3 camera intrinsic matrix.
                - camera_extrinsic (np.ndarray or None): 4x4 camera to base extrinsic matrix.
        """

        def _reshape_matrix(value, shape):
            arr = np.array(value, dtype=np.float32)
            if arr.shape == shape:
                return arr
            if arr.size == shape[0] * shape[1]:
                return arr.reshape(shape)
            return None

        # 1. Retrieve stored paths
        video_path = self.trajectory_video_paths[index]
        depth_path = self.trajectory_depth_paths[index]
        data_path = self.trajectory_data_paths[index]
        traj_root = self.trajectory_dirs[index]

        # 2. Parse Parquet data to extract camera intrinsics / extrinsics
        camera_intrinsic = None
        camera_extrinsic = None
        try:
            df = pd.read_parquet(data_path)

            if "observation.camera_intrinsic" in df.columns and len(df) > 0:
                camera_intrinsic = _reshape_matrix(
                    df["observation.camera_intrinsic"].tolist()[0], (3, 3)
                )

            if "observation.camera_extrinsic" in df.columns and len(df) > 0:
                camera_extrinsic = _reshape_matrix(
                    df["observation.camera_extrinsic"].tolist()[0], (4, 4)
                )

            if camera_intrinsic is not None:
                print(
                    f"Loaded camera intrinsic for trajectory {index}: \n{camera_intrinsic}"
                )
        except Exception as e:
            print(f"Error reading parquet {data_path}: {e}")

        # 3. Fallback: try meta/info.json for Pi3X conditioning intrinsics
        if camera_intrinsic is None:
            info_json_path = os.path.join(traj_root, "meta", "info.json")
            if os.path.exists(info_json_path):
                try:
                    with open(info_json_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if "head_camera_intrinsic" in meta:
                        camera_intrinsic = _reshape_matrix(
                            meta["head_camera_intrinsic"], (3, 3)
                        )
                        print(
                            f"Loaded head_camera_intrinsic from info.json for trajectory {index}."
                        )
                except Exception as e:
                    print(f"Error reading intrinsic json {info_json_path}: {e}")

        return video_path, depth_path, camera_intrinsic, camera_extrinsic
