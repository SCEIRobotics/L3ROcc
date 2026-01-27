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
        self.trajectory_video_paths = []  # Path to the .mp4 video file

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
                    rgb_dir = os.path.join(
                        entire_task_dir, "videos/chunk-000/observation.images.rgb/"
                    )
                    data_path = os.path.join(
                        entire_task_dir, "data/chunk-000/episode_000000.parquet"
                    )

                    # Define the potential video location
                    video_folder_path = os.path.join(
                        entire_task_dir, "videos/chunk-000/observation.video.trajectory"
                    )

                    # Locate the specific .mp4 video file
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

                    # Validate that all required components exist before registering
                    if (
                        os.path.exists(rgb_dir)
                        and os.path.exists(data_path)
                        and video_file_path
                    ):
                        self.trajectory_dirs.append(entire_task_dir)
                        self.trajectory_rgb_paths.append(rgb_dir)
                        self.trajectory_data_paths.append(data_path)
                        self.trajectory_video_paths.append(video_file_path)

        print(f"Found {len(self.trajectory_dirs)} valid trajectories.")

    def __len__(self):
        return len(self.trajectory_dirs)

    def get_trajectory_info(self, index):
        """
        Retrieves information for a specific trajectory by index.

        Args:
            index (int): The index of the trajectory sequence.

        Returns:
            tuple: (video_path, camera_intrinsic)
                - video_path (str): Absolute path to the video file.
                - camera_intrinsic (np.ndarray or None): 3x3 camera intrinsic matrix.
        """

        # 1. Retrieve stored paths
        video_path = self.trajectory_video_paths[index]
        data_path = self.trajectory_data_paths[index]

        # 2. Parse Parquet data to extract camera intrinsics
        camera_intrinsic = None
        try:
            df = pd.read_parquet(data_path)
            # Flattened format [fx, 0, cx, 0, fy, cy, 0, 0, 1] -> Reshape to (3, 3)
            camera_intrinsic = np.vstack(
                np.array(df["observation.camera_intrinsic"].tolist()[0])
            ).reshape(3, 3)
            print(
                f"Loaded camera intrinsic for trajectory {index}: \n{camera_intrinsic}"
            )
        except Exception as e:
            print(f"Error reading parquet {data_path}: {e}")
            # Caller must handle None return type
            camera_intrinsic = None

        return video_path, camera_intrinsic
