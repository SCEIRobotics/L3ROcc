import fcntl
import json
import os
import shutil

import numpy as np
import pandas as pd

from L3ROcc.base import DataGenerator


class SimpleVideoDataGenerator(DataGenerator):
    """
        Generator designed for standard single-video input.
        Generates a complete InternNav-compliant directory structure including
        full metadata (info.json, episodes.jsonl, tasks.jsonl, AND episodes_stats.jsonl).

    my_custom_dataset/
    └── custom_videos/                  <-- Group Dir
        └── office/                     <-- Scene Dir
            └── trajectory_0/           <-- Trajectory Dir
                ├── data/
                │   └── chunk-000/
                │       ├── episode_000000.parquet
                │       ├── all_occ.npz
                │       └── origin_pcd.ply
                └── videos/
                    └── chunk-000/
                        ├── observation.occ.mask/
                        │   └── mask_sequence.npz
                        ├── observation.occ.view/
                        │   └── occ_sequence.npz
                        └── observation.video.trajectory/
                            └── office.mp4
                │       └── origin_pcd.ply
                └── meta/
                    └── episode_stats.jsonl
                    └── episode.jsonl
                    └── info.json
                    └── tasks.jsonl
    """

    def __init__(self, config_path, save_dir, model_dir):
        super().__init__(config_path, save_dir, model_dir)
        self.default_group = "custom_videos"
        self.default_traj_name = "trajectory_0"

    def get_io_paths(self, input_path):
        # 1. Basic IDs
        video_name = os.path.splitext(os.path.basename(input_path))[0]
        self.scene_id = video_name
        self.episode_id = 0

        # 2. Directory Structure
        self.traj_root = os.path.join(
            self.save_dir, self.default_group, self.scene_id, self.default_traj_name
        )

        self.data_chunk_dir = os.path.join(self.traj_root, "data", "chunk-000")
        self.video_chunk_root = os.path.join(self.traj_root, "videos", "chunk-000")
        self.meta_dir = os.path.join(self.traj_root, "meta")

        self.video_obs_dir = os.path.join(
            self.video_chunk_root, "observation.video.trajectory"
        )
        self.occ_view_dir = os.path.join(self.video_chunk_root, "observation.occ.view")
        self.occ_mask_dir = os.path.join(self.video_chunk_root, "observation.occ.mask")

        for d in [
            self.data_chunk_dir,
            self.video_obs_dir,
            self.occ_view_dir,
            self.occ_mask_dir,
            self.meta_dir,
        ]:
            if not os.path.exists(d):
                os.makedirs(d)

        # 3. File Paths
        paths = {
            "ply": os.path.join(self.data_chunk_dir, "origin_pcd.ply"),
            "global_occ": os.path.join(self.data_chunk_dir, "all_occ.npz"),
            "parquet": os.path.join(self.data_chunk_dir, "episode_000000.parquet"),
            "occ_seq": os.path.join(self.occ_view_dir, "occ_sequence.npz"),
            "mask_seq": os.path.join(self.occ_mask_dir, "mask_sequence.npz"),
            "target_video_path": os.path.join(
                self.video_obs_dir, os.path.basename(input_path)
            ),
            # --- Meta Files ---
            "meta_info_json": os.path.join(self.meta_dir, "info.json"),
            "meta_episodes_jsonl": os.path.join(self.meta_dir, "episodes.jsonl"),
            "meta_tasks_jsonl": os.path.join(self.meta_dir, "tasks.jsonl"),
            "meta_episodes_stats_jsonl": os.path.join(
                self.meta_dir, "episodes_stats.jsonl"
            ),
        }
        return paths

    def update_metadata(
        self, paths, all_camera_poses, all_camera_intrinsics, input_path
    ):
        """
        Generates full metadata suite: Parquet, info, episodes, tasks, stats.
        """

        # --- 0. Prepare Data Dictionary (Used for Parquet and Stats) ---
        total_frames = len(all_camera_poses)

        data_dict = {
            #'episode_id': np.array([self.episode_id] * total_frames, dtype=np.int64),
            # ,
            "index": np.arange(total_frames, dtype=np.int64),  # alias for frame_id
            "observation.camera_intrinsic": all_camera_intrinsics,
            "observation.camera_extrinsic": all_camera_poses,
            "action": all_camera_poses,  # Assuming action is pose
            "observation.camera_extrinsic_occ": all_camera_poses,
            "observation.camera_intrinsic_occ": all_camera_intrinsics,
            "timestamp": np.linspace(
                0, total_frames / self.fps, total_frames, dtype=np.float32
            ),
            "frame_index": np.arange(total_frames, dtype=np.int64),
            "episode_index": np.zeros(total_frames, dtype=np.int64),
            "task_index": np.zeros(total_frames, dtype=np.int64),  # Dummy task index
        }

        # --- 1. Save Parquet ---
        parquet_path = paths.get("parquet")

        # Convert structured arrays to flat lists if necessary for Parquet compatibility
        # Or keep as is if pyarrow handles it. Usually pyarrow handles lists of lists fine.
        parquet_data = data_dict.copy()
        try:
            pd.DataFrame(parquet_data).to_parquet(parquet_path, engine="pyarrow")
            print(f"Parquet saved: {parquet_path}")
        except Exception as e:
            print(f"Failed to save parquet: {e}")
            raise e

        # --- 2. Generate episodes_stats.jsonl ---
        self._save_stats_jsonl(paths["meta_episodes_stats_jsonl"], data_dict)

        # --- 3. Generate info.json ---
        self._save_info_json(paths["meta_info_json"], total_frames)

        # --- 4. Generate episodes.jsonl ---
        rel_video_path = os.path.relpath(paths["target_video_path"], self.traj_root)
        rel_data_path = os.path.relpath(parquet_path, self.traj_root)
        self._save_episodes_jsonl(
            paths["meta_episodes_jsonl"], rel_video_path, rel_data_path, total_frames
        )

        # --- 5. Generate tasks.jsonl ---
        self._save_tasks_jsonl(paths["meta_tasks_jsonl"])

    def _compute_stats(self, key, values):
        """
        Helper to compute min, max, mean, std, count for a sequence of data.
        Returns result in the format: {"min": [[val]], "max": [[val]], ...}
        """
        # Convert list of arrays to a single numpy stack: (N, ...)
        if isinstance(values, list):
            arr = np.stack(values)
        else:
            arr = np.array(values)

        # If 1D array (scalars), ensure it's (N, 1) for consistent stats shape
        if arr.ndim == 1:
            arr = arr[:, None]

        stats = {
            "min": [np.min(arr, axis=0).tolist()],
            "max": [np.max(arr, axis=0).tolist()],
            "mean": [np.mean(arr, axis=0).tolist()],
            "std": [np.std(arr, axis=0).tolist()],
            "count": [len(arr)],
        }
        return stats

    def _save_stats_jsonl(self, jsonl_path, data_dict):
        """
        Calculates statistics for all features and saves to jsonl.
        """
        stats_content = {}

        # Iterate over all keys in our data dictionary
        for key, val in data_dict.items():
            # Skip keys that don't need stats or are redundant
            if key in ["episode_id"]:
                continue

            try:
                stats_content[key] = self._compute_stats(key, val)
            except Exception as e:
                print(f"Warning: Could not compute stats for {key}: {e}")

        # Wrap in the episode index structure
        final_entry = {"episode_index": self.episode_id, "stats": stats_content}

        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(final_entry) + "\n")
        print(f"Stats saved: {jsonl_path}")

    def _save_info_json(self, json_path, total_frames):
        def mat_names(rows, cols, prefix):
            return [f"{prefix}_{i}_{j}" for i in range(rows) for j in range(cols)]

        meta_content = {
            "codebase_version": "v2.1",
            "robot_type": "unknown",
            "total_episodes": 1,
            "total_frames": total_frames,
            "total_tasks": 1,
            "total_videos": 1,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": getattr(self, "fps", 30),
            "splits": {"train": "0:1"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.camera_intrinsic": {
                    "dtype": "float32",
                    "shape": [3, 3],
                    "names": mat_names(3, 3, "intrinsic"),
                },
                "observation.camera_extrinsic": {
                    "dtype": "float32",
                    "shape": [4, 4],
                    "names": mat_names(4, 4, "extrinsic"),
                },
                "action": {
                    "dtype": "float32",
                    "shape": [4, 4],
                    "names": mat_names(4, 4, "action"),
                },
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
                "observation.camera_extrinsic_occ": {
                    "dtype": "float32",
                    "shape": [4, 4],
                    "names": mat_names(4, 4, "extrinsic"),
                },
                "observation.camera_intrinsic_occ": {
                    "dtype": "float32",
                    "shape": [3, 3],
                    "names": mat_names(3, 3, "intrinsic"),
                },
            },
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta_content, f, indent=4)
        print(f"Meta info saved: {json_path}")

    def _save_episodes_jsonl(self, jsonl_path, rel_video_path, rel_data_path, length):
        episode_entry = {
            "episode_index": self.episode_id,
            "tasks": [0],
            "scene_id": self.scene_id,
            "trajectory_id": self.default_traj_name,
            "episode_chunk": 0,
            "video_key": "observation.video.trajectory",
            "data_path": rel_data_path,
            "video_path": rel_video_path,
            "length": length,
            "scale": 1.0,
        }
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(episode_entry) + "\n")
        print(f"Episodes index saved: {jsonl_path}")

    def _save_tasks_jsonl(self, jsonl_path):
        task_entry = {
            "task_index": 0,
            # "episode_index": self.episode_id,
            "task": "Navigate through the custom scene.",
        }
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(task_entry) + "\n")
        print(f"Tasks index saved: {jsonl_path}")

    def _copy_video_to_structure(self, input_path, target_path):
        if os.path.exists(target_path):
            return
        try:
            shutil.copy2(input_path, target_path)
        except Exception as e:
            print(f"Warning: Failed to copy video file: {e}")

    def run_pipeline(self, input_path, pcd_save=True, mesh=False):
        print(f"Processing video: {input_path}")

        pcd, self.camera_pose, self.norm_cam_ray = self.pcd_reconstruction(input_path)

        if pcd_save:
            print("Computing 4D sequence data...")
            paths = self.get_io_paths(input_path)
            self._copy_video_to_structure(input_path, paths["target_video_path"])

            # Compute sequence data
            arr_4d_occ, arr_4d_mask, all_poses, all_intrinsics = (
                self.compute_sequence_data(pcd, mesh=mesh)
            )

            # Save global data
            self.save_global_data(paths)

            # Save sequence data
            self.save_sequence_data(paths, arr_4d_occ, arr_4d_mask)

            # Save all metadata files
            self.update_metadata(paths, all_poses, all_intrinsics, input_path)

        print(f"Process Complete! Dataset ready at: {self.traj_root}")


# ================= Usage Example =================

if __name__ == "__main__":
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    config_path = "./L3ROcc/configs/config.yaml"
    save_root = "./outputs/my_custom_dataset"
    model_dir = "./ckpt"
    video_path = "./inputs/office.mp4"

    if os.path.exists(video_path):
        generator = SimpleVideoDataGenerator(config_path, save_root, model_dir)
        generator.run_pipeline(video_path, pcd_save=True)
    else:
        print(f"Video not found: {video_path}")
