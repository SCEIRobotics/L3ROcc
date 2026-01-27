import fcntl
import json
import os
import time

import numpy as np
import pandas as pd

from occ.base import DataGenerator


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
    ):
        """
        Initialize the InternNavDataGenerator.

        Args:
            config_path (str): Path to the YAML configuration file.
            save_dir (str): Root directory where outputs will be saved.
            model_dir (str): Directory containing model checkpoints.
        """
        super().__init__(config_path, save_dir, model_dir)

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

    def get_gt_poses(self, input_path):
        """
        Parses Ground Truth (GT) camera trajectories specific to the InternNav dataset.

        It attempts to locate the `episode_000000.parquet` file relative to the input path
        or the save directory to extract the 'action' column containing pose matrices.

        Args:
            input_path (str): Path to the input source used to locate the trajectory root.

        Returns:
            np.ndarray or None:
                - If successful, returns an array of shape (N, 4, 4) representing camera poses.
                - If the file is missing or data is invalid, returns None.
        """
        try:
            traj_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(input_path)))
            )
            origin_parquet_path = os.path.join(
                traj_root, "data", "chunk-000", "episode_000000.parquet"
            )

            # Fallback to save_path if original not found
            if not os.path.exists(origin_parquet_path):
                origin_parquet_path = os.path.join(
                    self.save_path, "data", "chunk-000", "episode_000000.parquet"
                )
                if not os.path.exists(origin_parquet_path):
                    return None

            df = pd.read_parquet(origin_parquet_path, engine="pyarrow")
            col_name = "action"
            if col_name not in df.columns:
                return None

            gt_raw = df[col_name].tolist()
            gt_poses_np = []

            for p in gt_raw:
                if p is None:
                    continue
                mat = np.array(p)

                # Handle varying matrix shapes/flattened arrays
                if mat.shape == (4, 4):
                    pass
                elif mat.ndim == 1 and (mat.shape[0] == 4 or mat.shape[0] == 3):
                    try:
                        mat = np.vstack(mat)
                    except:
                        continue

                if mat.size == 16:
                    mat = mat.reshape(4, 4)
                elif mat.size == 12:
                    mat = mat.reshape(3, 4)
                    mat = np.vstack([mat, [0, 0, 0, 1]])

                if mat.shape != (4, 4):
                    continue

                gt_poses_np.append(mat)

            if len(gt_poses_np) == 0:
                return None

            return np.array(gt_poses_np)

        except Exception as e:
            print(f"[Subclass Error] Failed to load GT poses: {e}")
            raise e

    def update_metadata(self, paths, all_poses, all_intrinsics, input_path):
        """
        Updates Parquet and JSON metadata files with generated camera parameters.

        Args:
            paths (dict): Dictionary of file paths (output of `get_io_paths`).
            all_poses (list or np.ndarray): Generated camera extrinsic matrices (N, 4, 4).
            all_intrinsics (list or np.ndarray): Camera intrinsic matrices (N, 3, 3).
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
                gen_len = len(all_poses)

                # Validate data consistency
                if gen_len != curr_len:
                    raise ValueError(
                        f"[Length Mismatch] Parquet has {curr_len} frames, "
                        f"but generated poses have {gen_len} frames."
                    )

                df["observation.camera_extrinsic_occ"] = all_poses
                df["observation.camera_intrinsic_occ"] = all_intrinsics

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

    def run_pipeline(self, input_path, pcd_save=True):
        """
        Executes the full data generation pipeline:
        Reconstruction -> GT Scale Alignment -> Global Storage -> Sequence Calculation -> Metadata Update

        Args:
            input_path (str): Path to the input video file.
            pcd_save (bool, optional): Whether to save 3D artifacts (point cloud, etc.). Defaults to True.

        Returns:
            None
        """
        # Initialization
        if self.camera_intric is None:
            self.camera_intric = np.array(
                [[168.0, 0, 240], [0, 192.0, 135], [0, 0, 1]], dtype=np.float32
            )

        # 3D Reconstruction
        pcd, self.camera_pose, self.norm_cam_ray = self.pcd_reconstruction(input_path)

        # Align with Ground Truth Scale
        # self.camera_pose and pcd are updated to the aligned scale here
        pcd, scale = self.align_with_gt_scale(input_path, pcd)
        print(f"[Scale Info] Aligned with target scale: {scale:.4f}")

        self.update_meta_episodes_jsonl(scale)

        # Convert to occupancy (pcd is maintained at aligned scale)
        self.occ_pcd = self.pcd_to_occ(self.pcd)

        if not pcd_save:
            return

        print("Start processing sequence frames...")

        paths = self.get_io_paths(input_path)

        # Save global data
        self.save_global_data(paths)

        # Execute core computation
        arr_4d_occ, arr_4d_mask, all_camera_poses, all_camera_intrinsics = (
            self.compute_sequence_data()
        )

        # Save sequence data
        print("Saving 4D Sequence Arrays...")
        self.save_sequence_data(paths, arr_4d_occ, arr_4d_mask)

        # Update metadata
        self.update_metadata(paths, all_camera_poses, all_camera_intrinsics, input_path)
