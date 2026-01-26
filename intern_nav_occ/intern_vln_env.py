import torch
import os
import random
import yaml
import time
import glob
import numpy as np
import open3d as o3d
from collections import deque
from scipy import sparse
import pandas as pd
import json
import fcntl

# from pi3.utils.geometry import homogenize_points
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3
from pi3.utils.basic import (
    load_images_as_tensor,
    load_depths_as_tensor,
    write_ply,
)

from vln_env import DataGenerator


class InternNavDataGenerator(DataGenerator):
    def __init__(
        self,
        config_path,
        save_dir,
        model_dir,
    ):
        super().__init__(config_path, save_dir, model_dir)

    
    def get_io_paths(self, input_path):
        """生成并创建针对 InternNav 数据集输出所需的所有路径字典"""
        
        # 1. 构建目录
        data_chunk_dir = os.path.join(self.save_path, "data", "chunk-000")
        video_chunk_dir = os.path.join(self.save_path, "videos", "chunk-000")
        occ_view_dir = os.path.join(video_chunk_dir, "observation.occ.view")
        occ_mask_dir = os.path.join(video_chunk_dir, "observation.occ.mask")

        for d in [data_chunk_dir, video_chunk_dir, occ_view_dir, occ_mask_dir]:
            if not os.path.exists(d): os.makedirs(d)

        # 2. 定义文件路径
        paths = {
            'ply': os.path.join(data_chunk_dir, "origin_pcd.ply"),
            'global_occ': os.path.join(data_chunk_dir, "all_occ.npz"),
            'parquet': os.path.join(data_chunk_dir, "episode_000000.parquet"),
            'occ_seq': os.path.join(occ_view_dir, "occ_sequence.npz"),
            'mask_seq': os.path.join(occ_mask_dir, "mask_sequence.npz")
        }
        return paths

    def get_target_poses(self, input_path):
        """
        针对InternNav数据集设计的GT相机轨迹解析函数
        """
        try:
            traj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(input_path))))
            origin_parquet_path = os.path.join(traj_root, 'data', 'chunk-000', 'episode_000000.parquet')
            
            if not os.path.exists(origin_parquet_path):
                origin_parquet_path = os.path.join(self.save_path, 'data', 'chunk-000', 'episode_000000.parquet')
                if not os.path.exists(origin_parquet_path):
                    return None
            df = pd.read_parquet(origin_parquet_path, engine='pyarrow')
            col_name = 'action'
            if col_name not in df.columns:
                return None

            gt_raw = df[col_name].tolist()
            gt_poses_np = []
            
            for p in gt_raw:
                if p is None: continue
                mat = np.array(p)
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
        """Update Parquet and JSON files for InternNav dataset"""
        
        # --- Update Parquet (Per trajectory, usually safe, but good to be careful) ---
        # Note: Parquet doesn't support simple text locking easily. 
        # Since 'parquet_path' is usually specific to the chunk/trajectory (episode_000000.parquet),
        # it is generally safe in multi-processing IF chunks are processed by unique workers.
        # If multiple workers touch the SAME parquet file, you need a different locking strategy.
        
        parquet_path = paths.get('parquet')
        if parquet_path and os.path.exists(parquet_path):
            print(f"Updating Parquet: {parquet_path}")
            try:
                df = pd.read_parquet(parquet_path, engine='pyarrow')
                curr_len = len(df)
                gen_len = len(all_poses)
                
                # Check length consistency
                if gen_len != curr_len:
                    raise ValueError(
                        f"[Length Mismatch] Parquet has {curr_len} frames, "
                        f"but generated poses have {gen_len} frames."
                    )
                
                df['observation.camera_extrinsic_occ'] = all_poses
                df['observation.camera_intrinsic_occ'] = all_intrinsics
                
                df.to_parquet(parquet_path, engine='pyarrow')
                print("Parquet updated.")
                
            except Exception as e:
                print(f"Parquet update failed: {e}")
                raise e
        else:
            print(f"Parquet file not found at {parquet_path}")

        # --- Update JSON (SHARED RESOURCE - NEEDS LOCK) ---
        traj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(input_path))))
        json_path = os.path.join(traj_root, "meta", "info.json")
        
        # Define the logic to update the JSON data
        def update_info_logic(meta):
            feat_ext = {"dtype": "float32", "shape": [4, 4], "names": [f"extrinsic_{i}_{j}" for i in range(4) for j in range(4)]}
            feat_int = {"dtype": "float32", "shape": [3, 3], "names": [f"intrinsic_{i}_{j}" for i in range(3) for j in range(3)]}   
            
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
        Read meta/episodes.jsonl and write the calculated scale.
        """
        import json
        
        meta_dir = os.path.join(self.save_path, "meta")
        jsonl_path = os.path.join(meta_dir, "episodes.jsonl")

        if not os.path.exists(jsonl_path):
            print(f"[Meta Warning] episodes.jsonl not found at: {jsonl_path}")
            return

        # Retry mechanism for locking
        for _ in range(5):
            try:
                # Open with r+ mode
                with open(jsonl_path, 'r+', encoding='utf-8') as f:
                    # [LOCK]
                    fcntl.flock(f, fcntl.LOCK_EX)
                    
                    try:
                        # [READ]
                        lines = f.readlines()
                        entries = [json.loads(line) for line in lines if line.strip()]
                        
                        # [MODIFY]
                        scale_val = float(scale)
                        for entry in entries:
                            entry['scale'] = scale_val

                        # [WRITE]
                        f.seek(0)
                        f.truncate()
                        for entry in entries:
                            f.write(json.dumps(entry) + '\n')
                        f.flush()
                        os.fsync(f.fileno())
                        
                        print(f"[Meta Updated] Saved scale ({scale_val:.4f}) to {jsonl_path}")
                        
                    finally:
                        # [UNLOCK]
                        fcntl.flock(f, fcntl.LOCK_UN)
                
                break # Success
            
            except BlockingIOError:
                time.sleep(0.1)
            except Exception as e:
                print(f"[Meta Error] Failed to update episodes.jsonl: {e}")
                # You might choose to raise e here depending on how critical this is
                raise e
                
    def _update_json_safely(self, file_path, update_func):
        """
        Generic safe update function: uses file locking to prevent multi-process conflicts.
        """
        if not os.path.exists(file_path):
            return

        # Retry mechanism
        for _ in range(5):
            try:
                # Open with 'r+' mode for reading and writing
                with open(file_path, 'r+', encoding='utf-8') as f:
                    # Acquire exclusive lock (blocks if locked by others)
                    fcntl.flock(f, fcntl.LOCK_EX)
                    
                    try:
                        # Handle empty files just in case
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError:
                            data = {}

                        # Call the callback function
                        new_data = update_func(data)
                        
                        if new_data is not None:
                            f.seek(0)        # Reset pointer
                            f.truncate()     # Clear content
                            json.dump(new_data, f, indent=4)
                            f.flush()        # Flush buffer
                            os.fsync(f.fileno()) # Sync to disk
                        
                    finally:
                        # Always release the lock
                        fcntl.flock(f, fcntl.LOCK_UN)
                
                break # Success
                
            except BlockingIOError:
                time.sleep(0.1) # Wait if locked
            except Exception as e:
                print(f"Error updating {file_path}: {e}")
                break

# ================= 主函数 =================

if __name__ == "__main__":

    config_path = "./occ/config.yaml"
    save_dir = "./outputs"
    model_dir = "./ckpt"

    input_path = "/mnt/data/huangbinling/project/occgen/inputs"
    video_name = "office.mp4"
    generator = InternNavDataGenerator(config_path, save_dir, model_dir)
    generator.run_pipeline(os.path.join(input_path, video_name), pcd_save=True)