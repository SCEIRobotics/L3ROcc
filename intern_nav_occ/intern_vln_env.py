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

# from pi3.utils.geometry import homogenize_points
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3
from pi3.utils.basic import (
    load_images_as_tensor,
    load_depths_as_tensor,
    write_ply,
)  # Assuming you have a helper function

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
        """生成并创建输出所需的所有路径字典"""
        
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


    def update_metadata(self, paths, all_poses, all_intrinsics, input_path):
        # --- 更新 Parquet ---
        parquet_path = paths.get('parquet')
        if parquet_path and os.path.exists(parquet_path):
            print(f"Updating Parquet: {parquet_path}")
            try:
                df = pd.read_parquet(parquet_path, engine='pyarrow')
                curr_len = len(df)
                
                # 截取或填充数据以匹配 DataFrame 长度
                poses_to_save = all_poses[:curr_len]
                intrs_to_save = all_intrinsics[:curr_len]
                
                if len(poses_to_save) < curr_len:
                    poses_to_save.extend([None] * (curr_len - len(poses_to_save)))
                    intrs_to_save.extend([None] * (curr_len - len(intrs_to_save)))
                
                df['observation.camera_extrinsic_occ'] = poses_to_save
                df['observation.camera_intrinsic_occ'] = intrs_to_save
                
                df.to_parquet(parquet_path, engine='pyarrow')
                print("Parquet updated.")
            except Exception as e:
                print(f"Parquet update failed: {e}")
        else:
            print(f"⚠️ Parquet file not found at {parquet_path}")

        # --- 更新 JSON ---
        # 寻找 info.json (通常在 traj_root/meta/info.json)
        traj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(input_path))))
        json_path = os.path.join(traj_root, "meta", "info.json")
        
        if os.path.exists(json_path):
            print(f"Updating JSON: {json_path}")
            try:
                with open(json_path, 'r') as f: meta = json.load(f)
                
                feat_ext = {"dtype": "float32", "shape": [4, 4], "names": [f"extrinsic_{i}_{j}" for i in range(4) for j in range(4)]}
                feat_int = {"dtype": "float32", "shape": [3, 3], "names": [f"intrinsic_{i}_{j}" for i in range(3) for j in range(3)]}   
                
                if "features" in meta:
                    meta["features"]["observation.camera_extrinsic_occ"] = feat_ext
                    meta["features"]["observation.camera_intrinsic_occ"] = feat_int
                    
                    with open(json_path, 'w') as f: json.dump(meta, f, indent=4)
                    print("JSON updated.")
            except Exception as e:
                print(f"JSON update failed: {e}")

# ================= 主函数 =================

if __name__ == "__main__":

    config_path = "./occ/config.yaml"
    save_dir = "./outputs"
    model_dir = "./ckpt"

    input_path = "/mnt/data/huangbinling/project/occgen/inputs"
    video_name = "office.mp4"
    generator = InternNavDataGenerator(config_path, save_dir, model_dir)
    generator.run_pipeline(os.path.join(input_path, video_name), pcd_save=True)


