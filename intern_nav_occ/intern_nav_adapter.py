import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

class InternNavSequenceLoader:
    def __init__(self, root_dirs):
        self.root_dirs = root_dirs
        self.trajectory_dirs = []       # 存储每条轨迹的根目录
        self.trajectory_rgb_paths = []  # 存储对应的Rgb文件夹路径
        self.trajectory_data_paths = [] # 存储对应的parquet文件夹路径
        self.trajectory_video_paths = [] # 存储对应的视频文件路径

        self._scan_dataset()

    def _scan_dataset(self):
        """遍历数据集目录，找到所有有效的轨迹"""
        print(f"Scanning dataset in {self.root_dirs}...")

        # 1. 遍历场景组 (如 gibson_zed, 3dfront_d435i)
        group_dirs = [d for d in os.listdir(self.root_dirs) if os.path.isdir(os.path.join(self.root_dirs, d))]

        for group_dir in group_dirs:
            group_path = os.path.join(self.root_dirs, group_dir)
            scene_dirs = os.listdir(group_path)

            # 2. 遍历场景(如 00154c06...)
            for scene_dir in scene_dirs:
                scene_path = os.path.join(group_path, scene_dir)
                if not os.path.isdir(scene_path): continue

                traj_dirs = os.listdir(scene_path)

                # 3. 遍历轨迹(如 trajectory_1)
                for traj_dir in traj_dirs:
                    entire_task_dir = os.path.join(scene_path, traj_dir)
                    
                    # 构造关键路径
                    rgb_dir = os.path.join(entire_task_dir, "videos/chunk-000/observation.images.rgb/")
                    data_path = os.path.join(entire_task_dir, 'data/chunk-000/episode_000000.parquet')
                    
                    # 视频文件夹路径 
                    video_folder_path = os.path.join(entire_task_dir, 'videos/chunk-000/observation.video.trajectory')
                    
                    # 寻找具体的视频文件
                    video_file_path = None
                    if os.path.exists(video_folder_path):
                        if os.path.isfile(video_folder_path) and video_folder_path.endswith('.mp4'):
                            video_file_path = video_folder_path
                        elif os.path.isdir(video_folder_path):
                            # 遍历文件夹找到 .mp4 文件 
                            files = os.listdir(video_folder_path)
                            for f in files:
                                if f.endswith('.mp4'):
                                    video_file_path = os.path.join(video_folder_path, f)
                                    break
                    
                    # 校验文件是否存在
                    if os.path.exists(rgb_dir) and os.path.exists(data_path) and video_file_path:
                        self.trajectory_dirs.append(entire_task_dir)
                        self.trajectory_rgb_paths.append(rgb_dir)
                        self.trajectory_data_paths.append(data_path)
                        self.trajectory_video_paths.append(video_file_path)

        print(f"Found {len(self.trajectory_dirs)} valid trajectories.")
    
    def __len__(self):
        return len(self.trajectory_dirs)

    def get_trajectory_info(self, index):
        """
        获取指定索引的轨迹信息
        Args:
            index: 轨迹的序号 (第几条轨迹)
        Returns: 
            video_path: 该轨迹对应的原视频路径 (绝对路径)
            camera_intrinsic: 该轨迹的相机内参矩阵 (3x3)
        """
        
        # 1. 获取路径
        video_path = self.trajectory_video_paths[index]
        data_path = self.trajectory_data_paths[index] 
        
        # 2. 解析 Parquet 获取内参
        camera_intrinsic = None
        try:
            df = pd.read_parquet(data_path)
            # 提取相机内参 [fx, 0, cx, 0, fy, cy, 0, 0, 1] -> (3,3)
            camera_intrinsic = np.vstack(np.array(df['observation.camera_intrinsic'].tolist()[0])).reshape(3, 3)
        except Exception as e:
            print(f"Error reading parquet {data_path}: {e}")
            # 如果读取失败，返回 None，外部代码需要处理 None 的情况
            camera_intrinsic = None 

        return video_path, camera_intrinsic