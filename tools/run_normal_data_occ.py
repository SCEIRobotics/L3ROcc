import os 
import sys 
import numpy as np
import traceback
from occ.base import DataGenerator
sys.path.append('.') 

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)


def run_normal_data_pipeline():
    # ================= 1. 配置参数 =================
    
    # 数据集根目录 
    input_path = "/mnt/data/huangbinling/project/occgen/inputs/" 

    # 视频文件名
    video_name = "office.mp4"
    
    # 输出结果的根目录
    save_dir = "/mnt/data/huangbinling/project/occgen/outputs/"
    
    # 模型 Checkpoint 路径
    model_dir = os.path.join(project_root, "ckpt")
    
    # 配置文件路径
    config_path = os.path.join(project_root, "occ","configs", "config.yaml")

    
    # ================= 2. 初始化 =================
    generator = DataGenerator(config_path, save_dir, model_dir)
    
    # ================= 3. 开始处理 =================
    generator.run_pipeline(os.path.join(input_path, video_name), pcd_save=True)

if __name__ == "__main__":
    run_normal_data_pipeline()