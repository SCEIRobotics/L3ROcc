import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys 
import numpy as np
import traceback

from occ.generater.intern_vln_env import InternNavDataGenerator
from occ.dataset.intern_nav_adapter import InternNavSequenceLoader
sys.path.append('.') 

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

def run_dataset_pipeline():
    # ================= 1. 配置参数 =================
    
    # 数据集根目录 
    dataset_root = "/mnt/data/huangbinling/project/occgen/all_outputs/debug/debug_seg_fault/traj_data/" 
    
    # 输出结果的根目录
    output_root = "/mnt/data/huangbinling/project/occgen/all_outputs/debug/debug_seg_fault/traj_data/"
    
    # 模型 Checkpoint 路径
    model_dir = os.path.join(project_root, "ckpt")
    
    # 配置文件路径
    config_path = os.path.join(project_root, "occ","configs", "config.yaml")
    
    # ================= 2. 初始化 =================
    
    print(f" 初始化数据加载器，扫描路径: {dataset_root} ...")
    loader = InternNavSequenceLoader(dataset_root)
    print(f"扫描完成，共发现 {len(loader)} 条轨迹。")

    print("初始化 OCC 生成器...")
    generator = InternNavDataGenerator(
        config_path=config_path, 
        save_dir=output_root, # 临时根目录，后面会针对每条轨迹修改
        model_dir=model_dir
    )
    
    # ================= 3. 开始循环处理 =================
    for i in range(len(loader)): 
        try:
            # A. 从加载器获取信息
            video_path, cam_intrinsics = loader.get_trajectory_info(i)
            if video_path is None:
                print(f"⚠️ 跳过第 {i} 条轨迹：未找到视频文件。")
                continue

            # DataGenerator 需要 input_path 指向视频文件本身
            # 所以我们传入 video_path (即 .../observation.video.trajectory/0.mp4
            input_path_for_gen = video_path

            # B. 构建该条轨迹的专属输出路径
            # 逻辑：output_root / group_name / scene_id / trajectory_id
            # 我们利用 video_path 的目录结构来反推
            # video_path: .../traj_data/3dfront/scene_abc/traj_1/videos/...
            path_parts = video_path.split(os.sep)
            

            try:
                start_idx = path_parts.index("traj_data") + 1
                # 取出 group, scene, traj (例如: 3dfront_d435i, 00154..., trajectory_1)
                relative_path = os.path.join(*path_parts[start_idx : start_idx+3])  #3dfront_d435i/00154.../trajectory_1
            except ValueError:
                # 如果路径里没有 traj_data，就用简单的索引命名
                relative_path = f"trajectory_{i:06d}"

            current_save_dir = os.path.join(output_root, relative_path) #/mnt/.../traj_data/3dfront_d435i/00154.../trajectory_1
            
            # 创建输出目录
            if not os.path.exists(current_save_dir):
                os.makedirs(current_save_dir)

            print(f"\n[{i+1}/{len(loader)}] 正在处理: {relative_path}")
            print(f"   输入: {input_path_for_gen}")
            print(f"   输出: {current_save_dir}")

            # C. 注入参数给 Generator
            # 1. 覆盖 save_path (让结果存到对应的子文件夹)
            generator.save_path = current_save_dir
            
            # 2. 注入真实内参 (如果有)
            if cam_intrinsics is not None:
                generator.camera_intric = cam_intrinsics.astype(np.float32)
            else:
                print("未读取到 Parquet 内参，使用默认值。")

            # 3. 清空历史缓存 (防止上一条轨迹影响这一条)
            if hasattr(generator, 'occ_history_buffer'):
                generator.occ_history_buffer.clear()

            # D. 运行核心管线
            # pcd_save=True 才会执行保存逻辑
            generator.run_pipeline(input_path_for_gen, pcd_save=True)
            
            print("处理成功！")

        except Exception as e:
            print(f"处理失败: {e}")
            traceback.print_exc()
            continue

if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    # 设置 GPU (如果有多卡)
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    
    run_dataset_pipeline()