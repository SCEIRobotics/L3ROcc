import os
import numpy as np
import open3d as o3d
import glob
from tqdm import tqdm

def analyze_dataset_bounds(dataset_root, sample_ratio=0.1):
    """
    遍历数据集，计算合理的 pc_range
    dataset_root: 数据集根目录 (包含 group/scene/traj...)
    """
    print(f"正在扫描数据集: {dataset_root} ...")
    
    # 寻找所有的 origin_pcd.ply
    # 路径模式: .../data/chunk-000/origin_pcd.ply
    search_pattern = os.path.join(dataset_root, "**", "origin_pcd.ply")
    ply_files = glob.glob(search_pattern, recursive=True)
    
    if not ply_files:
        print("❌ 未找到任何 origin_pcd.ply 文件，请检查路径！")
        return

    print(f"找到 {len(ply_files)} 个点云文件。正在采样分析...")
    
    min_bound_global = np.array([np.inf, np.inf, np.inf])
    max_bound_global = np.array([-np.inf, -np.inf, -np.inf])
    
    # 为了速度，可以只抽样一部分，或者设为 1.0 跑全量
    import random
    random.shuffle(ply_files)
    num_samples = max(1, int(len(ply_files) * sample_ratio))
    
    for i in tqdm(range(num_samples)):
        ply_path = ply_files[i]
        try:
            pcd = o3d.io.read_point_cloud(ply_path)
            points = np.asarray(pcd.points)
            
            if len(points) == 0: continue
            
            # 计算当前文件的 min/max
            min_b = points.min(axis=0)
            max_b = points.max(axis=0)
            
            # 更新全局 min/max
            min_bound_global = np.minimum(min_bound_global, min_b)
            max_bound_global = np.maximum(max_bound_global, max_b)
            
        except Exception as e:
            print(f"Error reading {ply_path}: {e}")

    print("\n" + "="*40)
    print("📊 数据集统计结果 (单位: 米)")
    print("="*40)
    print(f"X range: {min_bound_global[0]:.2f} ~ {max_bound_global[0]:.2f}")
    print(f"Y range: {min_bound_global[1]:.2f} ~ {max_bound_global[1]:.2f}")
    print(f"Z range: {min_bound_global[2]:.2f} ~ {max_bound_global[2]:.2f}")
    
    # 推荐的 pc_range (稍微留一点余量 padding)
    padding = 2.0 # 留2米余量防止边缘截断
    rec_min = np.floor(min_bound_global - padding)
    rec_max = np.ceil(max_bound_global + padding)
    
    print("\n✅ 推荐的 pc_range 设置:")
    print(f"[{rec_min[0]}, {rec_min[1]}, {rec_min[2]}, {rec_max[0]}, {rec_max[1]}, {rec_max[2]}]")
    
    # 计算场景尺寸
    dims = rec_max - rec_min
    print(f"\n场景最大尺寸: {dims[0]}m x {dims[1]}m x {dims[2]}m")
    
    return dims

if __name__ == "__main__":
    # 修改为你的数据集路径
    DATASET_ROOT = "/mnt/data/huangbinling/project/occgen/e2/traj_data/"
    analyze_dataset_bounds(DATASET_ROOT, sample_ratio=1.0) # 采样 100% 的文件进行估算
