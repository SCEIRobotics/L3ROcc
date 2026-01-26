import numpy as np
import os
import argparse

def calculate_mask_stats(mask_path):
    print(f"📂 Loading Mask: {mask_path}")
    
    if not os.path.exists(mask_path):
        print("❌ File not found!")
        return

    # 1. 加载压缩数据 (只加载到内存，不解压)
    try:
        loader = np.load(mask_path)
        packed_data = loader['data']   # (N, Packed_Size)
        shape = loader['shape']        # (H, W, D)
        
        N = packed_data.shape[0]
        H, W, D = shape
        voxels_per_frame = H * W * D
        total_voxels_all = N * voxels_per_frame
        
        print(f"   Shape: {shape} x {N} frames")
        print(f"   Total Voxels (Space): {total_voxels_all:,}")
        
    except Exception as e:
        print(f"❌ Load failed: {e}")
        return

    # 2. 流式计数 (防止内存爆炸)
    print("🚀 Counting bits frame by frame...")
    
    total_ones = 0
    
    # 进度条
    for i in range(N):
        # A. 取出一帧的压缩数据
        packed_frame = packed_data[i]
        
        # B. 解压 (CPU 瞬间完成，内存占用极小)
        unpacked = np.unpackbits(packed_frame)
        
        # C. 截断填充位 (关键！)
        # packbits 会在末尾补0凑齐8位，我们要把多余的切掉，否则 0 的统计会虚高
        valid_bits = unpacked[:voxels_per_frame]
        
        # D. 统计 1 的个数
        total_ones += np.sum(valid_bits)
        
        if i % 20 == 0:
            print(f"   Processed frame {i}/{N}...", end='\r')

    print(f"   Processed frame {N}/{N} (Done).")

    # 3. 计算结果
    total_zeros = total_voxels_all - total_ones
    
    # 防止除以0
    ratio = total_zeros / total_ones if total_ones > 0 else float('inf')
    percent_ones = (total_ones / total_voxels_all) * 100
    percent_zeros = (total_zeros / total_voxels_all) * 100

    print("\n" + "="*40)
    print("       MASK STATISTICS REPORT")
    print("="*40)
    print(f"🟦 Count of 1s (Visible):  {total_ones:,}")
    print(f"⬜ Count of 0s (Unknown):  {total_zeros:,}")
    print("-" * 40)
    print(f"📊 Ratio (0 : 1)        :  {ratio:.2f} : 1")
    print(f"📈 Percentage of 1s     :  {percent_ones:.2f}%")
    print(f"📉 Percentage of 0s     :  {percent_zeros:.2f}%")
    print("="*40)

    # 结论分析
    if ratio > 10:
        print("💡 结论: 数据非常稀疏！存储为压缩格式非常划算。")
    elif ratio < 1:
        print("💡 结论: 数据非常稠密（这不符合视锥体常理，请检查代码）。")
    else:
        print("💡 结论: 数据密度适中。")

if __name__ == "__main__":
    # 替换为你实际的文件路径
    PATH = "/mnt/data/huangbinling/project/occgen/small_vln_n1_split_e3/traj_data/3d/111/trajectory_15/videos/chunk-000/observation.occ.mask/mask_sequence_csr_final.npz"
    calculate_mask_stats(PATH)