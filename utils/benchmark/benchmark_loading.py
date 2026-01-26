import time
import os
import numpy as np
import scipy.sparse as sparse
import sys


# 1.原始 NPY
PATH_LEGACY_NPY = "/mnt/data/huangbinling/project/occgen/small_vln_n1_split_e3/traj_data/3d/111/trajectory_15/videos/chunk-000/observation.occ.mask/mask_sequence.npy" 

# 2.普通 NPZ 
PATH_LEGACY_NPZ = "/mnt/data/huangbinling/project/occgen/small_vln_n1_split_e3/traj_data/3d/111/trajectory_15/videos/chunk-000/observation.occ.mask/mask_sequence.npz"

# 3.Packbits 优化版 
PATH_OPTIMIZED_PACKED = "/mnt/data/huangbinling/project/occgen/small_vln_n1_split_e3/traj_data/3d/111/trajectory_15/videos/chunk-000/observation.occ.mask/mask_sequence_csr_final.npz"

# ==========================================

def print_stat(name, t_cost, file_size_mb):
    speed = file_size_mb / t_cost if t_cost > 0 else 0
    print(f"[{name}]")
    print(f"   ⏱️ Time Cost : {t_cost:.4f} s")
    print(f"   📦 File Size : {file_size_mb:.2f} MB")
    print(f"   🚀 Speed     : {speed:.2f} MB/s")
    print("-" * 40)

def get_size_mb(path):
    if not os.path.exists(path): return 0
    return os.path.getsize(path) / (1024 * 1024)

# --- 1. 测试读取 .npy ---
def benchmark_npy():
    if not PATH_LEGACY_NPY or not os.path.exists(PATH_LEGACY_NPY):
        print(" Skip NPY test (File not found)")
        return

    print(f"\nTesting Legacy .npy (Warning: High Memory Usage)...")
    try:
        t_start = time.time()
        arr = np.load(PATH_LEGACY_NPY)
        _ = arr.sum() 
        t_cost = time.time() - t_start
        print_stat("Legacy .npy", t_cost, get_size_mb(PATH_LEGACY_NPY))
        del arr # 立即释放
    except Exception as e:
        print(f" NPY Load Failed (OOM?): {e}")

# --- 2. 测试读取 .npz (普通压缩) ---
def benchmark_npz_legacy():
    if not PATH_LEGACY_NPZ or not os.path.exists(PATH_LEGACY_NPZ):
        print("Skip Legacy .npz test (File not found)")
        return

    print(f"\n🐢 Testing Legacy .npz (Standard Compression)...")
    try:
        t_start = time.time()
        # 这步会非常慢，因为 CPU 要解压 11GB 的数据
        with np.load(PATH_LEGACY_NPZ) as loader:
            # 假设保存时的 key 是 'arr_0' 或 'data'
            key = loader.files[0]
            arr = loader[key]
            _ = arr.sum() # 强制读取
        t_cost = time.time() - t_start
        print_stat("Legacy .npz", t_cost, get_size_mb(PATH_LEGACY_NPZ))
    except Exception as e:
        print(f" NPZ Load Failed: {e}")



# --- 3. 测试读取 新版 Packbits ---
def benchmark_optimized():
    if not PATH_OPTIMIZED_PACKED or not os.path.exists(PATH_OPTIMIZED_PACKED):
        print("Skip Optimized test (File not found)")
        return

    print(f"\n⚡ Testing Optimized Packbits...")
    try:
        t_start = time.time()
        
        # A. IO 读取
        loader = np.load(PATH_OPTIMIZED_PACKED)
        packed_data = loader['data']
        shape = loader['shape'] # 可能只有 (H, W, D)
        
        if len(shape) == 3:
            H, W, D = shape
            N = packed_data.shape[0] # 从数据第一维获取帧数
        elif len(shape) == 4:
            N, H, W, D = shape
        else:
            raise ValueError(f"Unknown shape format: {shape}")
        unpacked = np.unpackbits(packed_data, axis=-1)
  
        DO_FULL_RESHAPE = True 
        
        if DO_FULL_RESHAPE:
            mask_dense = unpacked.reshape(N, -1)[:, :H*W*D].reshape(N, H, W, D)
            _ = mask_dense[0,0,0,0]
            del mask_dense # 立即释放内存
        
        t_cost = time.time() - t_start
        print_stat("Optimized Packbits", t_cost, get_size_mb(PATH_OPTIMIZED_PACKED))
        
    except Exception as e:
        print(f" Optimized Load Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("="*40)
    print("       DATA LOADING BENCHMARK")
    print("="*40)
    
    benchmark_npy()
    benchmark_npz_legacy()
    benchmark_optimized()





