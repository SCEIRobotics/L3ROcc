import time
import os
import numpy as np
import scipy.sparse as sparse
import random

# ================= 配置路径 =================
OCC_PATH = "/mnt/data/huangbinling/project/occgen/small_vln_n1_split_e3/traj_data/3d/111/trajectory_15/videos/chunk-000/observation.occ.view/occ_sequence_csr_final.npz"
MASK_PATH = "/mnt/data/huangbinling/project/occgen/small_vln_n1_split_e3/traj_data/3d/111/trajectory_15/videos/chunk-000/observation.occ.mask/mask_sequence_csr_final.npz"

GRID_SIZE = (400, 400, 400)
# ===========================================

def benchmark_slicing():
    print(f"🚀 Benchmarking On-the-fly Slicing (Single Frame & Clip)...\n")
    
    # ================= 1. 初始化阶段 (模拟 Dataset.__init__) =================
    # 这一步只加载元数据和压缩数据到内存，不解压
    print("--- [Step 1: Init / Memory Mapping] ---")
    
    t_init_start = time.time()
    
    # OCC: 加载 CSR 结构 (几 MB)
    occ_matrix = sparse.load_npz(OCC_PATH)
    
    # Mask: 加载压缩的 Numpy 数组 (87 MB)
    mask_loader = np.load(MASK_PATH)
    mask_packed = mask_loader['data'] # (N, Packed_Size)
    
    t_init = time.time() - t_init_start
    print(f"✅ Init finished in {t_init:.4f}s")
    print(f"   Total Frames available: {occ_matrix.shape[0]}\n")


    # ================= 2. 单帧读取 (模拟 __getitem__) =================
    idx = random.randint(0, occ_matrix.shape[0] - 1)
    print(f"--- [Step 2: Single Frame Access (Index {idx})] ---")
    
    # --- 测试 OCC 单帧 ---
    t_start = time.time()
    # CSR 切片极其高效，toarray 只分配 64MB 内存
    occ_frame = occ_matrix[idx].toarray().reshape(GRID_SIZE)
    t_occ = time.time() - t_start
    
    # --- 测试 Mask 单帧 ---
    t_start = time.time()
    # 1. 取出压缩的一行
    packed_row = mask_packed[idx] 
    # 2. 解压 (只解压这 8MB 数据)
    unpacked = np.unpackbits(packed_row)
    # 3. Reshape
    H, W, D = GRID_SIZE
    mask_frame = unpacked[:H*W*D].reshape(H, W, D).astype(bool)
    t_mask = time.time() - t_start
    
    print(f"⚡ OCC Single Frame  : {t_occ*1000:.2f} ms")
    print(f"⚡ Mask Single Frame : {t_mask*1000:.2f} ms")
    
    # 验证一下
    print(f"   (Verify Shapes: OCC {occ_frame.shape}, Mask {mask_frame.shape})")
    del occ_frame, mask_frame


    # ================= 3. Clip 读取 (模拟 5-frame Video Clip) =================
    clip_len = 5
    start_idx = 0
    end_idx = start_idx + clip_len
    print(f"\n--- [Step 3: Clip Access ({clip_len} Frames)] ---")
    
    # --- 测试 OCC Clip ---
    t_start = time.time()
    # CSR 支持切片索引 [0:5]
    occ_clip_sparse = occ_matrix[start_idx:end_idx]
    # 注意：toarray() 此时会生成 (5, Flat_Dim)，然后再 reshape
    occ_clip = occ_clip_sparse.toarray().reshape(clip_len, *GRID_SIZE)
    t_occ_clip = time.time() - t_start
    
    # --- 测试 Mask Clip ---
    t_start = time.time()
    # 1. 取出 5 行压缩数据
    packed_rows = mask_packed[start_idx:end_idx]
    # 2. 批量解压 (numpy 会自动广播)
    unpacked_clip = np.unpackbits(packed_rows, axis=-1)
    # 3. Reshape
    mask_clip = unpacked_clip.reshape(clip_len, -1)[:, :H*W*D].reshape(clip_len, H, W, D).astype(bool)
    t_mask_clip = time.time() - t_start
    
    print(f"🎞️ OCC Clip ({clip_len} frames)  : {t_occ_clip*1000:.2f} ms")
    print(f"🎞️ Mask Clip ({clip_len} frames) : {t_mask_clip*1000:.2f} ms")
    print(f"   (Avg per frame: {t_mask_clip/clip_len*1000:.2f} ms)")
    
    # 验证一下
    print(f"   (Verify Shapes: OCC {occ_clip.shape}, Mask {mask_clip.shape})")
    del occ_clip, mask_clip

if __name__ == "__main__":
    benchmark_slicing()