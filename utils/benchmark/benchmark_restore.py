import time
import os
import numpy as np
import scipy.sparse as sparse
import gc

# ================= 配置路径 =================
OCC_PATH = "/mnt/data/huangbinling/project/occgen/small_vln_n1_split_e3/traj_data/3d/111/trajectory_15/videos/chunk-000/observation.occ.view/occ_sequence_csr_final.npz"
MASK_PATH = "/mnt/data/huangbinling/project/occgen/small_vln_n1_split_e3/traj_data/3d/111/trajectory_15/videos/chunk-000/observation.occ.mask/mask_sequence_csr_final.npz"

# 目标形状 (用于验证)
TARGET_SHAPE = (178, 400, 400, 400)
# ===========================================

def format_size(bytes_size):
    return f"{bytes_size / (1024**3):.2f} GB"

def benchmark_restore():
    print(f"🚀 Starting Benchmark: Restore to N*H*W*C {TARGET_SHAPE}...\n")

    # =================  测试 OCC 还原 =================
    print(f"--- [Testing OCC Restoration] ---")
    if os.path.exists(OCC_PATH):
        try:
            # A. IO读取阶段
            t_start = time.time()
            occ_sparse = sparse.load_npz(OCC_PATH)
            t_io = time.time() - t_start
            print(f"1. Disk Load (CSR): {t_io:.4f}s")

            # B. 转换阶段 (CSR -> Dense N*H*W*C)
            t_conv_start = time.time()
            
            # .toarray() 会把稀疏矩阵炸开成稠密矩阵
            # .reshape() 把它变成 4D
            occ_dense = occ_sparse.toarray().reshape(TARGET_SHAPE)
            
            t_conv = time.time() - t_conv_start
            
            print(f"2. Dense Restore  : {t_conv:.4f}s (CPU Heavy)")
            print(f"   -> Total Time  : {t_io + t_conv:.4f}s")
            print(f"   -> Memory Used : {format_size(occ_dense.nbytes)}")
            print(f"   -> Verify Shape: {occ_dense.shape}")
            
            # 立即释放内存！
            del occ_dense
            del occ_sparse
            gc.collect()
            print("OCC Memory Released.\n")
            
        except Exception as e:
            print(f"OCC Failed: {e}\n")
    else:
        print("OCC file not found.\n")

    # ================= 2. 测试 Mask 还原 =================
    print(f"--- [Testing Mask Restoration] ---")
    if os.path.exists(MASK_PATH):
        try:
            # A. IO读取阶段
            t_start = time.time()
            loader = np.load(MASK_PATH)
            packed_data = loader['data']
            stored_shape = tuple(loader['shape']) # 或者是 TARGET_SHAPE
            t_io = time.time() - t_start
            print(f"1. Disk Load (Pack): {t_io:.4f}s")

            # B. 转换阶段 (Packbits -> Dense N*H*W*C)
            t_conv_start = time.time()
            
            # 1. 位解压
            unpacked = np.unpackbits(packed_data, axis=-1)
            
            # 2. 截断与重塑 (这是内存消耗最大的瞬间)
            N, H, W, D = TARGET_SHAPE
            flat_len = H * W * D
            
            # 为了省内存，我们可以分步 reshape，虽然 Python 内部还是会申请新内存
            mask_dense = unpacked.reshape(N, -1)[:, :flat_len].reshape(N, H, W, D)
            
            # 转换为 bool 以确保它是 1 byte/voxel (虽然 unpackbits 默认就是 uint8 0/1)
            mask_dense = mask_dense.astype(bool)

            t_conv = time.time() - t_conv_start
            
            print(f"2. Dense Restore  : {t_conv:.4f}s (CPU Heavy)")
            print(f"   -> Total Time  : {t_io + t_conv:.4f}s")
            print(f"   -> Memory Used : {format_size(mask_dense.nbytes)}")
            print(f"   -> Verify Shape: {mask_dense.shape}")

            # 立即释放内存！
            del mask_dense
            del packed_data
            del unpacked
            gc.collect()
            print("Mask Memory Released.\n")
            
        except Exception as e:
            print(f"Mask Failed: {e}\n")
    else:
        print("Mask file not found.\n")

if __name__ == "__main__":
    benchmark_restore()