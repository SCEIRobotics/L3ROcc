"""
实验：对于 Pi3X 的三维重建，哪个“绝对尺度(scale)”更准确 ——
是模型自己预测的尺度(metric 头)，还是用相机真实深度图推算出来的尺度？

真值(GT)采用机器人里程计轨迹(LeRobot parquet 中的 observation.state)，
通过手眼标定(hand-eye)换算到相机坐标系下的相机中心轨迹。该真值与两个待比较的
尺度来源(模型 metric 头、深度传感器)都相互独立，因此可以无循环依赖地公正裁定。

流程：先用 Pi3X 仅 RGB 跑一次，得到一个受控的“基准重建”，再用两种方式给它赋绝对尺度：
    * "model" ：信任 res["metric"]                  -> 修正系数 c = 1.0
    * "depth" ：s_depth = median(D_sensor/D_pred)    -> 修正系数 c = s_depth
第三个变体 "model_dc" 则是把深度图作为条件输入再跑一次 Pi3X。

每个变体都用“相机轨迹长度 vs 里程计真值长度”的相对误差来打分，
并输出逐帧诊断曲线和若干可视化图。

运行(在项目根目录、L3ROcc conda 环境内)：
    python tools/exp_scale_compare.py --rosbag_dir G:/vln_real_data/lerobot_data/20260601/rosbag_20260529_155555 --episode episode_001
"""

import os
import sys
import json
import argparse
import faulthandler
from contextlib import nullcontext as _nullcontext

faulthandler.enable()

# fp32 的 Pi3X(13.6 亿参数)约占 5.4GB；在 6GB 显卡上跑(非 Flash)注意力会显存溢出(OOM)。
# 允许强制使用 CPU。这一步必须在导入 torch 之前完成，因此提前扫描 argv。
if "--cpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 空字符串 "" 含义不明确；"-1" 才能可靠地屏蔽所有 GPU
    # CPU 推理是多线程的，放开线程数让它用上多核。
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) // 2)))
else:
    # GPU 路径下限制 BLAS 线程数(与流水线其余部分保持一致)。
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

# 无论从哪个目录启动，都让本地包可被 import。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import pandas as pd

# -----------------------------------------------------------------------------------------
# Windows 下 safetensors 的规避方案。
# 在本环境(huggingface_hub 1.16.x + safetensors 0.7 + torch 2.5.1 cu121)下，调用
# Pi3X.from_pretrained() 会段错误：它会先构造模型，而在 safetensors.load_file 之前导入
# Pi3X 模块(dinov2 / 注意力的 CUDA 初始化)会破坏 mmap 加载(在 torch/storage.py 中报
# "access violation")。若在导入模型模块“之前”先把权重加载到 CPU，则可正常加载。
# 因此这里先预加载权重，再 monkeypatch from_pretrained，让它构造模型并载入缓存的 state dict
# (之后把模型 .to(cuda) 不会有问题)。
# -----------------------------------------------------------------------------------------
from safetensors.torch import load_file as _st_load_file

_PI3X_CKPT_DIR = os.path.join(_PROJECT_ROOT, "ckpt", "pi3x")
print(f"[load] pre-loading Pi3X weights to CPU from {_PI3X_CKPT_DIR} ...")
_PI3X_STATE_DICT = _st_load_file(os.path.join(_PI3X_CKPT_DIR, "model.safetensors"), device="cpu")

# 权重已在内存中，此时再导入模型模块就安全了。
from third_party.pi3.pi3.models.pi3x import Pi3X


def _safe_from_pretrained(cls, *args, **kwargs):
    """替换原 from_pretrained 的等价实现，规避 Windows 下的段错误。"""
    model = Pi3X(use_multimodal=True)
    missing, unexpected = model.load_state_dict(_PI3X_STATE_DICT, strict=False)
    if missing or unexpected:
        print(f"[load] state_dict missing={len(missing)} unexpected={len(unexpected)}")
    return model  # 调用方随后会执行 .to(device).eval()


Pi3X.from_pretrained = classmethod(_safe_from_pretrained)

from L3ROcc.generater.normal_data_vln_env import SimpleVideoDataGenerator
from L3ROcc.utils import load_images_as_tensor


# --------------------------------------------------------------------------------------
# 几何小工具
# --------------------------------------------------------------------------------------
def quat_wxyz_to_R(q):
    """四元数 (w, x, y, z) -> 3x3 旋转矩阵。对前置维度做向量化处理。"""
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def path_length(points):
    """沿 (N, 3) 折线，对相邻点欧氏距离求和(即轨迹总长)。"""
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def umeyama_scale(source, target):
    """
    闭式 Umeyama/Sim3 “尺度”，把 source 映射到 target(target ≈ s*R*source + t)。
    这里只需要标量尺度 s，它等于两组点云去中心后 RMS 范围之比 —— 不做 SVD
    (从而避免 Windows 上 numpy-LAPACK/MKL 与 torch 冲突导致的崩溃)。
    """
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    ok = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
    src, dst = src[ok], dst[ok]
    if len(src) < 2:
        return float("nan")
    src_c = src - src.mean(0)
    dst_c = dst - dst.mean(0)
    denom = float((src_c ** 2).sum())
    if denom < 1e-12:
        return float("nan")
    return float(np.sqrt((dst_c ** 2).sum() / denom))


# --------------------------------------------------------------------------------------
# 由里程计 + 手眼标定得到真值相机轨迹
# --------------------------------------------------------------------------------------
def load_gt_camera_positions(parquet_path, info_json_path, interval, n_keep):
    """
    为(下采样后的)各帧构建真值相机中心轨迹。

    observation.state 共 14 维：x,y,z, vx,vy,vz, q_w,q_x,q_y,q_z, roll,pitch,yaw,yaw_speed
    相机刚性安装在夹爪上：X_gripper = R_c2g @ X_cam + t_c2g，
    因此相机中心在世界系下 = R_world_gripper @ t_c2g + p_gripper。

    返回：
        cam_pos   : (n_keep, 3) 下采样各帧的真值相机中心
        grip_pos  : (n_keep, 3) 真值夹爪中心(忽略杆臂，用于对照检查)
    """
    df = pd.read_parquet(parquet_path)
    state = np.stack(df["observation.state"].values).astype(np.float64)  # (T, 14)
    grip_pos_full = state[:, 0:3]
    quat_full = state[:, 6:10]  # (w, x, y, z)

    # 手眼标定：从 info.json 取 t_cam2gripper(相机原点在夹爪坐标系中的位置)。
    with open(info_json_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    ext = info.get("head_camera_extrinsic", {})
    t_c2g = np.asarray(ext.get("t_cam2gripper", [[0.0], [0.0], [0.0]]), dtype=np.float64).reshape(3)

    R_wg = quat_wxyz_to_R(quat_full)                       # (T, 3, 3)
    cam_pos_full = np.einsum("tij,j->ti", R_wg, t_c2g) + grip_pos_full  # (T, 3)

    # 用与 RGB 帧相同的 interval 做下采样，再截断到 n_keep。
    cam_pos = cam_pos_full[0::interval][:n_keep]
    grip_pos = grip_pos_full[0::interval][:n_keep]
    return cam_pos, grip_pos


# --------------------------------------------------------------------------------------
# Pi3X 推理
# --------------------------------------------------------------------------------------
@torch.no_grad()
def run_pi3x(gen, imgs, conditions=None):
    """运行 Pi3X(conditions 为 None 时即仅 RGB)，并把关键张量取回 CPU。"""
    use_cuda = gen.device == "cuda"
    # CPU 上以 fp32 原生运行(不用 autocast)；GPU 上用配置好的半精度 dtype。
    ctx = torch.amp.autocast("cuda", dtype=gen.amp_dtype) if use_cuda else _nullcontext()
    with ctx:
        if conditions is None:
            res = gen.model(imgs[None])
        else:
            res = gen.model(imgs[None], **conditions)

    out = {
        "cam_pos": res["camera_poses"][0][:, :3, 3].float().cpu().numpy(),   # (N, 3) 公制(米)
        "pred_depth": res["local_points"][0][..., 2].float().cpu().numpy(),  # (N, H, W) 公制深度 Z
        "conf": torch.sigmoid(res["conf"][0][..., 0]).float().cpu().numpy(), # (N, H, W) 置信度
        "metric": float(res["metric"].reshape(-1)[0].float().cpu()),
    }
    return out


# --------------------------------------------------------------------------------------
# 由传感器深度估计尺度
# --------------------------------------------------------------------------------------
def depth_scale_ratios(pred_depth, sensor_depth, conf, conf_thr, dmin, dmax):
    """
    逐帧鲁棒尺度 s_i = median(D_sensor / D_pred)(只在有效像素上统计)，
    以及把所有帧所有有效像素汇总后的全局尺度。

    返回：per_frame (N,)、global_scale (float)、valid_counts (N,)
    """
    N = pred_depth.shape[0]
    per_frame = np.full(N, np.nan, dtype=np.float64)
    counts = np.zeros(N, dtype=np.int64)
    all_ratios = []
    for i in range(N):
        ps = pred_depth[i]
        ds = sensor_depth[i]
        # 有效像素：预测/传感器深度均有限、传感器深度落在可信区间、预测深度为正、置信度足够。
        m = (
            np.isfinite(ps) & np.isfinite(ds)
            & (ds > dmin) & (ds < dmax)
            & (ps > 1e-3)
            & (conf[i] > conf_thr)
        )
        counts[i] = int(m.sum())
        if counts[i] >= 50:
            r = ds[m] / ps[m]
            per_frame[i] = float(np.median(r))
            all_ratios.append(r)
    global_scale = float(np.median(np.concatenate(all_ratios))) if all_ratios else float("nan")
    return per_frame, global_scale, counts


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="对比 Pi3X 的“模型预测尺度”与“深度推算尺度”。")
    ap.add_argument("--rosbag_dir", type=str,
                    default=r"G:/vln_real_data/lerobot_data/20260601/rosbag_20260529_155555")
    ap.add_argument("--episode", type=str, default="episode_001")
    ap.add_argument("--out_dir", type=str, default=os.path.join(_PROJECT_ROOT, "tools", "exp_out"))
    ap.add_argument("--conf_thr", type=float, default=0.1, help="像素有效所需的 Pi3X 最小置信度")
    ap.add_argument("--dmin", type=float, default=0.25, help="可信传感器深度下限(米)")
    ap.add_argument("--dmax", type=float, default=6.0, help="可信传感器深度上限(米)")
    ap.add_argument("--cpu", action="store_true",
                    help="强制 CPU 推理(显存 < ~8GB 时需要；在导入阶段即生效)。")
    ap.add_argument("--pixel_limit", type=int, default=255000,
                    help="送入 Pi3X 的每帧最大像素数。CPU 运行时调小(如 120000)可加速。")
    args = ap.parse_args()

    rb = args.rosbag_dir
    rgb_path = os.path.join(rb, "videos", "chunk-000", "observation.images.RGB", f"{args.episode}.mp4")
    depth_path = os.path.join(rb, "videos", "chunk-000", "observation.images.depth", f"{args.episode}.mkv")
    if not os.path.exists(depth_path):
        depth_path = os.path.join(rb, "videos", "chunk-000", "observation.images.depth", f"{args.episode}.mp4")
    parquet_path = os.path.join(rb, "data", "chunk-000", f"{args.episode}.parquet")
    info_json_path = os.path.join(rb, "meta", "info.json")

    for p in [rgb_path, depth_path, parquet_path, info_json_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    out_dir = os.path.join(args.out_dir, args.episode)
    os.makedirs(out_dir, exist_ok=True)

    # 深度条件所需的相机内参。
    with open(info_json_path, "r", encoding="utf-8") as f:
        intr_np = np.array(json.load(f)["head_camera_intrinsic"], dtype=np.float32)

    # 加载模型 + 配置(复用流水线的模型加载 / amp dtype 处理逻辑)。
    config_path = os.path.join(_PROJECT_ROOT, "L3ROcc", "configs", "config.yaml")
    model_dir = os.path.join(_PROJECT_ROOT, "ckpt")
    gen = SimpleVideoDataGenerator(config_path, out_dir, model_dir, use_multimodal=True)
    interval = gen.interval
    print(f"[cfg] interval={interval}  device={gen.device}  amp={gen.amp_dtype}")

    # 构建输入(下采样后的 RGB + 公制传感器深度 + 缩放后的内参)。
    imgs, traj_len, conditions = load_images_as_tensor(
        rgb_path, interval=interval, PIXEL_LIMIT=args.pixel_limit,
        condit_depth_path=depth_path, intrinsics_np=intr_np, device=gen.device,
    )
    imgs = imgs.to(gen.device)
    N = imgs.shape[0]
    if conditions.get("depths") is None:
        raise RuntimeError("传感器深度加载失败(需要公制的 gray16le 深度)。")
    sensor_d = conditions["depths"][0].float().cpu().numpy()  # (N, H, W) 单位：米
    print(f"[data] kept {N} frames @ {imgs.shape[-2]}x{imgs.shape[-1]} ; sensor depth {sensor_d.shape}")

    # 两次模型推理。
    print("[run] Pi3X RGB-only ...")
    rgb = run_pi3x(gen, imgs, conditions=None)
    print("[run] Pi3X depth-conditioned ...")
    dc = run_pi3x(gen, imgs, conditions=conditions)

    # 真值相机轨迹(下采样以与重建帧对齐)。
    gt_cam, gt_grip = load_gt_camera_positions(parquet_path, info_json_path, interval, N)
    n = min(N, len(gt_cam))
    gt_cam = gt_cam[:n]
    rgb_pos = rgb["cam_pos"][:n]
    dc_pos = dc["cam_pos"][:n]

    L_gt = path_length(gt_cam)
    L_model = path_length(rgb_pos)
    L_model_dc = path_length(dc_pos)

    # 理想修正系数(Umeyama 尺度，source->target 即 重建->真值)。
    c_gt_rgb = umeyama_scale(rgb_pos, gt_cam)
    if not np.isfinite(c_gt_rgb):
        c_gt_rgb = L_gt / max(L_model, 1e-9)

    # 深度推算尺度 + 逐帧比值。
    s_pf_rgb, s_depth, _ = depth_scale_ratios(rgb["pred_depth"][:n], sensor_d[:n], rgb["conf"][:n],
                                              args.conf_thr, args.dmin, args.dmax)
    s_pf_dc, s_depth_dc, _ = depth_scale_ratios(dc["pred_depth"][:n], sensor_d[:n], dc["conf"][:n],
                                                args.conf_thr, args.dmin, args.dmax)

    # 相对真值的轨迹长度误差(独立的裁判量)。
    e_model = abs(L_model - L_gt) / L_gt
    e_depth = abs(s_depth * L_model - L_gt) / L_gt
    e_model_dc = abs(L_model_dc - L_gt) / L_gt

    # 累计长度曲线。
    def cum(points, scale=1.0):
        d = np.linalg.norm(np.diff(points, axis=0), axis=1) * scale
        return np.concatenate([[0.0], np.cumsum(d)])

    gt_grip = gt_grip[:n]
    M = {
        "episode": args.episode,
        "n_frames": int(n),
        "metric_rgb": rgb["metric"],
        "metric_dc": dc["metric"],
        "s_depth": s_depth,
        "s_depth_dc": s_depth_dc,
        "c_gt_rgb": float(c_gt_rgb),
        "L_gt": L_gt,
        "L_gt_gripper": path_length(gt_grip),
        "L_model": L_model,
        "L_model_dc": L_model_dc,
        "L_depth": s_depth * L_model,
        "e_model": e_model,
        "e_depth": e_depth,
        "e_model_dc": e_model_dc,
        # 尺度空间误差(修正系数 vs 理想系数)
        "scale_err_model": abs(1.0 - c_gt_rgb) / c_gt_rgb,
        "scale_err_depth": abs(s_depth - c_gt_rgb) / c_gt_rgb,
        "s_per_frame_rgb": s_pf_rgb.tolist(),
        "s_per_frame_dc": s_pf_dc.tolist(),
        "cum_gt": cum(gt_cam).tolist(),
        "cum_model": cum(rgb_pos).tolist(),
        "cum_depth": cum(rgb_pos, s_depth).tolist(),
        "cum_dc": cum(dc_pos).tolist(),
    }

    # ---- 打印结果 ----
    print("\n================ RESULTS ================")
    print(f"GT camera path length (odometry+handeye): {L_gt:.4f} m   (gripper-only {M['L_gt_gripper']:.4f} m)")
    print(f"Pi3X metric head scalar : RGB={rgb['metric']:.4f}  depth-cond={dc['metric']:.4f}")
    print(f"Ideal scale c_gt (Umeyama recon->GT)     : {c_gt_rgb:.4f}")
    print(f"Depth-derived scale s_depth              : {s_depth:.4f}")
    print("-----------------------------------------")
    print(f"[model       ] L={L_model:.4f} m  ->  traj-len err = {e_model*100:5.2f}%   scale err = {M['scale_err_model']*100:5.2f}%")
    print(f"[depth-scaled] L={s_depth*L_model:.4f} m  ->  traj-len err = {e_depth*100:5.2f}%   scale err = {M['scale_err_depth']*100:5.2f}%")
    print(f"[model_dc    ] L={L_model_dc:.4f} m  ->  traj-len err = {e_model_dc*100:5.2f}%")
    winner = min([("model", e_model), ("depth", e_depth), ("model_dc", e_model_dc)], key=lambda x: x[1])
    print(f">>> Most accurate scale on this episode: '{winner[0]}'  ({winner[1]*100:.2f}% length error)")
    print("=========================================\n")

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(M, f, indent=2)
    print(f"[out] metrics -> {os.path.join(out_dir, 'metrics.json')}")

    # 保存绘图所需的全部数据，然后在一个“从不导入 torch”的独立子进程中渲染 ——
    # 在本 Windows 环境下，一旦 torch 的 MKL 驻留在同一进程，matplotlib 的 numpy/LAPACK
    # 调用就会崩溃(0xc06d007f)。
    npz_path = os.path.join(out_dir, "plotdata.npz")
    np.savez_compressed(
        npz_path,
        frame_idx=np.arange(n) * interval,
        pred_d_rgb=rgb["pred_depth"][:n].astype(np.float32),
        pred_d_dc=dc["pred_depth"][:n].astype(np.float32),
        sensor_d=sensor_d[:n].astype(np.float32),
        conf_rgb=rgb["conf"][:n].astype(np.float32),
        dmin=args.dmin, dmax=args.dmax, conf_thr=args.conf_thr,
    )
    print(f"[out] plot data -> {npz_path}")

    import subprocess
    plot_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_plot.py")
    print("[viz] rendering figures in a clean subprocess ...")
    plot_env = dict(os.environ)
    plot_env["MKL_THREADING_LAYER"] = "SEQUENTIAL"
    plot_env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    plot_env["OMP_NUM_THREADS"] = "1"
    r = subprocess.run([sys.executable, plot_script, out_dir], capture_output=True, text=True, env=plot_env)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        print(f"[viz] plotting subprocess failed (rc={r.returncode}); data is in {npz_path}")


if __name__ == "__main__":
    main()
