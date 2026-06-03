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

# 无论脚本放在 tools/ 下哪一层、从哪个目录启动，都向上搜索定位项目根目录
# (含 L3ROcc/ 与 third_party/ 的那一层)，再让本地包可被 import。
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
while _PROJECT_ROOT != os.path.dirname(_PROJECT_ROOT):
    if os.path.isdir(os.path.join(_PROJECT_ROOT, "L3ROcc")) and \
       os.path.isdir(os.path.join(_PROJECT_ROOT, "third_party")):
        break
    _PROJECT_ROOT = os.path.dirname(_PROJECT_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import pandas as pd

# Windows safetensors 规避（仅 Windows 需要；Linux/服务器走原生 from_pretrained）。
# 经验规则：必须在导入 generator（进而 open3d / Pi3X 模块）之前先把权重加载到 CPU，
# 否则之后构造 Pi3X 时会在卷积层初始化处段错误(access violation)。这里先 load_file，
# 再 monkeypatch Pi3X.from_pretrained 用预加载权重构造模型；base.py 惰性调用 from_pretrained
# 会命中此 monkeypatch。Linux 上跳过本段，base.py 直接原生加载。
if sys.platform == "win32":
    from safetensors.torch import load_file as _st_load_file
    _PI3X_SD = _st_load_file(
        os.path.join(_PROJECT_ROOT, "ckpt", "pi3x", "model.safetensors"), device="cpu"
    )
    from third_party.pi3.pi3.models.pi3x import Pi3X as _Pi3X

    def _safe_from_pretrained(cls, *args, **kwargs):
        model = _Pi3X(use_multimodal=True)
        missing, unexpected = model.load_state_dict(_PI3X_SD, strict=False)
        if missing or unexpected:
            print(f"[load] state_dict missing={len(missing)} unexpected={len(unexpected)}")
        return model

    _Pi3X.from_pretrained = classmethod(_safe_from_pretrained)

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
# 跨平台绘图：Linux 进程内直接调用；Windows 用独立子进程(避开 torch+MKL 同进程崩溃)
# --------------------------------------------------------------------------------------
def render(out_dir, mode):
    """mode ∈ {"episode", "summary"}。out_dir 为对应数据(plotdata.npz / summary.json)所在目录。"""
    if sys.platform == "win32":
        import subprocess
        plot_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_plot.py")
        env = dict(os.environ)
        env["MKL_THREADING_LAYER"] = "SEQUENTIAL"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["OMP_NUM_THREADS"] = "1"
        r = subprocess.run([sys.executable, plot_script, out_dir, mode],
                           capture_output=True, text=True, env=env)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            sys.stderr.write(r.stderr)
            print(f"[viz] plotting subprocess failed (rc={r.returncode})")
    else:
        _sd = os.path.dirname(os.path.abspath(__file__))
        if _sd not in sys.path:
            sys.path.insert(0, _sd)
        import exp_plot
        (exp_plot.plot_summary if mode == "summary" else exp_plot.plot_episode)(out_dir)


# --------------------------------------------------------------------------------------
# episode / rosbag 发现
# --------------------------------------------------------------------------------------
def discover_episodes(rb):
    """返回某个 rosbag 目录下所有 episode 名(按 RGB 视频名)。"""
    d = os.path.join(rb, "videos", "chunk-000", "observation.images.RGB")
    if not os.path.isdir(d):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.lower().endswith(".mp4"))


def discover_rosbags(root):
    """返回 input_root 下所有 rosbag_* 子目录。"""
    if not os.path.isdir(root):
        return []
    return sorted(os.path.join(root, d) for d in os.listdir(root)
                  if d.startswith("rosbag_") and os.path.isdir(os.path.join(root, d)))


# --------------------------------------------------------------------------------------
# 单 episode 处理(复用已加载的 gen，模型只加载一次)
# --------------------------------------------------------------------------------------
def process_episode(gen, rb, episode, out_dir, label, args, do_plots):
    """处理一个 episode：两次推理 + 计算指标 + 存盘(+可选出图)。返回指标 dict M 或 None。"""
    rgb_path = os.path.join(rb, "videos", "chunk-000", "observation.images.RGB", f"{episode}.mp4")
    depth_path = os.path.join(rb, "videos", "chunk-000", "observation.images.depth", f"{episode}.mkv")
    if not os.path.exists(depth_path):
        depth_path = os.path.join(rb, "videos", "chunk-000", "observation.images.depth", f"{episode}.mp4")
    parquet_path = os.path.join(rb, "data", "chunk-000", f"{episode}.parquet")
    info_json_path = os.path.join(rb, "meta", "info.json")

    for p in [rgb_path, depth_path, parquet_path, info_json_path]:
        if not os.path.exists(p):
            print(f"[skip] {label}: 缺少文件 {p}")
            return None

    os.makedirs(out_dir, exist_ok=True)

    with open(info_json_path, "r", encoding="utf-8") as f:
        intr_np = np.array(json.load(f)["head_camera_intrinsic"], dtype=np.float32)

    interval = gen.interval
    imgs, traj_len, conditions = load_images_as_tensor(
        rgb_path, interval=interval, PIXEL_LIMIT=args.pixel_limit,
        condit_depth_path=depth_path, intrinsics_np=intr_np, device=gen.device,
    )
    imgs = imgs.to(gen.device)
    N = imgs.shape[0]
    if conditions.get("depths") is None:
        print(f"[skip] {label}: 传感器深度加载失败(需要公制 gray16le 深度)")
        return None
    sensor_d = conditions["depths"][0].float().cpu().numpy()  # (N, H, W) 米
    print(f"\n=== {label} === kept {N} frames @ {imgs.shape[-2]}x{imgs.shape[-1]}")

    print("[run] Pi3X RGB-only ...")
    rgb = run_pi3x(gen, imgs, conditions=None)
    print("[run] Pi3X depth-conditioned ...")
    dc = run_pi3x(gen, imgs, conditions=conditions)

    gt_cam, gt_grip = load_gt_camera_positions(parquet_path, info_json_path, interval, N)
    n = min(N, len(gt_cam))
    gt_cam, gt_grip = gt_cam[:n], gt_grip[:n]
    rgb_pos = rgb["cam_pos"][:n]
    dc_pos = dc["cam_pos"][:n]

    L_gt = path_length(gt_cam)
    L_model = path_length(rgb_pos)
    L_model_dc = path_length(dc_pos)

    c_gt_rgb = umeyama_scale(rgb_pos, gt_cam)
    if not np.isfinite(c_gt_rgb):
        c_gt_rgb = L_gt / max(L_model, 1e-9)

    s_pf_rgb, s_depth, _ = depth_scale_ratios(rgb["pred_depth"][:n], sensor_d[:n], rgb["conf"][:n],
                                              args.conf_thr, args.dmin, args.dmax)
    s_pf_dc, s_depth_dc, _ = depth_scale_ratios(dc["pred_depth"][:n], sensor_d[:n], dc["conf"][:n],
                                                args.conf_thr, args.dmin, args.dmax)

    L_gt_safe = max(L_gt, 1e-9)
    e_model = abs(L_model - L_gt) / L_gt_safe
    e_depth = abs(s_depth * L_model - L_gt) / L_gt_safe
    e_model_dc = abs(L_model_dc - L_gt) / L_gt_safe

    def cum(points, scale=1.0):
        d = np.linalg.norm(np.diff(points, axis=0), axis=1) * scale
        return np.concatenate([[0.0], np.cumsum(d)])

    # 近静止 episode：相机几乎没动，尺度不可观测 → 标记并从汇总剔除。
    skipped = bool(L_gt < args.min_motion)

    M = {
        "episode": label,
        "n_frames": int(n),
        "skipped": skipped,
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
        "scale_err_model": abs(1.0 - c_gt_rgb) / c_gt_rgb,
        "scale_err_depth": abs(s_depth - c_gt_rgb) / c_gt_rgb,
        "s_per_frame_rgb": s_pf_rgb.tolist(),
        "s_per_frame_dc": s_pf_dc.tolist(),
        "cum_gt": cum(gt_cam).tolist(),
        "cum_model": cum(rgb_pos).tolist(),
        "cum_depth": cum(rgb_pos, s_depth).tolist(),
        "cum_dc": cum(dc_pos).tolist(),
    }

    print(f"GT path {L_gt:.4f} m (gripper {M['L_gt_gripper']:.4f}) | metric RGB={rgb['metric']:.3f} dc={dc['metric']:.3f}"
          f" | c_gt={c_gt_rgb:.3f} s_depth={s_depth:.3f}")
    print(f"[model {e_model*100:5.2f}%] [depth {e_depth*100:5.2f}%] [model_dc {e_model_dc*100:5.2f}%]"
          + ("   (skipped: 近静止)" if skipped else ""))

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(M, f, indent=2)

    # 仅在需要逐集出图、且非近静止时，保存绘图数据并渲染该集的 6 张图。
    if do_plots and not skipped:
        np.savez_compressed(
            os.path.join(out_dir, "plotdata.npz"),
            frame_idx=np.arange(n) * interval,
            pred_d_rgb=rgb["pred_depth"][:n].astype(np.float32),
            pred_d_dc=dc["pred_depth"][:n].astype(np.float32),
            sensor_d=sensor_d[:n].astype(np.float32),
            conf_rgb=rgb["conf"][:n].astype(np.float32),
            dmin=args.dmin, dmax=args.dmax, conf_thr=args.conf_thr,
        )
        render(out_dir, "episode")

    return M


# --------------------------------------------------------------------------------------
# 跨 episode 汇总
# --------------------------------------------------------------------------------------
def write_summary(results, out_root):
    """把多 episode 的指标写成 summary.json / summary.csv，并返回 summary dict。"""
    import csv
    keys = ["episode", "n_frames", "L_gt", "L_model", "L_model_dc", "s_depth", "c_gt_rgb",
            "metric_rgb", "metric_dc", "e_model", "e_depth", "e_model_dc",
            "scale_err_model", "scale_err_depth"]
    rows = [{k: M.get(k) for k in keys} for M in results]

    def agg(key):
        vals = [r[key] for r in rows if r[key] is not None and np.isfinite(r[key])]
        if not vals:
            return None
        return {"mean": float(np.mean(vals)), "median": float(np.median(vals)), "std": float(np.std(vals))}

    win_counts = {"model": 0, "depth": 0, "model_dc": 0}
    for M in results:
        w = min(("model", "depth", "model_dc"), key=lambda k: M[f"e_{k}"])
        win_counts[w] += 1

    summary = {
        "n_episodes": len(rows),
        "win_counts": win_counts,
        "agg": {k: agg(k) for k in ["e_model", "e_depth", "e_model_dc",
                                     "s_depth", "c_gt_rgb", "scale_err_model", "scale_err_depth"]},
        "rows": rows,
    }
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_root, "summary.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return summary


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="对比 Pi3X 的“模型预测尺度”与“深度推算尺度”。")
    ap.add_argument("--rosbag_dir", type=str,
                    default=r"G:/vln_real_data/lerobot_data/20260601/rosbag_20260529_155555",
                    help="单个 rosbag 目录(未给 --input_root 时使用)。") # 该rosbag目录已经过lerobot_data_builder.py预处理
    ap.add_argument("--input_root", type=str, default="",
                    help="可选：包含多个 rosbag_* 的根目录，遍历其下所有 rosbag。")
    ap.add_argument("--episode", type=str, default="episode_001",
                    help="episode 名；用 'all' 处理该 rosbag 下全部 episode。")
    ap.add_argument("--out_dir", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_out"))
    ap.add_argument("--conf_thr", type=float, default=0.1, help="像素有效所需的 Pi3X 最小置信度")
    ap.add_argument("--dmin", type=float, default=0.25, help="可信传感器深度下限(米)")
    ap.add_argument("--dmax", type=float, default=6.0, help="可信传感器深度上限(米)")
    ap.add_argument("--min_motion", type=float, default=0.3,
                    help="真值相机轨迹长度低于此值(米)的 episode 视为近静止，跳过(不计入汇总)。")
    ap.add_argument("--per_episode_plots", action="store_true",
                    help="批量时也为每个 episode 出 6 张图(默认仅单集出图，批量只出汇总图)。")
    ap.add_argument("--cpu", action="store_true",
                    help="强制 CPU 推理(显存 < ~8GB 时需要；在导入阶段即生效)。")
    ap.add_argument("--pixel_limit", type=int, default=255000,
                    help="送入 Pi3X 的每帧最大像素数。CPU 运行时调小(如 120000)可加速。")
    args = ap.parse_args()

    # 构建任务列表：(rosbag 目录, episode, 输出子目录, 标签)
    jobs = []
    if args.input_root:
        rosbags = discover_rosbags(args.input_root)
        if not rosbags:
            print(f"[error] --input_root 下未找到 rosbag_* 目录: {args.input_root}")
            return
        for rb in rosbags:
            eps = discover_episodes(rb) if args.episode == "all" else [args.episode]
            for ep in eps:
                name = os.path.basename(rb.rstrip("/\\"))
                jobs.append((rb, ep, os.path.join(args.out_dir, name, ep), f"{name}__{ep}"))
    else:
        rb = args.rosbag_dir
        eps = discover_episodes(rb) if args.episode == "all" else [args.episode]
        for ep in eps:
            jobs.append((rb, ep, os.path.join(args.out_dir, ep), ep))

    if not jobs:
        print("[error] 未发现任何 episode。检查 --rosbag_dir / --input_root / --episode。")
        return
    print(f"[plan] 共 {len(jobs)} 个 episode 待处理。")

    # 单集默认出逐集图；批量默认只出汇总图(除非 --per_episode_plots)。
    do_plots = args.per_episode_plots or (len(jobs) == 1)

    # 模型只加载一次。
    config_path = os.path.join(_PROJECT_ROOT, "L3ROcc", "configs", "config.yaml")
    model_dir = os.path.join(_PROJECT_ROOT, "ckpt")
    os.makedirs(args.out_dir, exist_ok=True)
    gen = SimpleVideoDataGenerator(config_path, args.out_dir, model_dir, use_multimodal=True)
    # Windows 预加载的 state dict 已被 load_state_dict 复制进模型，释放它(~5.4GB)给推理腾内存。
    if sys.platform == "win32":
        globals().pop("_PI3X_SD", None)
        import gc
        gc.collect()
    print(f"[cfg] interval={gen.interval}  device={gen.device}  amp={gen.amp_dtype}")

    results = []
    for rb, ep, odir, label in jobs:
        try:
            M = process_episode(gen, rb, ep, odir, label, args, do_plots)
        except Exception as e:
            import traceback
            print(f"[skip] {label}: {e}")
            traceback.print_exc()
            continue
        if M is not None:
            results.append(M)

    # 汇总(批量时)。
    if len(jobs) > 1:
        kept = [M for M in results if not M.get("skipped")]
        print(f"\n[summary] {len(kept)}/{len(results)} 个有效 episode 计入汇总 "
              f"(跳过近静止 {len(results) - len(kept)} 个)。")
        if kept:
            s = write_summary(kept, args.out_dir)
            for m in ("e_model", "e_depth", "e_model_dc"):
                a = s["agg"][m]
                if a:
                    print(f"  {m:12s} mean={a['mean']*100:5.2f}%  median={a['median']*100:5.2f}%")
            print(f"  win counts: {s['win_counts']}")
            render(args.out_dir, "summary")
        else:
            print("[summary] 无有效 episode，跳过汇总。")


if __name__ == "__main__":
    main()
