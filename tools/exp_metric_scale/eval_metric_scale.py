"""
实验：Pi3X 的 metric_head 输出的绝对尺度（≈0.3）在真实世界里到底准不准、能不能直接用？

背景
----
Pi3X 在原始 Pi3（尺度不变）之上加了一个 metric_head（third_party/pi3/pi3/models/pi3x.py
第 399 行），输出一个**全局标量** metric（log 空间，.exp() 后得到），统一缩放 points /
camera_poses 平移 / local_points，把重建拉到“近似公制”。已有观察：跨数据集、无论内参/深度
是否参与输入，metric 都≈0.3。两种可能：(a) 这些场景真实尺度恰好相近；(b) metric_head 塌缩
成了常数。本脚本判定它在真实世界里准不准。

关键约束（与 tools/exp_scale/exp_scale_compare.py 的不同点）
----------------------------------------------------------
**绝不使用 GT odom 位姿对齐/纠正尺度**（即不走 intern_vln_env.align_with_gt_scale，也不读
parquet 的 observation.state / action 轨迹做 Sim3）。本脚本只用与 odom 无关的独立公制参照来
**测量**误差，绝不把测得尺度回灌去修正模型输出。

三条互补证据链
--------------
* 方法 A（主，定量）：逐像素深度比值 vs 度量传感器。模型只喂 RGB(+内参)，**不喂深度**；
  传感器 16-bit 公制深度（N1: 0.1mm/LSB PNG；Unitree: 1mm/LSB gray16le）仅作参照。
  r = pred_depth / sensor_depth，median(r)≈1.0 说明 metric 准，≈0.3 说明系统性偏小。
* 方法 B（辅，零 GT）：导出重建点云 PLY + 自动地面 RANSAC 距离统计，供在点云查看器里量取
  已知尺寸结构（层高/门高/桌面高/机器人自身尺寸）做交叉验证。
* 方法 C（诊断）：跨序列统计 metric 标量分布。若恒≈0.3 而方法 A 的真实 ratio 随场景大幅
  变化 → 证明 metric_head 已塌缩为常数、不可直接当公制用。

输入条件只测 仅RGB / RGB+内参 两种（喂深度作输入=作弊，会污染尺度验证）。
自检用 --cheat-depth 可临时把传感器深度喂进模型（ratio 应明显趋近 1，验证比较逻辑正确）。

用法
----
    python tools/exp_metric_scale/eval_metric_scale.py --input <数据集根> \
        [--episode all] [--cond rgb rgb_intr] [--intrinsics K.npy] \
        [--export-ply] [--out-dir tools/exp_metric_scale/exp_out] [--cpu]
"""

import os
import sys
import json
import argparse
import faulthandler
from contextlib import nullcontext as _nullcontext

faulthandler.enable()

# fp32 的 Pi3X(约 13.6 亿参数)较吃显存；允许强制 CPU。必须在导入 torch 之前完成，故先扫 argv。
if "--cpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) // 2)))
else:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

# 向上搜索定位项目根目录(含 L3ROcc/ 与 third_party/ 的那一层)，再让本地包可被 import。
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

# Windows safetensors 规避(仅 Windows 需要；Linux/服务器走原生 from_pretrained)。
# 必须在导入 base(进而 open3d / Pi3X 模块)之前把权重预加载到 CPU 并 monkeypatch from_pretrained，
# 否则之后构造 Pi3X 会在卷积层初始化处段错误。与 tools/exp_scale/exp_scale_compare.py 一致。
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

import cv2  # noqa: E402  (反光空洞边界腐蚀；cv2 已是 L3ROcc/utils.py 依赖)
import open3d as o3d  # noqa: E402

from L3ROcc.base import DataGenerator  # noqa: E402
from L3ROcc.dataset.intern_nav_adapter import InternNavSequenceLoader  # noqa: E402
from L3ROcc.utils import load_images_as_tensor  # noqa: E402
from third_party.pi3.pi3.utils.basic import write_ply  # noqa: E402
from third_party.pi3.pi3.utils.geometry import depth_edge  # noqa: E402


# =====================================================================================
# 内参读取(优先级 CLI npy > parquet observation.camera_intrinsic > info.json head_camera_intrinsic)
# 与 tools/exp_scale/exp_scale_compare.py 保持一致的解析逻辑。
# =====================================================================================
def _load_intrinsics_from_json(json_path):
    if not json_path or not os.path.isfile(json_path) or not json_path.endswith(".json"):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "head_camera_intrinsic" not in data:
            return None
        return np.array(data["head_camera_intrinsic"], dtype=np.float32)
    except Exception as e:
        print(f"[warn] 读取内参 JSON 失败 {json_path}: {e}")
        return None


def _load_intrinsics_from_parquet(parquet_path):
    if not parquet_path or not os.path.isfile(parquet_path):
        return None
    try:
        df = pd.read_parquet(parquet_path, columns=["observation.camera_intrinsic"])
    except Exception:
        return None
    if "observation.camera_intrinsic" not in df.columns or len(df) == 0:
        return None
    raw = df["observation.camera_intrinsic"].iloc[0]
    try:
        raw_list = raw.tolist() if hasattr(raw, "tolist") else raw
        arr = np.asarray(raw_list, dtype=np.float32)
    except Exception:
        return None
    if arr.shape == (3, 3):
        return arr
    if arr.size == 9:
        return arr.reshape(3, 3)
    return None


def resolve_intrinsics(cli_intr, parquet_path, info_json_path, label):
    """CLI npy > parquet > info.json > None(让 Pi3X 自行反算)。返回 (K_np_or_None, source_str)。"""
    if cli_intr is not None:
        return cli_intr, "cli"
    k = _load_intrinsics_from_parquet(parquet_path)
    if k is not None:
        return k, "parquet"
    k = _load_intrinsics_from_json(info_json_path)
    if k is not None:
        return k, "info_json"
    print(f"[{label}] 未找到内参(CLI/parquet/info.json 均无)，rgb_intr 将退化等同 rgb。")
    return None, "none_pi3x_back_calc"


# =====================================================================================
# Pi3X 推理(白名单过滤 conditions；可选作弊喂深度)
# =====================================================================================
_PI3X_KWARGS = {"poses", "depths", "intrinsics"}


@torch.no_grad()
def run_pi3x(gen, imgs, conditions=None):
    """运行 Pi3X(conditions=None 即仅 RGB)，取回逐像素深度 / 置信度 / 非边缘掩码 / metric / 世界点。"""
    use_cuda = gen.device == "cuda"
    ctx = torch.amp.autocast("cuda", dtype=gen.amp_dtype) if use_cuda else _nullcontext()
    with ctx:
        if conditions is None:
            res = gen.model(imgs[None])
        else:
            cond = {k: v for k, v in conditions.items() if k in _PI3X_KWARGS}
            res = gen.model(imgs[None], **cond)

    local_z = res["local_points"][0][..., 2]                 # (N, H, W) 相机系公制深度 Z(已被 metric 缩放)
    non_edge = (~depth_edge(local_z, rtol=0.03)).cpu().numpy()  # (N, H, W) bool
    return {
        "pred_depth": local_z.float().cpu().numpy(),                 # (N, H, W) 米
        "conf": torch.sigmoid(res["conf"][0][..., 0]).float().cpu().numpy(),  # (N, H, W)
        "non_edge": non_edge,                                        # (N, H, W) bool
        "metric": float(res["metric"].reshape(-1)[0].float().cpu()),
        "points": res["points"][0].float().cpu().numpy(),            # (N, H, W, 3) 世界系米(供方法 B)
    }


# =====================================================================================
# 方法 A：逐像素深度比值统计(传感器深度仅作参照，绝不喂入模型，也绝不碰 odom)
#
# 实采(lerobot/ZED)深度在反光/镜面处会出现空洞(=0)与"在量程内但物理错误"的异常值。
# 鲁棒过滤(默认开)针对性处理三类污染:
#   1) 空洞边界飞点  -> 对有效传感器掩码做 hole_erode px 二值腐蚀
#   2) 传感器深度突变 -> 对 sensor 深度做 depth_edge(带 mask 忽略空洞)
#   3) 镜面在量程内错值 -> 对 log(pred/sensor) 做 med±k·MAD 离群剔除(影响 mean 类指标)
# 同时输出未过滤的 *_raw 对照与反光严重度诊断(hole_frac/valid_frac/outlier_frac)。
# =====================================================================================
def _sensor_invalid_edges(sensor_d, valid_sensor, sensor_rtol):
    """对整段 (N,H,W) sensor 深度做 depth_edge(带 mask 忽略空洞)，返回 (N,H,W) bool 边缘掩码。"""
    st = torch.from_numpy(sensor_d.astype(np.float32))
    mt = torch.from_numpy(valid_sensor)
    with torch.no_grad():
        edge = depth_edge(st, rtol=sensor_rtol, mask=mt)
    return edge.cpu().numpy()


def _stats_from_ratio(pred, sensor):
    """给定配对 pred/sensor 一维数组，算 scale_ratio / correction / 校正后 absrel/δ1.25。"""
    r = pred / sensor
    q1, q3 = np.percentile(r, [25, 75])
    correction = float(np.median(sensor / pred))  # 乘到 pred 上以匹配传感器
    pred_c = pred * correction
    absrel = float(np.mean(np.abs(pred_c - sensor) / sensor))
    thresh = np.maximum(pred_c / sensor, sensor / pred_c)
    delta = float(np.mean(thresh < 1.25))
    return {
        "scale_ratio_median": float(np.median(r)),
        "scale_ratio_iqr": float(q3 - q1),
        "log_ratio_std": float(np.std(np.log(r))),
        "correction_factor": correction,
        "absrel": absrel,
        "delta1_25": delta,
    }


def depth_ratio_stats(pred_d, sensor_d, conf, non_edge, conf_thr, dmin, dmax,
                      robust=True, hole_erode=1, sensor_rtol=0.1, mad_k=3.0):
    """
    r = pred_depth / sensor_depth(只在有效像素上统计)。默认开启鲁棒过滤(见上)，
    并在 *_raw 字段保留未过滤结果以便对照。

    headline 字段(鲁棒, robust=False 时与 raw 相同):
        scale_ratio_median : median(r)  —— ≈1.0 准；<1 偏小、>1 偏大(核心数)
                             注意: 与 metric 标量(≈0.3)不是一回事! metric 只是作用在归一化
                             几何上的内部乘子, 其数值本身不代表准不准; 衡量真实尺度准确性的
                             是这个 pred/sensor 比值是否≈1。
        scale_ratio_iqr / log_ratio_std / correction_factor / absrel / delta1_25
        scale_ratio_median_of_frames : 逐帧 median 的 median(抗帧间像素数不均)
        valid_pixels / per_frame_ratio
    诊断字段:
        hole_frac    : ds≤dmin 占"有限像素"比例(反光/空洞严重度代理)
        valid_frac   : 进入统计的像素 / 总像素
        outlier_frac : MAD 离群剔除掉的比例(相对鲁棒前的有效像素)
        *_raw        : 未过滤(仅基础掩码)的同名指标
    """
    N = pred_d.shape[0]
    total_px = int(pred_d.size)
    per_frame = np.full(N, np.nan, dtype=np.float64)

    # --- 反光严重度: 在"有限像素"里统计空洞(≤dmin)占比 ---
    finite_sensor = np.isfinite(sensor_d)
    n_finite = int(finite_sensor.sum())
    hole_frac = float((finite_sensor & (sensor_d <= dmin)).sum() / max(n_finite, 1))

    # --- 传感器侧鲁棒掩码: 空洞边界腐蚀 + sensor 深度边缘 ---
    valid_sensor = finite_sensor & (sensor_d > dmin) & (sensor_d < dmax)  # (N,H,W)
    sensor_ok = valid_sensor.copy()
    if robust:
        if hole_erode > 0:
            k = np.ones((2 * hole_erode + 1, 2 * hole_erode + 1), np.uint8)
            for i in range(N):  # 逐帧腐蚀(cv2 要 2D uint8)
                sensor_ok[i] = cv2.erode(valid_sensor[i].astype(np.uint8), k,
                                         iterations=1).astype(bool)
        sensor_edge = _sensor_invalid_edges(sensor_d, valid_sensor, sensor_rtol)
        sensor_ok &= ~sensor_edge

    # --- 基础逐像素掩码(raw 与 robust 共享的非传感器部分) ---
    base = np.isfinite(pred_d) & (pred_d > 1e-3) & (conf > conf_thr) & non_edge
    raw_mask = base & valid_sensor   # 旧行为: 只剔空洞+量程+预测边缘
    rob_mask = base & sensor_ok      # 叠加传感器侧腐蚀/边缘

    def _collect(mask):
        all_p, all_s = [], []
        for i in range(N):
            mi = mask[i]
            if mi.sum() >= 50:
                all_p.append(pred_d[i][mi])
                all_s.append(sensor_d[i][mi])
        if not all_p:
            return None, None
        return np.concatenate(all_p), np.concatenate(all_s)

    pred_raw, sensor_raw = _collect(raw_mask)
    if pred_raw is None:
        nan = float("nan")
        return {
            "scale_ratio_median": nan, "scale_ratio_iqr": nan, "log_ratio_std": nan,
            "correction_factor": nan, "absrel": nan, "delta1_25": nan,
            "scale_ratio_median_of_frames": nan, "valid_pixels": 0,
            "hole_frac": hole_frac, "valid_frac": 0.0, "outlier_frac": nan,
            "per_frame_ratio": per_frame.tolist(),
            "scale_ratio_median_raw": nan, "absrel_raw": nan, "delta1_25_raw": nan,
            "valid_pixels_raw": 0, "robust": bool(robust),
        }

    raw_stats = _stats_from_ratio(pred_raw, sensor_raw)

    # robust 主集合
    if robust:
        pred_rob, sensor_rob = _collect(rob_mask)
        if pred_rob is None:  # 腐蚀/边缘过严，退回 raw
            pred_rob, sensor_rob = pred_raw, sensor_raw
    else:
        pred_rob, sensor_rob = pred_raw, sensor_raw
    n_before_mad = int(pred_rob.size)

    # --- MAD 离群剔除(对 log-ratio) ---
    outlier_frac = 0.0
    if robust and mad_k > 0 and n_before_mad >= 50:
        logr = np.log(pred_rob / sensor_rob)
        med = np.median(logr)
        mad = float(np.median(np.abs(logr - med)))
        if mad > 1e-9:
            keep = np.abs(logr - med) <= mad_k * 1.4826 * mad
            outlier_frac = float((~keep).sum() / n_before_mad)
            if keep.sum() >= 50:
                pred_rob, sensor_rob = pred_rob[keep], sensor_rob[keep]

    # --- 逐帧 median(用 robust 掩码) ---
    fmask = rob_mask if robust else raw_mask
    for i in range(N):
        mi = fmask[i]
        if mi.sum() >= 50:
            per_frame[i] = float(np.median(pred_d[i][mi] / sensor_d[i][mi]))

    out = _stats_from_ratio(pred_rob, sensor_rob)
    pf = per_frame[np.isfinite(per_frame)]
    out.update({
        "scale_ratio_median_of_frames": float(np.median(pf)) if pf.size else float("nan"),
        "valid_pixels": int(pred_rob.size),
        "hole_frac": hole_frac,
        "valid_frac": float(pred_rob.size / max(total_px, 1)),
        "outlier_frac": outlier_frac,
        "per_frame_ratio": per_frame.tolist(),
        # raw 对照
        "scale_ratio_median_raw": raw_stats["scale_ratio_median"],
        "absrel_raw": raw_stats["absrel"],
        "delta1_25_raw": raw_stats["delta1_25"],
        "valid_pixels_raw": int(pred_raw.size),
        "robust": bool(robust),
    })
    return out


# =====================================================================================
# 方法 B：点云导出 + 自动地面 RANSAC 距离统计(供人工量取已知尺寸结构)
# =====================================================================================
def ground_plane_stats(points, colors, conf, non_edge, conf_thr):
    """
    对重建点云做 Open3D RANSAC 地面拟合，报告平面以上点的高度跨度(粗略层高估计)。
    仅用于**测量**(与真实层高/门高比对)，不做任何对齐或尺度纠正。

    复用 intern_vln_env.align_with_gt_scale 中的 RANSAC 思路(distance_threshold=0.05m)。
    返回 (stats_dict, pcd_filtered_np, color_filtered_np)。
    """
    # 用与方法 A 相同的有效性筛选拉直点云
    pts = points.reshape(-1, 3)
    cols = colors.reshape(-1, 3)
    valid = (conf.reshape(-1) > conf_thr) & non_edge.reshape(-1) & np.isfinite(pts).all(1)
    pts, cols = pts[valid], cols[valid]
    stats = {"plane_model": None, "inlier_ratio": float("nan"),
             "height_span_p5_p95": float("nan"), "n_points": int(pts.shape[0])}
    if pts.shape[0] < 200:
        return stats, pts, cols

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    try:
        plane, inliers = pcd.segment_plane(distance_threshold=0.05,
                                           ransac_n=3, num_iterations=300)
    except Exception as e:
        print(f"[ground] RANSAC 失败: {e}")
        return stats, pts, cols
    a, b, c, d = plane
    n = np.array([a, b, c], dtype=np.float64)
    nrm = np.linalg.norm(n) + 1e-12
    signed = (pts @ n + d) / nrm  # 各点到平面的有符号距离(米)
    # 让“地面以上”为正方向
    if np.median(signed) < 0:
        signed = -signed
    p5, p95 = np.percentile(signed, [5, 95])
    stats.update({
        "plane_model": [float(x) for x in plane],
        "inlier_ratio": float(len(inliers) / pts.shape[0]),
        "height_span_p5_p95": float(p95 - p5),
    })
    return stats, pts, cols


# =====================================================================================
# 任务发现(复用 InternNavSequenceLoader：统一识别 单 unit / 多 unit 父目录 / N1 布局)
# =====================================================================================
def discover_jobs(dataset_root, target_episode):
    loader = InternNavSequenceLoader(dataset_root)
    jobs = []
    for i in range(len(loader)):
        rb = loader.trajectory_dirs[i]
        rgb_path = loader.trajectory_video_paths[i]
        depth_path = loader.trajectory_depth_paths[i]
        parquet_path = loader.trajectory_data_paths[i]
        ep = os.path.splitext(os.path.basename(rgb_path))[0]
        if target_episode != "all" and ep != target_episode:
            continue
        rb_name = os.path.basename(rb.rstrip("/\\"))
        if os.path.abspath(rb) == os.path.abspath(dataset_root):
            label = ep
        else:
            label = f"{rb_name}__{ep}"
        info_json = os.path.join(rb, "meta", "info.json")
        jobs.append(dict(rb=rb, ep=ep, label=label, rgb=rgb_path, depth=depth_path,
                         parquet=parquet_path, info_json=info_json))
    return jobs


# =====================================================================================
# 单 episode 处理
# =====================================================================================
def process_episode(gen, job, args, cli_intr):
    label = job["label"]
    out_dir = os.path.join(args.out_dir, label)

    if job["depth"] is None or not os.path.exists(job["depth"]):
        print(f"[skip] {label}: 无公制深度视频/目录，方法 A 无法进行")
        return None
    if not os.path.exists(job["rgb"]):
        print(f"[skip] {label}: 缺少 RGB {job['rgb']}")
        return None
    os.makedirs(out_dir, exist_ok=True)

    intr_np, intr_src = resolve_intrinsics(cli_intr, job["parquet"], job["info_json"], label)

    imgs, _, conditions = load_images_as_tensor(
        job["rgb"], interval=gen.interval, PIXEL_LIMIT=args.pixel_limit,
        condit_depth_path=job["depth"], intrinsics_np=intr_np, device=gen.device,
    )
    imgs = imgs.to(gen.device)
    if conditions.get("depths") is None:
        print(f"[skip] {label}: 传感器深度加载失败(depth_source="
              f"{conditions.get('depth_source')})，方法 A 需要公制深度")
        return None

    N = imgs.shape[0]
    sensor_d = conditions["depths"][0].float().cpu().numpy()  # (N, H, W) 米 —— 仅作参照
    img_colors = imgs.permute(0, 2, 3, 1).float().cpu().numpy()  # (N, H, W, 3)
    print(f"\n=== {label} === {N} 帧 @ {imgs.shape[-2]}x{imgs.shape[-1]} | "
          f"depth_source={conditions.get('depth_source')} | intr={intr_src} | "
          f"sensor_d median={np.median(sensor_d[sensor_d > 0]):.3f}m")

    # 构造各输入条件
    cond_int = {k: v for k, v in conditions.items() if k != "depths"}
    cond_int["depths"] = None
    runners = {"rgb": None, "rgb_intr": cond_int}
    if args.cheat_depth:  # 自检上界：把传感器深度喂进模型，ratio 应趋近 1
        runners["cheat_depth"] = conditions

    M = {"episode": label, "n_frames": int(N),
         "depth_source": conditions.get("depth_source", "unknown"),
         "intrinsic_source": intr_src,
         "intrinsic_used": intr_np.tolist() if intr_np is not None else None,
         "conds": {}}
    plot = {"sensor_d": sensor_d.astype(np.float32)}

    for cond_name in args.cond + (["cheat_depth"] if args.cheat_depth else []):
        if cond_name not in runners:
            continue
        print(f"[run] Pi3X {cond_name} ...")
        out = run_pi3x(gen, imgs, conditions=runners[cond_name])
        a = depth_ratio_stats(out["pred_depth"], sensor_d, out["conf"], out["non_edge"],
                              args.conf_thr, args.dmin, args.dmax,
                              robust=not args.no_robust, hole_erode=args.hole_erode,
                              sensor_rtol=args.sensor_rtol, mad_k=args.mad_k)
        a["metric_scalar"] = out["metric"]
        M["conds"][cond_name] = a
        plot[f"pred_d_{cond_name}"] = out["pred_depth"].astype(np.float32)
        plot[f"conf_{cond_name}"] = out["conf"].astype(np.float32)
        print(f"   metric={out['metric']:.4f} | scale_ratio(pred/sensor) median="
              f"{a['scale_ratio_median']:.4f} (raw {a['scale_ratio_median_raw']:.4f}) "
              f"iqr={a['scale_ratio_iqr']:.4f} | correction(x→sensor)={a['correction_factor']:.4f} | "
              f"AbsRel={a['absrel']:.4f} (raw {a['absrel_raw']:.4f}) δ1.25={a['delta1_25']:.4f} | "
              f"valid_px={a['valid_pixels']} | hole={a['hole_frac']:.3f} "
              f"valid_frac={a['valid_frac']:.3f} outlier={a['outlier_frac']:.3f}")

        # 方法 B：导出点云 + 地面统计(默认对第一个条件做，避免重复巨大 PLY)
        if args.export_ply and cond_name == args.cond[0]:
            g, pcd_f, col_f = ground_plane_stats(out["points"], img_colors,
                                                 out["conf"], out["non_edge"], args.conf_thr)
            M["ground_plane"] = g
            ply_path = os.path.join(out_dir, f"pcd_{cond_name}.ply")
            try:
                write_ply(pcd_f, col_f, ply_path)
                print(f"   [方法B] 点云已存 {ply_path} | 地面以上高度跨度(p5-p95)="
                      f"{g['height_span_p5_p95']:.3f}m inlier={g['inlier_ratio']:.2f}")
            except Exception as e:
                print(f"   [方法B] write_ply 失败: {e}")

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(M, f, indent=2, ensure_ascii=False)
    np.savez_compressed(os.path.join(out_dir, "plotdata.npz"),
                        dmin=args.dmin, dmax=args.dmax, conf_thr=args.conf_thr, **plot)
    return M


# =====================================================================================
# 汇总(方法 C：metric 标量分布 + 方法 A 汇总) + 出图
# =====================================================================================
def summarize(all_metrics, args):
    summary = {"n_episodes": len(all_metrics), "per_cond": {}}
    cond_names = set()
    for m in all_metrics:
        cond_names.update(m["conds"].keys())

    def _col(cond, key):
        return np.asarray([m["conds"][cond][key] for m in all_metrics
                           if cond in m["conds"] and np.isfinite(m["conds"][cond].get(key, np.nan))],
                          dtype=np.float64)

    for cond in sorted(cond_names):
        metrics = _col(cond, "metric_scalar")
        ratios = _col(cond, "scale_ratio_median")          # 鲁棒 headline
        ratios_raw = _col(cond, "scale_ratio_median_raw")  # 未过滤对照
        absrels = _col(cond, "absrel")
        deltas = _col(cond, "delta1_25")
        summary["per_cond"][cond] = {
            # 方法 C：metric 标量分布(均值/方差/变异系数;CV 小 ⇒ 近常数 ⇒ 塌缩)
            "metric_mean": float(metrics.mean()) if metrics.size else float("nan"),
            "metric_std": float(metrics.std()) if metrics.size else float("nan"),
            "metric_cv": float(metrics.std() / (metrics.mean() + 1e-12)) if metrics.size else float("nan"),
            "metric_min": float(metrics.min()) if metrics.size else float("nan"),
            "metric_max": float(metrics.max()) if metrics.size else float("nan"),
            # 方法 A 汇总：真实 scale ratio(pred/sensor)，鲁棒 headline
            "scale_ratio_median_of_medians": float(np.median(ratios)) if ratios.size else float("nan"),
            "scale_ratio_min": float(ratios.min()) if ratios.size else float("nan"),
            "scale_ratio_max": float(ratios.max()) if ratios.size else float("nan"),
            "scale_ratio_cv": float(ratios.std() / (ratios.mean() + 1e-12)) if ratios.size else float("nan"),
            "absrel_mean": float(np.mean(absrels)) if absrels.size else float("nan"),
            "delta1_25_mean": float(np.mean(deltas)) if deltas.size else float("nan"),
            # raw 对照 + 反光严重度诊断(均值)
            "scale_ratio_median_of_medians_raw": float(np.median(ratios_raw)) if ratios_raw.size else float("nan"),
            "absrel_mean_raw": float(np.mean(_col(cond, "absrel_raw"))) if _col(cond, "absrel_raw").size else float("nan"),
            "hole_frac_mean": float(np.mean(_col(cond, "hole_frac"))) if _col(cond, "hole_frac").size else float("nan"),
            "valid_frac_mean": float(np.mean(_col(cond, "valid_frac"))) if _col(cond, "valid_frac").size else float("nan"),
            "outlier_frac_mean": float(np.mean(_col(cond, "outlier_frac"))) if _col(cond, "outlier_frac").size else float("nan"),
        }

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[汇总] 已写 {summary_path}")
    for cond, s in summary["per_cond"].items():
        print(f"  [{cond}] metric: mean={s['metric_mean']:.4f} std={s['metric_std']:.4f} "
              f"CV={s['metric_cv']:.3f} | 真实 scale_ratio(pred/sensor) "
              f"median={s['scale_ratio_median_of_medians']:.4f} "
              f"(raw {s['scale_ratio_median_of_medians_raw']:.4f}) "
              f"[{s['scale_ratio_min']:.3f},{s['scale_ratio_max']:.3f}] CV={s['scale_ratio_cv']:.3f} "
              f"| AbsRel={s['absrel_mean']:.4f} (raw {s['absrel_mean_raw']:.4f}) δ1.25={s['delta1_25_mean']:.4f} "
              f"| hole={s['hole_frac_mean']:.3f} valid_frac={s['valid_frac_mean']:.3f} "
              f"outlier={s['outlier_frac_mean']:.3f}")

    _plot_summary(all_metrics, summary, args)
    return summary


def _plot_summary(all_metrics, summary, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[viz] matplotlib 不可用，跳过出图: {e}")
        return

    cond_names = sorted(summary["per_cond"].keys())
    # 标题/标签一律用 ASCII，避免 DejaVu Sans 缺中文字形导致渲染成方块(tofu)。
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 图1：方法 C —— metric 标量分布(箱线)
    data = [[m["conds"][c]["metric_scalar"] for m in all_metrics if c in m["conds"]]
            for c in cond_names]
    axes[0].boxplot(data, labels=cond_names, showmeans=True)
    axes[0].set_title("Method C: metric scalar distribution\n(near-constant => head collapsed)")
    axes[0].set_ylabel("metric scalar")

    # 图2：方法 A —— 真实 scale ratio(pred/sensor) 逐序列 median 分布
    data2 = [[m["conds"][c]["scale_ratio_median"] for m in all_metrics
              if c in m["conds"] and np.isfinite(m["conds"][c]["scale_ratio_median"])]
             for c in cond_names]
    axes[1].boxplot(data2, labels=cond_names, showmeans=True)
    axes[1].axhline(1.0, color="g", ls="--", lw=1, label="ratio=1.0 (accurate)")
    axes[1].set_title("Method A: scale ratio = pred/sensor\n(=1 accurate, <1 too small, >1 too large)")
    axes[1].set_ylabel("median(pred/sensor)")
    axes[1].legend(fontsize=8)

    # 图3：metric 标量 vs 真实 scale ratio 散点(看 metric 标量数值能否解释真实尺度)
    for c in cond_names:
        xs = [m["conds"][c]["metric_scalar"] for m in all_metrics
              if c in m["conds"] and np.isfinite(m["conds"][c]["scale_ratio_median"])]
        ys = [m["conds"][c]["scale_ratio_median"] for m in all_metrics
              if c in m["conds"] and np.isfinite(m["conds"][c]["scale_ratio_median"])]
        axes[2].scatter(xs, ys, s=18, alpha=0.6, label=c)
    axes[2].axhline(1.0, color="g", ls="--", lw=1)
    axes[2].set_xlabel("metric scalar")
    axes[2].set_ylabel("real scale ratio (pred/sensor)")
    axes[2].set_title("Does metric scalar explain real scale?")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    out_png = os.path.join(args.out_dir, "summary.png")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"[viz] 汇总图已存 {out_png}")


# =====================================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Pi3X metric scale 真实世界准确性验证(不用 GT 对齐)")
    p.add_argument("--input", required=True, help="数据集根(单 unit / 多 unit 父目录 / N1 布局)")
    p.add_argument("--episode", default="all", help="只跑某个 episode id；默认 all")
    p.add_argument("--cond", nargs="+", default=["rgb", "rgb_intr"],
                   choices=["rgb", "rgb_intr"], help="输入条件(可多选)")
    p.add_argument("--intrinsics", default=None, help="可选 K.npy(3x3)，最高优先级覆盖")
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_out"))
    p.add_argument("--interval", type=int, default=None, help="帧下采样间隔；默认用 config 的 interval")
    p.add_argument("--conf-thr", type=float, default=0.1, help="置信度阈值(sigmoid)")
    p.add_argument("--dmin", type=float, default=0.1, help="传感器深度可信下限(米)")
    p.add_argument("--dmax", type=float, default=10.0, help="传感器深度可信上限(米)")
    p.add_argument("--pixel-limit", type=int, default=255000)
    # --- 实采深度反光空洞/异常值鲁棒过滤(默认开，同时输出 *_raw 对照) ---
    p.add_argument("--no-robust", action="store_true",
                   help="关闭鲁棒过滤(空洞腐蚀+sensor边缘+MAD)，回到旧行为")
    p.add_argument("--hole-erode", type=int, default=1,
                   help="有效 sensor 掩码二值腐蚀像素数(剔反光空洞边界飞点)；0=不腐蚀")
    p.add_argument("--sensor-rtol", type=float, default=0.1,
                   help="对 sensor 深度做 depth_edge 的相对阈值(剔深度突变)")
    p.add_argument("--mad-k", type=float, default=3.0,
                   help="log(pred/sensor) 的 med±k·MAD 离群剔除系数；0=不剔")
    p.add_argument("--export-ply", action="store_true", help="方法B: 导出点云 PLY + 地面统计")
    p.add_argument("--cheat-depth", action="store_true",
                   help="自检上界: 把传感器深度喂进模型(ratio 应趋近 1，验证比较逻辑)")
    p.add_argument("--config", default="./L3ROcc/configs/config.yaml")
    p.add_argument("--model-dir", default="./ckpt")
    p.add_argument("--cpu", action="store_true", help="强制 CPU 推理")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    jobs = discover_jobs(args.input, args.episode)
    if not jobs:
        print(f"[error] 在 {args.input} 下未发现任何轨迹(检查路径/布局)")
        sys.exit(1)
    print(f"发现 {len(jobs)} 个 episode 待评估。")

    cli_intr = None
    if args.intrinsics:
        cli_intr = np.load(args.intrinsics).astype(np.float32)
        assert cli_intr.shape == (3, 3), f"K.npy 必须是 3x3，得到 {cli_intr.shape}"

    # 模型只加载一次
    gen = DataGenerator(config_path=args.config, save_dir=args.out_dir,
                        model_dir=args.model_dir, model_type="pi3x")
    if args.interval is not None:
        gen.interval = args.interval

    all_metrics = []
    for job in jobs:
        try:
            m = process_episode(gen, job, args, cli_intr)
        except Exception as e:
            import traceback
            print(f"[error] {job['label']} 处理失败: {e}")
            traceback.print_exc()
            m = None
        if m is not None:
            all_metrics.append(m)

    if not all_metrics:
        print("[error] 没有任何 episode 成功评估。")
        sys.exit(1)
    summarize(all_metrics, args)
    print("\n判读(注意区分 metric 标量 与 scale ratio):"
          "\n  * scale_ratio(pred/sensor) median ≈1.0 ⇒ 真实尺度准, 可直接用; <1 偏小, >1 偏大。"
          "\n  * metric 标量数值(常≈0.3)本身不代表准不准 —— 它只是作用在归一化几何上的内部乘子。"
          "\n  * 若 metric 标量 CV 很小(近常数) 但 scale_ratio 仍随场景明显变化 ⇒ metric_head 已塌缩为常数, "
          "解释了'跨数据集都 0.3', 此时不可把 metric 标量当作真实公制尺度的指示。")


if __name__ == "__main__":
    main()
