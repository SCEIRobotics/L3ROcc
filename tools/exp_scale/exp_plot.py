"""
exp_scale_compare.py 的绘图模块。

可两种方式调用：
  1) 进程内(Linux)：``import exp_plot; exp_plot.plot_episode(dir)`` / ``plot_summary(dir)``；
  2) 独立子进程(Windows)：``python exp_plot.py <dir> [episode|summary]``。
Windows 上必须独立进程运行：一旦 torch 的 MKL 驻留同一解释器，matplotlib 的 numpy/LAPACK
调用会崩溃(0xc06d007f)。

- plot_episode(<dir>)：<dir> 含 metrics.json + plotdata.npz，输出该集 6 张图。
- plot_summary(<dir>)：<dir> 含 summary.json，输出跨 episode 的汇总图。
"""

import os
import sys
import json

# 仅 Windows 需要：在 import numpy 之前强制单线程 MKL，规避 matplotlib 崩溃(0xc06d007f)。
# Linux 无此问题，不设置(也无意义，子进程才在 numpy 前生效)。
if sys.platform == "win32":
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 3D trajectory helpers (第 7 张图与 PLY 导出共用) ------------------------------------

# 与 plot_episode 内 1~6 图保持一致的三变体配色;GT 用黑色。
_C_GT = "#111111"
_C_RGB = "#d9534f"
_C_INT = "#5cb85c"
_C_DC = "#5bc0de"


def _umeyama_sim3(source, target):
    """完整 Umeyama (1991) 闭式 Sim3 — 返回 (s, R, t),使 target ≈ s*R*source + t。

    用 SVD 计算最优旋转。本函数只在 exp_plot.py 内调用:Linux 进程内 / Windows 子进程,
    两种通路都不会与主进程的 torch+MKL 在同一解释器里冲突(主进程的 umeyama_scale 因此
    避开了 SVD,只算标量尺度)。失败/退化情形回退到 identity 变换,保证调用方拿到合法值。
    """
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    ok = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
    src, dst = src[ok], dst[ok]
    if len(src) < 3:
        return 1.0, np.eye(3), np.zeros(3)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_s, dst - mu_d
    H = src_c.T @ dst_c / len(src)
    try:
        U, D, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return 1.0, np.eye(3), np.zeros(3)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1  # 强制 det(R) = +1, 防止镜像反射
    R = Vt.T @ S @ U.T
    var_s = (src_c ** 2).sum() / len(src)
    s = float((D * np.diag(S)).sum() / max(var_s, 1e-12))
    t = mu_d - s * (R @ mu_s)
    return s, R, t


def _apply_sim3(points, s, R, t):
    """对 (N, 3) 点云应用 Sim3 变换:y = s * R @ x + t,返回 (N, 3)。"""
    pts = np.asarray(points, dtype=np.float64)
    return (s * (R @ pts.T)).T + t


def _plot_traj_3d(ax, pos_gt, aligned, M):
    """3D 子图:GT + 三变体(已对齐)叠画。每条轨迹 = 散点 + 细连接线,起点用大 marker。"""
    sets = [("GT", pos_gt, _C_GT)] + [
        ("model", aligned["model"], _C_RGB),
        ("model_int", aligned["model_int"], _C_INT),
        ("model_dc", aligned["model_dc"], _C_DC),
    ]
    for name, p, col in sets:
        ax.plot(p[:, 0], p[:, 1], p[:, 2], "-", color=col, lw=0.9, alpha=0.85)
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=14, color=col, label=name,
                   depthshade=False)
        ax.scatter(p[0, 0], p[0, 1], p[0, 2], s=60, color=col,
                   marker="o", edgecolors="white", linewidths=1.0, depthshade=False)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("3D trajectory (Sim3-aligned to GT)")
    ax.legend(fontsize=8, loc="best")


def _plot_traj_2d(ax, pos_gt, aligned, axes_pair, title):
    """2D 投影子图:axes_pair = (i, j) 选 X/Y/Z 中两轴。"""
    i, j = axes_pair
    label = ["X", "Y", "Z"]
    sets = [("GT", pos_gt, _C_GT)] + [
        ("model", aligned["model"], _C_RGB),
        ("model_int", aligned["model_int"], _C_INT),
        ("model_dc", aligned["model_dc"], _C_DC),
    ]
    for name, p, col in sets:
        ax.plot(p[:, i], p[:, j], "-", color=col, lw=0.9, alpha=0.85)
        ax.scatter(p[:, i], p[:, j], s=12, color=col, label=name)
        ax.scatter(p[0, i], p[0, j], s=55, color=col,
                   marker="o", edgecolors="white", linewidths=1.0)
    ax.set_xlabel(label[i]); ax.set_ylabel(label[j])
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")


def _write_trajectories_ply(path, pos_gt, aligned):
    """合并 4 条轨迹为一份 ASCII PLY,每个顶点带 per-vertex RGB(uint8)。

    GT 黑、model 红、model_int 绿、model_dc 蓝,对应 _plot_traj_* 的视觉约定。
    无 face,顶点顺序 = [GT..., model..., model_int..., model_dc...] (各 n 个)。
    """
    def _hex_to_rgb(h):
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    entries = [
        (pos_gt, _hex_to_rgb(_C_GT)),
        (aligned["model"], _hex_to_rgb(_C_RGB)),
        (aligned["model_int"], _hex_to_rgb(_C_INT)),
        (aligned["model_dc"], _hex_to_rgb(_C_DC)),
    ]
    total = sum(int(p.shape[0]) for p, _ in entries)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {total}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pts, (r, g, b) in entries:
            for x, y, z in pts:
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def plot_episode(out_dir):
    """渲染单个 episode 的 6 张诊断图(读 out_dir 下的 metrics.json + plotdata.npz)。"""
    with open(os.path.join(out_dir, "metrics.json"), "r", encoding="utf-8") as f:
        M = json.load(f)
    d = np.load(os.path.join(out_dir, "plotdata.npz"))
    frame_idx = d["frame_idx"]
    pred_d_rgb = d["pred_d_rgb"]
    pred_d_int = d["pred_d_int"]
    pred_d_dc = d["pred_d_dc"]
    sensor_d, conf_rgb = d["sensor_d"], d["conf_rgb"]
    dmin, dmax, conf_thr = float(d["dmin"]), float(d["dmax"]), float(d["conf_thr"])

    # 统一三变体颜色:RGB-only 红、RGB+intr 绿、RGB+intr+depth 蓝
    C_RGB, C_INT, C_DC = "#d9534f", "#5cb85c", "#5bc0de"

    # 1) Headline: trajectory-length relative error per variant --------------------------
    methods = ["model\n(RGB only)", "model_int\n(RGB + intr)", "model_dc\n(RGB + intr + depth)"]
    errs = [M["e_model"] * 100, M["e_model_int"] * 100, M["e_model_dc"] * 100]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(methods, errs, color=[C_RGB, C_INT, C_DC])
    for b, e in zip(bars, errs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{e:.1f}%",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Trajectory-length error vs odometry GT (%)")
    ax.set_title("Scale accuracy: lower is better")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "1_headline_error.png"), dpi=140); plt.close(fig)

    # 2) Each variant's metric vs its own GT-optimal Umeyama scale ----------------------
    # 三变体都直接信任 metric 头(c=1),对比 1.0 与各自理想 c_gt_*。
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    names = ["model", "model_int", "model_dc"]
    c_gts = [M["c_gt_rgb"], M["c_gt_int"], M["c_gt_dc"]]
    x = np.arange(len(names))
    w = 0.35
    bars_used = ax.bar(x - w / 2, [1.0] * len(names), w, color=[C_RGB, C_INT, C_DC], label="metric head (c=1)")
    bars_ideal = ax.bar(x + w / 2, c_gts, w, color="#999999", label="GT-optimal (Umeyama)")
    for i, v in enumerate(c_gts):
        ax.text(i + w / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Scale")
    ax.set_title("Metric-head scale (=1) vs the GT-optimal scale per variant")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "2_correction_factor.png"), dpi=140); plt.close(fig)

    # 3) Per-frame depth ratio (诊断:每种变体的预测深度是否系统偏离传感器) ---------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(frame_idx, M["s_per_frame_rgb"], "-o", ms=3, color=C_RGB, label="model: median(D_sensor/D_pred)")
    ax.plot(frame_idx, M["s_per_frame_int"], "-o", ms=3, color=C_INT, label="model_int: median(D_sensor/D_pred)")
    ax.plot(frame_idx, M["s_per_frame_dc"], "-o", ms=3, color=C_DC, label="model_dc: median(D_sensor/D_pred)")
    ax.axhline(1.0, color="gray", ls=":", lw=1, label="1.0 (model perfectly metric)")
    ax.set_xlabel("frame index"); ax.set_ylabel("sensor/pred depth ratio")
    ax.set_title("Per-frame sensor/pred depth ratio (drift & bias)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "3_per_frame_scale.png"), dpi=140); plt.close(fig)

    # 4) Cumulative trajectory length ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(M["cum_gt"], "-o", ms=3, color="k", label=f"GT (odometry)  L={M['L_gt']:.3f} m")
    ax.plot(M["cum_model"], "-o", ms=3, color=C_RGB, label=f"model  L={M['L_model']:.3f} m")
    ax.plot(M["cum_model_int"], "-o", ms=3, color=C_INT, label=f"model_int  L={M['L_model_int']:.3f} m")
    ax.plot(M["cum_model_dc"], "-o", ms=3, color=C_DC, label=f"model_dc  L={M['L_model_dc']:.3f} m")
    ax.set_xlabel("kept-frame index"); ax.set_ylabel("cumulative camera path length (m)")
    ax.set_title("Cumulative trajectory length vs GT")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "4_cumulative_length.png"), dpi=140); plt.close(fig)

    # 5) Depth agreement scatter (pred vs sensor) per variant ---------------------------
    def scatter_panel(ax, pred_d, title):
        xs, ys = [], []
        for i in range(pred_d.shape[0]):
            m = (np.isfinite(pred_d[i]) & np.isfinite(sensor_d[i])
                 & (sensor_d[i] > dmin) & (sensor_d[i] < dmax)
                 & (pred_d[i] > 1e-3) & (conf_rgb[i] > conf_thr))
            if m.sum() == 0:
                continue
            idx = np.where(m.ravel())[0]
            if idx.size > 400:
                idx = np.random.choice(idx, 400, replace=False)
            xs.append(sensor_d[i].ravel()[idx]); ys.append(pred_d[i].ravel()[idx])
        if not xs:
            ax.set_title(title + " (no valid px)"); return
        xs = np.concatenate(xs); ys = np.concatenate(ys)
        ax.scatter(xs, ys, s=2, alpha=0.2, color="#337ab7")
        lim = max(dmin, min(dmax, float(np.percentile(np.concatenate([xs, ys]), 99))))
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x")
        slope = float(np.median(ys / xs))
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("sensor depth (m)"); ax.set_ylabel("pred depth (m)")
        ax.set_title(f"{title}\nmedian pred/sensor = {slope:.3f}"); ax.legend(fontsize=8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    scatter_panel(axes[0], pred_d_rgb, "model (RGB only)")
    scatter_panel(axes[1], pred_d_int, "model_int (RGB + intr)")
    scatter_panel(axes[2], pred_d_dc, "model_dc (RGB + intr + depth)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "5_depth_scatter.png"), dpi=140); plt.close(fig)

    # 6) Sample depth error maps --------------------------------------------------------
    n = pred_d_rgb.shape[0]
    sample = sorted(set(np.linspace(0, n - 1, min(3, n)).astype(int)))
    fig, axes = plt.subplots(len(sample), 3, figsize=(11, 3.2 * len(sample)))
    if len(sample) == 1:
        axes = axes[None, :]
    for r, fi in enumerate(sample):
        ds = sensor_d[fi]; ps = pred_d_rgb[fi]
        valid = (ds > dmin) & (ds < dmax)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(valid & (ps > 1e-3), np.abs(ps - ds) / np.where(ds > 0, ds, np.nan), np.nan)
        im0 = axes[r, 0].imshow(np.where(valid, ds, np.nan), cmap="viridis"); axes[r, 0].set_title(f"f{fi} sensor depth"); fig.colorbar(im0, ax=axes[r, 0], fraction=0.046)
        im1 = axes[r, 1].imshow(ps, cmap="viridis"); axes[r, 1].set_title("pred depth (RGB-only)"); fig.colorbar(im1, ax=axes[r, 1], fraction=0.046)
        im2 = axes[r, 2].imshow(rel, cmap="magma", vmin=0, vmax=0.5); axes[r, 2].set_title("|pred-sensor|/sensor"); fig.colorbar(im2, ax=axes[r, 2], fraction=0.046)
        for c in range(3):
            axes[r, c].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "6_depth_error_maps.png"), dpi=130); plt.close(fig)

    # 7) Aligned 3D trajectory point clouds + PLY 导出 ----------------------------------
    # 兼容旧 plotdata.npz (没有 pos_* 字段) — KeyError 直接跳过,前 6 图照常出。
    n_figs = 6
    extras = []
    try:
        pos_gt = d["pos_gt"]
        pos_m  = d["pos_model"]
        pos_i  = d["pos_model_int"]
        pos_dc = d["pos_model_dc"]
    except KeyError:
        pass
    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registration side effect)

        aligned = {
            "model":     _apply_sim3(pos_m,  *_umeyama_sim3(pos_m,  pos_gt)),
            "model_int": _apply_sim3(pos_i,  *_umeyama_sim3(pos_i,  pos_gt)),
            "model_dc":  _apply_sim3(pos_dc, *_umeyama_sim3(pos_dc, pos_gt)),
        }

        fig = plt.figure(figsize=(11, 9))
        _plot_traj_3d(fig.add_subplot(2, 2, 1, projection="3d"), pos_gt, aligned, M)
        _plot_traj_2d(fig.add_subplot(2, 2, 2), pos_gt, aligned, axes_pair=(0, 1), title="XY top-down")
        _plot_traj_2d(fig.add_subplot(2, 2, 3), pos_gt, aligned, axes_pair=(0, 2), title="XZ side")
        _plot_traj_2d(fig.add_subplot(2, 2, 4), pos_gt, aligned, axes_pair=(1, 2), title="YZ side")
        fig.suptitle(
            f"{M.get('episode','?')}  L_gt={M['L_gt']:.3f} m  "
            f"e: rgb={M['e_model']*100:.2f}% / int={M['e_model_int']*100:.2f}% / "
            f"dc={M['e_model_dc']*100:.2f}%",
            fontsize=10,
        )
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "7_trajectories_3d.png"), dpi=140)
        plt.close(fig)
        extras.append("7_trajectories_3d.png")

        # 合并 4 条轨迹的 PLY (per-vertex RGB),可直接拖进 MeshLab/CloudCompare 互动查看。
        ply_path = os.path.join(out_dir, "trajectories.ply")
        _write_trajectories_ply(ply_path, pos_gt, aligned)
        extras.append("trajectories.ply")
        n_figs += 1

    extras_tail = f" + {', '.join(extras)}" if extras else ""
    print(f"[viz] saved {n_figs} figures to {out_dir}{extras_tail}")


def plot_summary(out_dir):
    """渲染跨 episode 汇总图(读 out_dir 下的 summary.json)，输出 summary.png。"""
    with open(os.path.join(out_dir, "summary.json"), "r", encoding="utf-8") as f:
        S = json.load(f)
    rows = S.get("rows", [])
    if not rows:
        print("[viz] summary.json 无有效 episode，跳过汇总图。")
        return

    methods = ["model", "model_int", "model_dc"]
    colors = {"model": "#d9534f", "model_int": "#5cb85c", "model_dc": "#5bc0de"}
    agg = S["agg"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # A) 三变体误差的均值±标准差 -------------------------------------------------------
    means = [agg[f"e_{m}"]["mean"] * 100 if agg.get(f"e_{m}") else 0.0 for m in methods]
    stds = [agg[f"e_{m}"]["std"] * 100 if agg.get(f"e_{m}") else 0.0 for m in methods]
    bars = axes[0].bar(methods, means, yerr=stds, capsize=5,
                       color=[colors[m] for m in methods])
    for b, mu in zip(bars, means):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height(), f"{mu:.1f}%",
                     ha="center", va="bottom", fontsize=10)
    axes[0].set_ylabel("Trajectory-length error vs GT (%)")
    axes[0].set_title(f"Mean±std error  (n={S['n_episodes']})")
    axes[0].grid(axis="y", alpha=0.3)

    # B) 各变体"最优次数" ---------------------------------------------------------------
    wins = S.get("win_counts", {})
    wbars = axes[1].bar(methods, [wins.get(m, 0) for m in methods],
                        color=[colors[m] for m in methods])
    for b, m in zip(wbars, methods):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height(), str(wins.get(m, 0)),
                     ha="center", va="bottom", fontsize=10)
    axes[1].set_ylabel("# episodes where most accurate")
    axes[1].set_title("Win counts")
    axes[1].grid(axis="y", alpha=0.3)

    # C) 各变体的"理想尺度 c_gt_*" 分布 ------------------------------------------------
    # 三变体 metric 头都假定 scale=1.0,理想 c_gt_* 越接近 1 说明该变体 metric 越准。
    c_gt_keys = {"model": "c_gt_rgb", "model_int": "c_gt_int", "model_dc": "c_gt_dc"}
    plotted = False
    for m in methods:
        vals = [r.get(c_gt_keys[m]) for r in rows]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if vals:
            axes[2].scatter([m] * len(vals), vals, s=28, color=colors[m], alpha=0.6, zorder=3)
            plotted = True
    axes[2].axhline(1.0, color="k", ls="--", lw=1, label="ideal = 1.0 (metric head perfect)")
    if plotted:
        axes[2].set_ylabel("GT-optimal scale c_gt (per episode)")
        axes[2].set_title("Per-episode ideal scale per variant\n(closer to 1.0 = metric head is more accurate)")
        axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "summary.png"), dpi=140)
    plt.close(fig)
    print(f"[viz] saved summary figure to {os.path.join(out_dir, 'summary.png')}")


if __name__ == "__main__":
    _dir = sys.argv[1]
    _mode = sys.argv[2] if len(sys.argv) > 2 else "episode"
    (plot_summary if _mode == "summary" else plot_episode)(_dir)
