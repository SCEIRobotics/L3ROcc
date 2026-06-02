"""
Standalone plotter for exp_scale_compare.py.

Run in its OWN process (no torch import) -- on this Windows env matplotlib's numpy/LAPACK
calls crash (0xc06d007f) once torch's MKL is resident in the same interpreter.

Usage:
    python tools/exp_plot.py <out_dir>
where <out_dir> contains metrics.json and plotdata.npz (written by exp_scale_compare.py).
"""

import os
import sys
import json

# numpy 2.2.6 + MKL on this Windows env crashes matplotlib (0xc06d007f) under the default
# threading layer. Force a single-threaded sequential MKL BEFORE numpy is imported.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(out_dir):
    with open(os.path.join(out_dir, "metrics.json"), "r", encoding="utf-8") as f:
        M = json.load(f)
    d = np.load(os.path.join(out_dir, "plotdata.npz"))
    frame_idx = d["frame_idx"]
    pred_d_rgb, pred_d_dc = d["pred_d_rgb"], d["pred_d_dc"]
    sensor_d, conf_rgb = d["sensor_d"], d["conf_rgb"]
    dmin, dmax, conf_thr = float(d["dmin"]), float(d["dmax"]), float(d["conf_thr"])

    # 1) Headline: trajectory-length relative error per method ---------------------------
    methods = ["model\n(RGB metric head)", "depth-scaled\n(RGB + sensor)", "model_dc\n(depth-conditioned)"]
    errs = [M["e_model"] * 100, M["e_depth"] * 100, M["e_model_dc"] * 100]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(methods, errs, color=["#d9534f", "#5cb85c", "#5bc0de"])
    for b, e in zip(bars, errs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{e:.1f}%",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Trajectory-length error vs odometry GT (%)")
    ax.set_title("Scale accuracy: lower is better")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "1_headline_error.png"), dpi=140); plt.close(fig)

    # 2) Correction factor vs the ideal (Umeyama) factor --------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = ["model (c=1)", "depth (s_depth)", "GT-optimal (Umeyama)"]
    vals = [1.0, M["s_depth"], M["c_gt_rgb"]]
    ax.bar(names, vals, color=["#d9534f", "#5cb85c", "#999999"])
    ax.axhline(M["c_gt_rgb"], color="k", ls="--", lw=1, label=f"ideal c_gt={M['c_gt_rgb']:.3f}")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Correction factor on RGB-only reconstruction")
    ax.set_title("Which correction lands closest to the GT-optimal scale?")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "2_correction_factor.png"), dpi=140); plt.close(fig)

    # 3) Per-frame scale drift ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(frame_idx, M["s_per_frame_rgb"], "-o", ms=3, color="#d9534f", label="RGB-only: median(D_sensor/D_pred)")
    ax.plot(frame_idx, M["s_per_frame_dc"], "-o", ms=3, color="#5bc0de", label="depth-cond: median(D_sensor/D_pred)")
    ax.axhline(M["c_gt_rgb"], color="k", ls="--", lw=1, label=f"GT-optimal scale = {M['c_gt_rgb']:.3f}")
    ax.axhline(1.0, color="gray", ls=":", lw=1, label="1.0 (model perfectly metric)")
    ax.set_xlabel("frame index"); ax.set_ylabel("sensor/pred depth ratio")
    ax.set_title("Per-frame implied scale (drift & bias)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "3_per_frame_scale.png"), dpi=140); plt.close(fig)

    # 4) Cumulative trajectory length ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(M["cum_gt"], "-o", ms=3, color="k", label=f"GT (odometry)  L={M['L_gt']:.3f} m")
    ax.plot(M["cum_model"], "-o", ms=3, color="#d9534f", label=f"model  L={M['L_model']:.3f} m")
    ax.plot(M["cum_depth"], "-o", ms=3, color="#5cb85c", label=f"depth-scaled  L={M['s_depth']*M['L_model']:.3f} m")
    ax.plot(M["cum_dc"], "-o", ms=3, color="#5bc0de", label=f"model_dc  L={M['L_model_dc']:.3f} m")
    ax.set_xlabel("kept-frame index"); ax.set_ylabel("cumulative camera path length (m)")
    ax.set_title("Cumulative trajectory length vs GT")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "4_cumulative_length.png"), dpi=140); plt.close(fig)

    # 5) Depth agreement scatter (pred vs sensor) ---------------------------------------
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

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    scatter_panel(axes[0], pred_d_rgb, "RGB-only")
    scatter_panel(axes[1], pred_d_dc, "depth-conditioned")
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

    print(f"[viz] saved 6 figures to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
