"""
实验：InternData-N1 数据生成 Pipeline 中各数据的坐标系对齐可视化对比（**不经过任何优化对齐**）。

背景
----
`tools/run_intern_nav_occ.py` 的 pipeline 依赖 `InternNavDataGenerator.align_to_world`
（GT-free 优化对齐：metric 尺度 + 地面 RANSAC 重力 + 帧0 yaw 规范）把 Pi3X 重建拉到世界系。
本脚本是一个**诊断**实验：**绝不调用 align_to_world / align_with_gt_scale**，仅做确定性的
坐标系链式变换，把三套数据放进同一个可比的 **base 系**直接叠加对比，从而量化 Pi3X 重建本身
（尺度 / 朝向 / 漂移）相对 N1 GT 的真实差距——也就是优化对齐实际在修正的误差来源。

三套数据（最终都在 base 系）
---------------------------
1. Pi3X points（点云场景）: world -> 帧0相机系 -> base
2. Pi3X camera_poses（重建轨迹）: world -> 帧0相机系 -> base
3. N1 GT action 轨迹: GT world -> 帧0相机系 -> base

坐标系事实（源码核对）
--------------------
- Pi3X(third_party/pi3/pi3/models/pi3x.py): `points` 在模型 world 系（锚定帧0相机, OpenCV RDF）；
  `camera_poses` 为 camera->world SE(3)（OpenCV），已 metric 缩放。
- N1 parquet `action`: 逐帧 GT camera->world 4x4。
- N1 parquet `observation.camera_extrinsic`: cam->base 4x4，取首行作 T_cam2base（下游仅用旋转）。

设计决策
----------------
- 参考系：各自帧0相机参考——Pi3X 用 inv(camera_poses[0])、N1 用 inv(action[0]) 归一到各自帧0相机系，
  再各自经 T_cam2base 到 base。无需任何优化对齐即可叠加。
- camera->base 仅用旋转（与生产 convert_pointcloud_camera_to_base 一致）。
- 不叠加 align_to_world 结果，只展示未优化结果。

相机约定翻转修正（C，OpenCV<->OpenGL）
-------------------------------------
Pi3X 相机系是 OpenCV (X-right, Y-down, Z-forward)，N1 action/T_cam2base 是 3D-Front 渲染相机的
OpenGL 约定 (X-right, Y-up, Z-back)，两者相差 C=R_OPENCV_TO_OPENGL=diag(1,-1,-1)（翻转 Y、Z = 绕
相机 X 轴 180°）。不换基会使 base 系 Pi3X 与 GT 整体绕 X 轴翻转(且 Z 朝下)。
本实验对 Pi3X 数据应用 C（fixed），对 N1 action 不应用，并同时输出未修正(raw)、修正前后误差
(pos_err_*_raw/_fixed)。
生产Pipeline已在 L3ROcc/base.py 同步修复（把 C 加入 compute_sequence_data 的 T_cam2base 中）。

用法
----
    python tools/exp_coord_align/exp_coord_align.py --dataset_root <数据集根> \
        [--out_dir tools/exp_coord_align/exp_out] [--episode all] \
        [--model_type pi3x] [--model_dir ./ckpt] [--config ./L3ROcc/configs/config.yaml] \
        [--no_intrinsic] [--use_depth] [--cpu]
"""

import os
import sys
import json
import argparse
import faulthandler

faulthandler.enable()

# 强制 CPU 必须在 import torch 前设置，故先扫 argv。
if "--cpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) // 2)))

# 向上搜索定位项目根目录（含 L3ROcc/ 与 third_party/），让本地包可被 import。
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
while _PROJECT_ROOT != os.path.dirname(_PROJECT_ROOT):
    if os.path.isdir(os.path.join(_PROJECT_ROOT, "L3ROcc")) and os.path.isdir(
        os.path.join(_PROJECT_ROOT, "third_party")
    ):
        break
    _PROJECT_ROOT = os.path.dirname(_PROJECT_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from L3ROcc.dataset.intern_nav_adapter import InternNavSequenceLoader  # noqa: E402
from L3ROcc.generater.intern_vln_env import InternNavDataGenerator  # noqa: E402
from L3ROcc.base import R_OPENCV_TO_OPENGL  # noqa: E402 (=diag(1,-1,-1), 与生产一致)
from third_party.pi3.pi3.utils.basic import write_ply  # noqa: E402


# =====================================================================================
# 轨迹 PLY 加密：在相邻位姿点之间线性插值，使稀疏轨迹在点云查看器里看起来连成线。
# =====================================================================================
def densify_polyline(points, steps_per_seg=8):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return points.astype(np.float32)
    out = []
    for a, b in zip(points[:-1], points[1:]):
        ts = np.linspace(0.0, 1.0, steps_per_seg, endpoint=False)[:, None]
        out.append(a[None] + ts * (b - a)[None])
    out.append(points[-1:][None].reshape(1, 3))
    return np.concatenate(out, axis=0).astype(np.float32)


def solid_color(n, rgb):
    return np.tile(np.asarray(rgb, dtype=np.float32)[None], (n, 1))


# =====================================================================================
# 未优化坐标变换：world -> 帧0相机系 -> base（复用 generator 的转换函数，仅用旋转到 base）
# =====================================================================================
def world_to_base_via_frame0(gen, points_world, T0_cam2world, T_cam2base, convention_fix=None):
    """把世界系下的点变换到 base 系（不经任何优化对齐）：
    1) world -> 帧0相机系：用 T0（cam0->world）做 convert_pointcloud_world_to_camera；
    2) （可选）相机约定换基：Pi3X 是 OpenCV 相机系，需先 ``p @ C.T`` 换基到 N1 OpenGL 渲染相机系，
       否则后续 T_cam2base 会让 base 系与 GT 绕 X 轴翻转；``convention_fix`` 传 ``R_OPENCV_TO_OPENGL``。
       N1 ``action`` 本就是 OpenGL 约定，传 None。
    3) 帧0相机系 -> base：用 T_cam2base 做 convert_pointcloud_camera_to_base（仅旋转）。
    """
    pts_cam0 = gen.convert_pointcloud_world_to_camera(points_world, T0_cam2world)
    pts_cam0 = np.asarray(pts_cam0, dtype=np.float64)
    if convention_fix is not None:
        pts_cam0 = pts_cam0 @ np.asarray(convention_fix, dtype=np.float64).T
    if T_cam2base is not None:
        pts_base = gen.convert_pointcloud_camera_to_base(pts_cam0, T_cam2base)
    else:
        pts_base = pts_cam0
    return np.asarray(pts_base, dtype=np.float64)


def kabsch_rfix(traj_pi3x_raw, traj_gt, T_cam2base):
    """护栏：从未修正的 Pi3X base 轨迹与 GT base 轨迹做 Kabsch，反求 base 系修正旋转 E
    (pi3x_raw -> gt)，并换回相机系 C_emp = R_c2bᵀ · E · R_c2b，与几何常量 C=R_OPENCV_TO_OPENGL 对比。
    返回 dict：经验 E、C_emp 对角、det、是否匹配、应用 E 后的相对残差。防止再次用错常量。
    """
    if traj_gt is None or len(traj_gt) < 3:
        return None
    n = min(len(traj_pi3x_raw), len(traj_gt))
    P = np.asarray(traj_pi3x_raw[:n], dtype=np.float64)
    Q = np.asarray(traj_gt[:n], dtype=np.float64)
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    det_raw = float(np.linalg.det(Vt.T @ U.T))
    d = np.sign(det_raw) if det_raw != 0 else 1.0
    E = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T  # proper rotation, pi3x_raw -> gt (base 系)
    resid = float(
        np.linalg.norm((E @ Pc.T).T - Qc) / (np.linalg.norm(Qc) + 1e-9)
    )
    out = {
        "kabsch_det_raw": det_raw,
        "R_fix_base": E.tolist(),
        "R_fix_base_diag": np.diag(E).round(3).tolist(),
        "resid_after_Rfix": resid,
    }
    if T_cam2base is not None:
        R_c2b = np.asarray(T_cam2base, dtype=np.float64)[:3, :3]
        C_emp = R_c2b.T @ E @ R_c2b
        C = np.asarray(R_OPENCV_TO_OPENGL, dtype=np.float64)
        out["C_emp_diag"] = np.diag(C_emp).round(3).tolist()
        out["c_matches_empirical"] = bool(np.allclose(C_emp, C, atol=0.2))
        out["C_emp_frob_diff_to_const"] = float(np.linalg.norm(C_emp - C))
    return out


# =====================================================================================
# 轨迹对比度量
# =====================================================================================
def polyline_length(pts):
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _traj_vs_gt(traj_pi3x, traj_gt):
    """Pi3X 轨迹 vs GT 的逐帧位置误差摘要。"""
    n = min(len(traj_pi3x), len(traj_gt))
    a, b = traj_pi3x[:n], traj_gt[:n]
    err = np.linalg.norm(a - b, axis=1)
    return {
        "pos_err_mean": float(err.mean()),
        "pos_err_median": float(np.median(err)),
        "pos_err_max": float(err.max()),
        "endpoint_err": float(np.linalg.norm(a[-1] - b[-1])),
    }


def compute_metrics(traj_pi3x_fixed, traj_pi3x_raw, traj_gt, pcd_base):
    """度量。同时报告相机约定修正前(raw, convention_fix=None)与修正后(fixed, C=R_OPENCV_TO_OPENGL)
    的 Pi3X↔GT 误差，量化 ``C`` 是否消除了绕 X 轴翻转。"""
    M = {}
    M["n_frames_pi3x"] = int(len(traj_pi3x_fixed))
    M["n_frames_gt"] = int(len(traj_gt)) if traj_gt is not None else 0
    M["convention_fix_diag"] = np.diag(R_OPENCV_TO_OPENGL).tolist()
    if pcd_base is not None and len(pcd_base) > 0:
        M["pcd_num"] = int(len(pcd_base))
        M["pcd_bbox_min"] = pcd_base.min(0).tolist()
        M["pcd_bbox_max"] = pcd_base.max(0).tolist()
    M["traj_len_pi3x"] = polyline_length(traj_pi3x_fixed)
    if traj_gt is not None and len(traj_gt) >= 1:
        M["traj_len_gt"] = polyline_length(traj_gt)
        M["scale_ratio_pi3x_over_gt"] = (
            float(M["traj_len_pi3x"] / M["traj_len_gt"])
            if M["traj_len_gt"] > 1e-9
            else None
        )
        for k, v in _traj_vs_gt(traj_pi3x_fixed, traj_gt).items():
            M[f"{k}_fixed"] = v
        for k, v in _traj_vs_gt(traj_pi3x_raw, traj_gt).items():
            M[f"{k}_raw"] = v
    return M


# =====================================================================================
# 可视化：3D + 三视图叠加 Pi3X vs GT 轨迹
# =====================================================================================
def plot_compare(traj_pi3x_fixed, traj_pi3x_raw, traj_gt, pcd_base, out_png, title):
    """叠加 GT(红)、Pi3X 修正后(蓝实线)、Pi3X 修正前(青虚线, 绕 X 轴翻转态)。"""
    fig = plt.figure(figsize=(14, 10))

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    if pcd_base is not None and len(pcd_base) > 0:
        sub = pcd_base
        if len(sub) > 20000:
            sub = sub[np.random.choice(len(sub), 20000, replace=False)]
        ax3d.scatter(
            sub[:, 0], sub[:, 1], sub[:, 2], s=0.4, c="lightgray", alpha=0.35
        )
    ax3d.plot(
        traj_pi3x_fixed[:, 0], traj_pi3x_fixed[:, 1], traj_pi3x_fixed[:, 2],
        "-o", ms=2, c="tab:blue", label="Pi3X fixed (C)",
    )
    ax3d.plot(
        traj_pi3x_raw[:, 0], traj_pi3x_raw[:, 1], traj_pi3x_raw[:, 2],
        "--", lw=1, c="tab:cyan", label="Pi3X raw (flipped)",
    )
    if traj_gt is not None:
        ax3d.plot(
            traj_gt[:, 0], traj_gt[:, 1], traj_gt[:, 2], "-o", ms=2, c="tab:red",
            label="N1 GT (base)",
        )
    ax3d.scatter([0], [0], [0], c="k", s=40, marker="*", label="frame0 origin")
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.set_title("3D (base frame)")
    ax3d.legend(loc="upper right", fontsize=8)

    planes = [("XY", 0, 1), ("XZ", 0, 2), ("YZ", 1, 2)]
    for k, (name, i, j) in enumerate(planes):
        ax = fig.add_subplot(2, 2, k + 2)
        if pcd_base is not None and len(pcd_base) > 0:
            ax.scatter(
                pcd_base[:, i], pcd_base[:, j], s=0.3, c="lightgray", alpha=0.3
            )
        ax.plot(traj_pi3x_fixed[:, i], traj_pi3x_fixed[:, j], "-o", ms=2,
                c="tab:blue", label="Pi3X fixed")
        ax.plot(traj_pi3x_raw[:, i], traj_pi3x_raw[:, j], "--", lw=1,
                c="tab:cyan", label="Pi3X raw")
        if traj_gt is not None:
            ax.plot(traj_gt[:, i], traj_gt[:, j], "-o", ms=2, c="tab:red", label="GT")
        ax.scatter([0], [0], c="k", s=40, marker="*")
        ax.set_xlabel(name[0]); ax.set_ylabel(name[1])
        ax.set_title(f"{name} 投影"); ax.axis("equal"); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# =====================================================================================
# 单条轨迹处理
# =====================================================================================
def process_trajectory(gen, loader, idx, args):
    video_path, depth_path, cam_intrinsics, T_cam2base = loader.get_trajectory_info(idx)
    if video_path is None:
        print(f"[skip] trajectory {idx}: 无 RGB 视频")
        return None

    unit_dir = loader.trajectory_dirs[idx]
    ep = os.path.splitext(os.path.basename(video_path))[0]
    rb_name = os.path.basename(unit_dir.rstrip("/\\"))
    label = ep if os.path.abspath(unit_dir) == os.path.abspath(args.dataset_root) else f"{rb_name}__{ep}"
    out_dir = os.path.join(args.out_dir, label)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== [{idx}] {label} ===")
    print(f"  video: {video_path}")
    if T_cam2base is None:
        print("  [warn] 无 observation.camera_extrinsic / 手眼外参，T_cam2base=None -> base 等同帧0相机系。")

    condit_depth_path = depth_path if (args.use_depth and depth_path and os.path.isfile(depth_path)) else None
    intrinsics_np = (
        cam_intrinsics.astype(np.float32)
        if (not args.no_intrinsic and cam_intrinsics is not None)
        else None
    )

    # --- 1) Pi3X 重建（world 系），复用生产路径，绝不调用 align_to_world ---
    pcd_world, cam_pose_world, _ = gen.pcd_reconstruction(
        video_path, condit_depth_path=condit_depth_path, intrinsics_np=intrinsics_np
    )
    pcd_world = np.asarray(pcd_world, dtype=np.float64)
    cam_pose_world = np.asarray(cam_pose_world, dtype=np.float64)
    pcd_color = getattr(gen, "pcd_color", None)

    # --- 2) N1 GT action 轨迹（GT world，cam->world） ---
    gt_poses = gen.get_gt_poses(video_path)

    # --- 3) 未优化坐标变换：world -> 帧0相机系 -> base ---
    # Pi3X 是 OpenCV 相机系，需用 C=R_OPENCV_TO_OPENGL 换基到 N1 OpenGL 渲染相机系（fixed）；
    # 不换基则绕 X 轴翻转（raw）。N1 action 本就是 OpenGL 约定，不换基。
    T0_p = cam_pose_world[0]
    pcd_base = world_to_base_via_frame0(
        gen, pcd_world, T0_p, T_cam2base, convention_fix=R_OPENCV_TO_OPENGL
    )
    traj_pi3x_base_fixed = world_to_base_via_frame0(
        gen, cam_pose_world[:, :3, 3], T0_p, T_cam2base, convention_fix=R_OPENCV_TO_OPENGL
    )
    traj_pi3x_base_raw = world_to_base_via_frame0(
        gen, cam_pose_world[:, :3, 3], T0_p, T_cam2base, convention_fix=None
    )

    traj_gt_base = None
    if gt_poses is not None and len(gt_poses) >= 1:
        gt_poses = np.asarray(gt_poses, dtype=np.float64)
        T0_g = gt_poses[0]
        traj_gt_base = world_to_base_via_frame0(
            gen, gt_poses[:, :3, 3], T0_g, T_cam2base, convention_fix=None
        )
    else:
        print("  [warn] 无 GT action 轨迹，仅输出 Pi3X 两套数据。")

    # --- 4) 落盘可视化产物 ---
    write_ply(pcd_base, pcd_color, os.path.join(out_dir, "scene_base.ply"))
    traj_p_dense = densify_polyline(traj_pi3x_base_fixed)
    write_ply(
        traj_p_dense,
        solid_color(len(traj_p_dense), [0.1, 0.4, 0.95]),
        os.path.join(out_dir, "traj_pi3x_base.ply"),
    )
    traj_p_raw_dense = densify_polyline(traj_pi3x_base_raw)
    write_ply(
        traj_p_raw_dense,
        solid_color(len(traj_p_raw_dense), [0.4, 0.8, 0.95]),
        os.path.join(out_dir, "traj_pi3x_base_raw.ply"),
    )
    if traj_gt_base is not None:
        traj_g_dense = densify_polyline(traj_gt_base)
        write_ply(
            traj_g_dense,
            solid_color(len(traj_g_dense), [0.95, 0.2, 0.15]),
            os.path.join(out_dir, "traj_gt_base.ply"),
        )

    plot_compare(
        traj_pi3x_base_fixed, traj_pi3x_base_raw, traj_gt_base, pcd_base,
        os.path.join(out_dir, "compare_traj.png"),
        title=f"{label}  (base frame, no align_to_world; C=OpenCV->OpenGL)",
    )

    # --- 5) 度量 + 经验 R_fix 自检（护栏）---
    M = compute_metrics(traj_pi3x_base_fixed, traj_pi3x_base_raw, traj_gt_base, pcd_base)
    selfcheck = kabsch_rfix(traj_pi3x_base_raw, traj_gt_base, T_cam2base)
    if selfcheck is not None:
        M["rfix_selfcheck"] = selfcheck
    M["label"] = label
    M["video_path"] = video_path
    M["T_cam2base_available"] = bool(T_cam2base is not None)
    M["use_depth"] = bool(condit_depth_path is not None)
    M["use_intrinsic"] = bool(intrinsics_np is not None)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(M, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        for k, v in M.items():
            f.write(f"{k}: {v}\n")

    sr = M.get("scale_ratio_pi3x_over_gt")
    pe_fixed = M.get("pos_err_mean_fixed")
    pe_raw = M.get("pos_err_mean_raw")
    sc = M.get("rfix_selfcheck") or {}
    print(
        f"  -> frames pi3x/gt = {M['n_frames_pi3x']}/{M['n_frames_gt']}, "
        f"scale_ratio(pi3x/gt) = {sr if sr is None else round(sr, 4)}, "
        f"pos_err_mean raw->fixed = "
        f"{pe_raw if pe_raw is None else round(pe_raw, 4)} -> "
        f"{pe_fixed if pe_fixed is None else round(pe_fixed, 4)} m"
    )
    if sc:
        print(
            f"  -> R_fix self-check: R_fix_base_diag={sc.get('R_fix_base_diag')}, "
            f"C_emp_diag={sc.get('C_emp_diag')}, "
            f"c_matches_empirical={sc.get('c_matches_empirical')}, "
            f"resid_after_Rfix={round(sc.get('resid_after_Rfix', 0.0), 4)}"
        )
    print(f"  outputs -> {out_dir}")
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_root", required=True, help="InternData-N1 数据集根 / 单 trajectory / 父目录")
    ap.add_argument("--out_dir", default=os.path.join(_PROJECT_ROOT, "tools/exp_coord_align/exp_out"))
    ap.add_argument("--episode", default="all", help="episode id（如 episode_000000）或 all；默认 all")
    ap.add_argument("--model_type", default="pi3x", choices=["pi3", "pi3x"])
    ap.add_argument("--model_dir", default="./ckpt")
    ap.add_argument("--config", default="./L3ROcc/configs/config.yaml")
    ap.add_argument("--no_intrinsic", action="store_true", help="不使用标定内参（回退 DLT 估计）")
    ap.add_argument("--use_depth", action="store_true", help="把公制深度喂入 Pi3X（默认关）")
    ap.add_argument("--cpu", action="store_true", help="强制 CPU 推理")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    loader = InternNavSequenceLoader(args.dataset_root)
    gen = InternNavDataGenerator(
        config_path=args.config,
        save_dir=args.out_dir,
        model_dir=args.model_dir,
        model_type=args.model_type,
    )

    all_metrics = []
    for i in range(len(loader)):
        ep = os.path.splitext(os.path.basename(loader.trajectory_video_paths[i]))[0]
        if args.episode != "all" and ep != args.episode:
            continue
        try:
            m = process_trajectory(gen, loader, i, args)
            if m is not None:
                all_metrics.append(m)
        except Exception as e:
            import traceback

            print(f"[error] trajectory {i} 失败: {e}")
            traceback.print_exc()

    if all_metrics:
        with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        print(f"\n[done] {len(all_metrics)} 条轨迹，汇总 -> {os.path.join(args.out_dir, 'summary.json')}")
    else:
        print("\n[done] 没有任何轨迹成功处理。")


if __name__ == "__main__":
    main()
