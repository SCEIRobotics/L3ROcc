"""
最小可行实验(MVP)：Pi3X 在以下两种输入下的三维重建是否有差异？
    A: 仅 RGB             (intrinsics=None, depths=None)
    B: RGB + 真实标定内参 (intrinsics=K_real, depths=None)

回答的核心问题：Pi3X 到底吃不吃内参 conditioning？

只采集两个关键指标：
    1) estimate_intrinsics(local_points) 与"输入内参"之间的相对误差
       - A 组：与 info.json 原始内参(按 resize 比例修正)比；模型完全自由
       - B 组：与作为条件传入的内参(同分辨率)比；若模型遵循条件，误差应接近 0
    2) 同一帧(第 0 帧)的点云保存为 PLY，便于在 MeshLab/CloudCompare 中叠加查看几何差异

用法示例：
    python tools/exp_intrinsic/exp_intrinsic_compare.py \
        --video_path /path/to/episode_001.mp4 \
        --intr_path  /path/to/meta/info.json \
        --out_dir    tools/exp_intrinsic/exp_out/episode_001
"""

import os
import sys
import json
import argparse
import faulthandler

faulthandler.enable()

if "--cpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) // 2)))
else:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
while _PROJECT_ROOT != os.path.dirname(_PROJECT_ROOT):
    if os.path.isdir(os.path.join(_PROJECT_ROOT, "L3ROcc")) and \
       os.path.isdir(os.path.join(_PROJECT_ROOT, "third_party")):
        break
    _PROJECT_ROOT = os.path.dirname(_PROJECT_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Windows safetensors 规避(与 exp_scale_compare.py 相同的预加载技巧)
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

import numpy as np
import torch

from L3ROcc.utils import load_images_as_tensor, estimate_intrinsics
from third_party.pi3.pi3.utils.basic import write_ply


def load_intrinsic_from_info(intr_path):
    """从 lerobot info.json 读取 head_camera_intrinsic (3x3)。"""
    with open(intr_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    K = np.array(info["head_camera_intrinsic"], dtype=np.float32)
    assert K.shape == (3, 3), f"expect 3x3 intrinsic, got {K.shape}"
    return K


def rescale_intrinsic(K, scale_x, scale_y):
    """按 resize 比例修正内参,与 utils.load_images_as_tensor 中的逻辑一致。"""
    K2 = K.astype(np.float32).copy()
    K2[0, 0] *= scale_x  # fx
    K2[0, 2] *= scale_x  # cx
    K2[1, 1] *= scale_y  # fy
    K2[1, 2] *= scale_y  # cy
    return K2


def relative_K_error(K_ref, K_est):
    """返回 (相对 Frobenius 误差, 各分量绝对相对误差 dict)。"""
    K_ref = np.asarray(K_ref, dtype=np.float64)
    K_est = np.asarray(K_est, dtype=np.float64)
    fro_err = np.linalg.norm(K_ref - K_est) / max(np.linalg.norm(K_ref), 1e-12)
    parts = {}
    for name, idx in [("fx", (0, 0)), ("fy", (1, 1)), ("cx", (0, 2)), ("cy", (1, 2))]:
        v_ref = K_ref[idx]
        v_est = K_est[idx]
        parts[name] = {
            "ref": float(v_ref),
            "est": float(v_est),
            "abs_err": float(abs(v_ref - v_est)),
            "rel_err": float(abs(v_ref - v_est) / max(abs(v_ref), 1e-12)),
        }
    return float(fro_err), parts


def select_amp_dtype(device):
    if device != "cuda":
        return torch.float16
    try:
        if torch.cuda.get_device_capability()[0] < 8:
            return torch.float16
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel
        probe = torch.zeros(1, 1, 8, 16, device="cuda", dtype=torch.bfloat16)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            F.scaled_dot_product_attention(probe, probe, probe)
        return torch.bfloat16
    except Exception:
        return torch.float16


def run_pi3x(model, imgs, conditions, device, amp_dtype):
    """统一推理入口：返回 res dict。"""
    with torch.no_grad():
        with torch.amp.autocast("cuda" if device == "cuda" else "cpu", dtype=amp_dtype):
            res = model(imgs[None], **conditions)
    return res


def extract_frame0_pcd(res, imgs):
    """从 res 中抽取第 0 帧的世界点云(过滤掉低置信和深度边缘)，附带颜色。"""
    from third_party.pi3.pi3.utils.geometry import depth_edge

    masks = torch.sigmoid(res["conf"][..., 0]) > 0.1
    non_edge = ~depth_edge(res["local_points"][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0]  # (N, H, W)

    points = res["points"][0]                # (N, H, W, 3)
    f0_mask = masks[0]
    pcd = points[0][f0_mask].detach().cpu().numpy()
    color = imgs[0].permute(1, 2, 0)[f0_mask].detach().cpu().numpy()
    return pcd.astype(np.float32), color.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Pi3X RGB-only vs RGB+intrinsic MVP comparison.")
    parser.add_argument("--video_path", type=str, required=True,
                        help="单个 RGB 视频(.mp4)路径")
    parser.add_argument("--intr_path", type=str, required=True,
                        help="lerobot meta/info.json,从中读取 head_camera_intrinsic")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="输出目录(PLY + report.json)")
    parser.add_argument("--ckpt_dir", type=str,
                        default=os.path.join(_PROJECT_ROOT, "ckpt", "pi3x"))
    parser.add_argument("--interval", type=int, default=10,
                        help="视频帧抽样间隔,与主流程一致即可")
    parser.add_argument("--max_frames", type=int, default=12,
                        help="只取前 N 帧,避免长视频显存压力")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda"
    print(f"[setup] device={device}  ckpt={args.ckpt_dir}")

    # ---------- 1. 读取 RGB + 真实内参 ----------
    K_orig = load_intrinsic_from_info(args.intr_path)
    print(f"[input] K_orig (info.json) =\n{K_orig}")

    imgs, _, conds_full = load_images_as_tensor(
        args.video_path,
        interval=args.interval,
        condit_depth_path=None,
        intrinsics_np=K_orig,
        device=device,
    )
    if args.max_frames and imgs.shape[0] > args.max_frames:
        imgs = imgs[: args.max_frames]
        if conds_full["intrinsics"] is not None:
            conds_full["intrinsics"] = conds_full["intrinsics"][:, : args.max_frames]
        print(f"[trim] keep first {args.max_frames} frames")
    imgs = imgs.to(device)
    N, _, H, W = imgs.shape
    print(f"[input] imgs shape={tuple(imgs.shape)}  (resized to {W}x{H})")

    # 计算 resize 比例并构造"resize 后的真实内参"作为参考
    import cv2
    cap = cv2.VideoCapture(args.video_path)
    W_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    scale_x = W / W_orig
    scale_y = H / H_orig
    K_rescaled = rescale_intrinsic(K_orig, scale_x, scale_y)
    print(f"[input] scale_x={scale_x:.4f} scale_y={scale_y:.4f}")
    print(f"[input] K_rescaled =\n{K_rescaled}")

    # ---------- 2. 加载 Pi3X ----------
    from third_party.pi3.pi3.models.pi3x import Pi3X
    model = Pi3X.from_pretrained(args.ckpt_dir).to(device).eval()
    amp_dtype = select_amp_dtype(device)
    print(f"[setup] amp_dtype={amp_dtype}")

    # ---------- 3. 两次推理 ----------
    results = {}
    for tag, conditions in [
        ("A_rgb_only", {"poses": None, "depths": None, "intrinsics": None}),
        ("B_rgb_plus_intr", {"poses": None, "depths": None,
                             "intrinsics": conds_full["intrinsics"]}),
    ]:
        print(f"\n========== Running variant [{tag}] ==========")
        res = run_pi3x(model, imgs, conditions, device, amp_dtype)

        # 反算内参(从 frame 0 的 local_points)
        K_est = estimate_intrinsics(res["local_points"][0][0]).detach().cpu().numpy()
        fro, parts = relative_K_error(K_rescaled, K_est)
        print(f"[{tag}] K_estimated =\n{K_est}")
        print(f"[{tag}] frobenius_rel_err vs K_rescaled = {fro:.4%}")
        for name, d in parts.items():
            print(f"           {name}: ref={d['ref']:.3f}  est={d['est']:.3f}  "
                  f"rel_err={d['rel_err']:.4%}")

        # 保存第 0 帧点云
        pcd, color = extract_frame0_pcd(res, imgs)
        ply_path = os.path.join(args.out_dir, f"{tag}_frame0.ply")
        write_ply(pcd, color, ply_path)
        print(f"[{tag}] saved pcd: {ply_path}  ({pcd.shape[0]} pts)")

        # camera_poses[0] 应当是恒等(参考系),记录 [1] 的位姿差异以备查
        cam_poses = res["camera_poses"][0].detach().cpu().numpy()
        results[tag] = {
            "K_estimated": K_est.tolist(),
            "K_rescaled_ref": K_rescaled.tolist(),
            "fro_rel_err": fro,
            "parts": parts,
            "num_points_frame0": int(pcd.shape[0]),
            "ply_path": ply_path,
            "pose1_translation": cam_poses[1, :3, 3].tolist() if len(cam_poses) > 1 else None,
        }

    # ---------- 4. 跨变体比较 ----------
    if "A_rgb_only" in results and "B_rgb_plus_intr" in results:
        Ka = np.array(results["A_rgb_only"]["K_estimated"])
        Kb = np.array(results["B_rgb_plus_intr"]["K_estimated"])
        ab_fro, ab_parts = relative_K_error(Ka, Kb)
        print("\n========== A vs B (estimated K difference) ==========")
        print(f"frobenius_rel_err = {ab_fro:.4%}")
        for name, d in ab_parts.items():
            print(f"  {name}: A={d['ref']:.3f}  B={d['est']:.3f}  rel_diff={d['rel_err']:.4%}")
        results["A_vs_B"] = {"fro_rel_err": ab_fro, "parts": ab_parts}

    # ---------- 5. Report output ----------
    report = {
        "video_path": args.video_path,
        "intr_path": args.intr_path,
        "resized_to": [int(W), int(H)],
        "original_size": [int(W_orig), int(H_orig)],
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "K_orig": K_orig.tolist(),
        "K_rescaled": K_rescaled.tolist(),
        "num_frames_used": int(N),
        "amp_dtype": str(amp_dtype),
        "results": results,
    }
    report_path = os.path.join(args.out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[done] report saved to {report_path}")
    print(f"[done] open the two PLY files in MeshLab/CloudCompare to compare geometry visually.")


if __name__ == "__main__":
    main()
