"""
实验：对于 Pi3X 的三维重建，哪种"输入模态组合"给出的绝对尺度(scale)更准？

三个对比组(均使用 Pi3X 模型,均直接信任 res["metric"] 作为尺度):
    * "model"     : 仅 RGB 输入                            (无任何 conditioning)
    * "model_int" : RGB + 标定内参                          (intrinsics conditioning, 无 depth)
    * "model_dc"  : RGB + 标定内参 + 传感器深度图           (intrinsics + depth conditioning)

之所以从"RGB + 内参"独立出一组，是因为之前的 use_depth / use_intrinsic 耦合在一起,
"只给内参不给深度"这一中间档无法测；解耦后(见 tools/run_normal_data_occ.py)才得以测试。

真值(GT)采用机器人里程计轨迹(LeRobot parquet 中的 observation.state)，
通过手眼标定(hand-eye)换算到相机坐标系下的相机中心轨迹。该真值与三个待比较的
变体都相互独立，因此可以无循环依赖地公正裁定。

每个变体都用"相机轨迹长度 vs 里程计真值长度"的相对误差来打分，
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

from L3ROcc.dataset.intern_nav_adapter import InternNavSequenceLoader
from L3ROcc.generater.intern_vln_env import InternNavDataGenerator
from L3ROcc.generater.normal_data_vln_env import SimpleVideoDataGenerator
from L3ROcc.utils import load_images_as_tensor


def _load_intrinsics_from_json(json_path):
    """Read a 3x3 ``head_camera_intrinsic`` from an info.json. Returns None on failure.

    与 tools/run_intern_nav_occ.py 中的同名函数保持一致:空串/非 .json/不存在/读取失败 均返回 None。
    """
    if (
        not json_path
        or not os.path.isfile(json_path)
        or not json_path.endswith(".json")
    ):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f_intr:
            data = json.load(f_intr)
        if "head_camera_intrinsic" not in data:
            return None
        return np.array(data["head_camera_intrinsic"], dtype=np.float32)
    except Exception as e:
        print(f"[Warning] Failed to read intrinsic JSON {json_path}: {e}")
        return None


def _load_intrinsics_from_parquet(parquet_path):
    """Read a 3x3 K from a parquet's ``observation.camera_intrinsic`` column (first row).

    InternData-N1 在每个 trajectory 的 parquet 中写入了 ``observation.camera_intrinsic``;
    其 meta/info.json 不再含 ``head_camera_intrinsic``。该列可能是 (3,3) 矩阵、长度 9
    的展开,或 3 个长度 3 的子列表 —— 任意一种都还原成 (3,3) np.float32;无法还原返回 None。
    """
    if not parquet_path or not os.path.isfile(parquet_path):
        return None
    try:
        df = pd.read_parquet(parquet_path, columns=["observation.camera_intrinsic"])
    except Exception:
        return None
    if "observation.camera_intrinsic" not in df.columns or len(df) == 0:
        return None
    raw = df["observation.camera_intrinsic"].tolist()[0]
    try:
        arr = np.asarray(raw, dtype=np.float32)
    except Exception:
        return None
    if arr.shape == (3, 3):
        return arr
    if arr.size == 9:
        return arr.reshape(3, 3)
    return None


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
    为(下采样后的)各帧构建真值相机中心轨迹。支持两种 parquet schema:

    A. lerobot rosbag —— ``observation.state`` 列(14 维):
       x,y,z, vx,vy,vz, q_w,q_x,q_y,q_z, roll,pitch,yaw,yaw_speed;state[0:3] 是身体中心,
       state[6:10] 是身体姿态。相机挂在身体上,身体系下偏移由 info.json 的
       ``head_camera_extrinsic.t_cam2gripper`` 给出(回退键 ``t_cam2robot``)。
       世界系相机中心 = R_world_body @ t_cam2gripper + p_body。
       info.json 缺失或缺该键时 GT 退化为身体中心轨迹(同语义旧分支)。

    B. InternData-N1 —— ``action`` 列(每行 4x4 SE(3) 矩阵,直接给出相机在世界系的位姿):
       平移列即相机轨迹;无独立身体中心,body_pos 取 cam_pos 同值(汇总 L_gt_robot 沿用旧字段名)。

    返回：
        cam_pos  : (n_keep, 3) 下采样各帧的真值相机中心
        body_pos : (n_keep, 3) 下采样各帧的身体中心 (N1 上等于 cam_pos)
    """
    df = pd.read_parquet(parquet_path)

    if "observation.state" in df.columns:
        state = np.stack(df["observation.state"].values).astype(np.float64)  # (T, 14)
        body_pos_full = state[:, 0:3]
        quat_full = state[:, 6:10]  # (w, x, y, z)

        # 手眼标定:优先 t_cam2gripper (lerobot_data_builder 实际写入的键名),
        # 回退到 t_cam2robot (历史命名,目前 vln_real_data 没有任何 rosbag 写这个键)。
        info = {}
        if info_json_path and os.path.isfile(info_json_path):
            with open(info_json_path, "r", encoding="utf-8") as f:
                info = json.load(f)
        ext = info.get("head_camera_extrinsic", {})
        t_raw = ext.get("t_cam2gripper", ext.get("t_cam2robot", None))
        if t_raw is None:
            print(f"[warn] {info_json_path} 中缺少 head_camera_extrinsic.t_cam2gripper/t_cam2robot"
                  f" (或文件不存在),GT 将退化为身体中心轨迹。")
            t_raw = [[0.0], [0.0], [0.0]]
        t_cam2body = np.asarray(t_raw, dtype=np.float64).reshape(3)

        R_wb = quat_wxyz_to_R(quat_full)                                       # (T, 3, 3)
        cam_pos_full = np.einsum("tij,j->ti", R_wb, t_cam2body) + body_pos_full  # (T, 3)
    elif "action" in df.columns:
        # InternData-N1: action 每行是 4x4 SE(3),取平移列作相机轨迹。
        actions = np.stack([
            np.asarray(a.tolist() if hasattr(a, "tolist") else a, dtype=np.float64)
            for a in df["action"].values
        ])  # (T, 4, 4)
        if actions.ndim != 3 or actions.shape[-2:] != (4, 4):
            raise KeyError(
                f"parquet {parquet_path} 的 action 列形状非 (T,4,4):{actions.shape}"
            )
        cam_pos_full = actions[:, :3, 3]
        body_pos_full = cam_pos_full.copy()
    else:
        raise KeyError(
            f"parquet {parquet_path} 既无 observation.state (lerobot rosbag),"
            f"也无 action (InternData-N1) —— 无法构建 GT 轨迹"
        )

    # 用与 RGB 帧相同的 interval 做下采样，再截断到 n_keep。
    cam_pos = cam_pos_full[0::interval][:n_keep]
    body_pos = body_pos_full[0::interval][:n_keep]
    return cam_pos, body_pos


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
            # K_rescaled 是 utils 透传的元数据,不是 Pi3X kwarg,splat 前剔除。
            cond_kwargs = {k: v for k, v in conditions.items() if k != "K_rescaled"}
            res = gen.model(imgs[None], **cond_kwargs)

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
# 任务发现:两种 engine 各自的 (rb, episode, out_dir, label) 列表构建
# --------------------------------------------------------------------------------------
def discover_episodes(rb):
    """返回某个 rosbag 目录下所有 episode 名(按 RGB 视频名)。"""
    d = os.path.join(rb, "videos", "chunk-000", "observation.images.RGB")
    if not os.path.isdir(d):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.lower().endswith(".mp4"))


def discover_rosbags(root):
    """返回 root 下所有 rosbag_* 子目录。"""
    if not os.path.isdir(root):
        return []
    return sorted(os.path.join(root, d) for d in os.listdir(root)
                  if d.startswith("rosbag_") and os.path.isdir(os.path.join(root, d)))


def _is_lerobot_rosbag(p):
    """单 rosbag 单元的判据:含 videos/chunk-000/observation.images.RGB 目录。"""
    return os.path.isdir(os.path.join(p, "videos", "chunk-000", "observation.images.RGB"))


def _build_jobs_normal(dataset_root, out_dir, target_episode):
    """run_normal_data_occ.py 风格的任务发现:仅识别 lerobot rosbag。

    若 dataset_root 自身即一个 rosbag,走单 rosbag 模式;否则当多 rosbag 父目录,
    遍历其下 rosbag_*。每个 rosbag 用 discover_episodes 列出 episode。
    返回 [(rb, ep, odir, label, rgb_path, depth_path), ...]。
    末两项固定为 None —— normal 引擎让 process_episode 按 lerobot 约定自行拼路径。
    """
    jobs = []
    if _is_lerobot_rosbag(dataset_root):
        rb = dataset_root
        eps = discover_episodes(rb) if target_episode == "all" else [target_episode]
        for ep in eps:
            jobs.append((rb, ep, os.path.join(out_dir, ep), ep, None, None))
    else:
        for rb in discover_rosbags(dataset_root):
            eps = discover_episodes(rb) if target_episode == "all" else [target_episode]
            for ep in eps:
                name = os.path.basename(rb.rstrip("/\\"))
                jobs.append((rb, ep, os.path.join(out_dir, name, ep), f"{name}__{ep}", None, None))
    return jobs


def _build_jobs_intern_nav(dataset_root, out_dir, target_episode):
    """run_intern_nav_occ.py 风格的任务发现:用 InternNavSequenceLoader 统一识别
    单 rosbag / 多 rosbag 父目录 / InternData-N1 三种布局。
    返回 [(rb, ep, odir, label, rgb_path, depth_path), ...];loader 已解析好的
    RGB/深度路径直接随 job 携带,避免下游按 lerobot 约定硬拼路径而漏认 N1 的
    observation.video.trajectory / observation.video.depth。
    若 loader 无任何轨迹,返回 None(由调用方报错)。
    """
    loader = InternNavSequenceLoader(dataset_root)
    if len(loader) == 0:
        return None
    jobs = []
    for i in range(len(loader)):
        rb = loader.trajectory_dirs[i]
        video_path = loader.trajectory_video_paths[i]
        depth_path = loader.trajectory_depth_paths[i]  # 可能为 None: 该 unit 无深度视频
        ep = os.path.splitext(os.path.basename(video_path))[0]
        if target_episode != "all" and ep != target_episode:
            continue
        rb_name = os.path.basename(rb.rstrip("/\\"))
        # 单 unit (rb 本身就是 dataset_root) 时,不再额外加一级 rb_name 子目录,保持与旧 --rosbag_dir 行为一致。
        if os.path.abspath(rb) == os.path.abspath(dataset_root):
            odir = os.path.join(out_dir, ep)
            label = ep
        else:
            odir = os.path.join(out_dir, rb_name, ep)
            label = f"{rb_name}__{ep}"
        jobs.append((rb, ep, odir, label, video_path, depth_path))
    return jobs


# --------------------------------------------------------------------------------------
# 单 episode 处理(复用已加载的 gen，模型只加载一次)
# --------------------------------------------------------------------------------------
def process_episode(gen, rb, episode, out_dir, label, args, do_plots, cli_intrinsics_np=None,
                    rgb_path=None, depth_path=None):
    """处理一个 episode：两次推理 + 计算指标 + 存盘(+可选出图)。返回指标 dict M 或 None。

    cli_intrinsics_np: 若非 None,作为最高优先级 K 覆盖该 rosbag 自身的 meta/info.json。
    rgb_path / depth_path: 调用端(通常是 intern_nav 引擎)预解析好的 RGB / 深度视频
        路径;为 None 时按 lerobot 约定 (observation.images.RGB / observation.images.depth)
        在该 rosbag 下硬拼路径。N1 布局 (observation.video.trajectory / .depth) 只有
        通过 InternNavSequenceLoader 传入路径才能被识别。
    """
    if rgb_path is None:
        rgb_path = os.path.join(rb, "videos", "chunk-000", "observation.images.RGB", f"{episode}.mp4")
    if depth_path is None:
        depth_path = os.path.join(rb, "videos", "chunk-000", "observation.images.depth", f"{episode}.mkv")
        if not os.path.exists(depth_path):
            depth_path = os.path.join(rb, "videos", "chunk-000", "observation.images.depth", f"{episode}.mp4")
    parquet_path = os.path.join(rb, "data", "chunk-000", f"{episode}.parquet")
    info_json_path = os.path.join(rb, "meta", "info.json")

    # depth_path 可能为 None: loader 没在该 unit 找到深度视频。校验时跳过 None;
    # 后续 load_images_as_tensor 会拿到 None/空串,conditions["depths"] 自然为 None,
    # 再由下面的 "[skip] ... 传感器深度加载失败" 分支正常退出。
    # info_json_path 也允许不存在: InternData-N1 单元可能没有 meta/info.json,
    # 此时内参从 parquet 取(见下方解析),外参 (load_gt_camera_positions) 自然取不到 —
    # 那一步只在确实使用 GT 的代码路径里才报错,这里不预先拦截。
    for p in [rgb_path, depth_path, parquet_path]:
        if p is None:
            continue
        if not os.path.exists(p):
            print(f"[skip] {label}: 缺少文件 {p}")
            return None

    os.makedirs(out_dir, exist_ok=True)

    # 内参优先级:
    #   CLI --condit_intr_path JSON
    #   > 当前 trajectory parquet 的 observation.camera_intrinsic 列(InternData-N1 写在这里)
    #   > 当前 rosbag 自身 meta/info.json 的 head_camera_intrinsic (lerobot rosbag 路径)
    #   > None — 让 Pi3X 自己反算 K (三变体中 int/dc 退化为无 K 条件,等价于 RGB-only)
    if cli_intrinsics_np is not None:
        intr_np = cli_intrinsics_np
        print(f"[{label}] 使用 CLI --condit_intr_path 的内参 (覆盖 {os.path.basename(rb)} 自身)。")
    else:
        intr_np = _load_intrinsics_from_parquet(parquet_path)
        if intr_np is not None:
            print(f"[{label}] 使用 parquet observation.camera_intrinsic 内参。")
        else:
            intr_np = _load_intrinsics_from_json(info_json_path)
            if intr_np is not None:
                print(f"[{label}] 使用 meta/info.json head_camera_intrinsic 内参。")
            else:
                print(f"[{label}] 未找到内参 (CLI/parquet/info.json 均无),由 Pi3X 自行反算。")

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

    # 构造三组 conditions:
    # - None              -> 仅 RGB (model)
    # - {intrinsics 单独}  -> RGB + 标定内参 (model_int)
    # - 完整 conditions    -> RGB + 内参 + 深度 (model_dc)
    # K_rescaled 是 load_images_as_tensor 透传的元数据(非 Pi3X kwarg),run_pi3x 内部已过滤。
    conditions_int = {k: v for k, v in conditions.items() if k != "depths"}
    conditions_int["depths"] = None

    print("[run] Pi3X RGB-only ...")
    rgb = run_pi3x(gen, imgs, conditions=None)
    print("[run] Pi3X RGB + intrinsic ...")
    intr = run_pi3x(gen, imgs, conditions=conditions_int)
    print("[run] Pi3X RGB + intrinsic + depth ...")
    dc = run_pi3x(gen, imgs, conditions=conditions)

    gt_cam, gt_body = load_gt_camera_positions(parquet_path, info_json_path, interval, N)
    n = min(N, len(gt_cam))
    gt_cam, gt_body = gt_cam[:n], gt_body[:n]
    rgb_pos = rgb["cam_pos"][:n]
    intr_pos = intr["cam_pos"][:n]
    dc_pos = dc["cam_pos"][:n]

    L_gt = path_length(gt_cam)
    L_model = path_length(rgb_pos)
    L_model_int = path_length(intr_pos)
    L_model_dc = path_length(dc_pos)

    # 三个变体各自的 Umeyama 理想尺度 c_gt_*(把该变体轨迹对齐到 GT 所需的最优系数)。
    # c_gt_rgb 保留旧名,作为汇总图与历史 metrics 字段兼容用。
    c_gt_rgb = umeyama_scale(rgb_pos, gt_cam)
    if not np.isfinite(c_gt_rgb):
        c_gt_rgb = L_gt / max(L_model, 1e-9)
    c_gt_int = umeyama_scale(intr_pos, gt_cam)
    if not np.isfinite(c_gt_int):
        c_gt_int = L_gt / max(L_model_int, 1e-9)
    c_gt_dc = umeyama_scale(dc_pos, gt_cam)
    if not np.isfinite(c_gt_dc):
        c_gt_dc = L_gt / max(L_model_dc, 1e-9)

    # 三个变体的"传感器深度 / 预测深度"中位比 —— 作为诊断量(metric 头有没有偏差)。
    s_pf_rgb, s_depth_rgb, _ = depth_scale_ratios(rgb["pred_depth"][:n], sensor_d[:n], rgb["conf"][:n],
                                                  args.conf_thr, args.dmin, args.dmax)
    s_pf_int, s_depth_int, _ = depth_scale_ratios(intr["pred_depth"][:n], sensor_d[:n], intr["conf"][:n],
                                                  args.conf_thr, args.dmin, args.dmax)
    s_pf_dc, s_depth_dc, _ = depth_scale_ratios(dc["pred_depth"][:n], sensor_d[:n], dc["conf"][:n],
                                                args.conf_thr, args.dmin, args.dmax)

    L_gt_safe = max(L_gt, 1e-9)
    e_model = abs(L_model - L_gt) / L_gt_safe
    e_model_int = abs(L_model_int - L_gt) / L_gt_safe
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
        "metric_int": intr["metric"],
        "metric_dc": dc["metric"],
        # 诊断:传感器深度 / 预测深度的中位比;模型若 metric 完美则该值 ≈ 1.0
        "s_depth_rgb": s_depth_rgb,
        "s_depth_int": s_depth_int,
        "s_depth_dc": s_depth_dc,
        # 三个变体各自的理想尺度
        "c_gt_rgb": float(c_gt_rgb),
        "c_gt_int": float(c_gt_int),
        "c_gt_dc": float(c_gt_dc),
        "L_gt": L_gt,
        "L_gt_robot": path_length(gt_body),  # 字段名沿用旧 schema; 实际语义是身体中心轨迹长度 (Unitree body frame)
        "L_model": L_model,
        "L_model_int": L_model_int,
        "L_model_dc": L_model_dc,
        # 主评分:三变体轨迹长度相对误差(对 GT)
        "e_model": e_model,
        "e_model_int": e_model_int,
        "e_model_dc": e_model_dc,
        # 纯尺度误差:metric 头与各自理想尺度的偏离(剥离轨迹形状误差)
        "scale_err_model": abs(1.0 - c_gt_rgb) / c_gt_rgb,
        "scale_err_model_int": abs(1.0 - c_gt_int) / c_gt_int,
        "scale_err_model_dc": abs(1.0 - c_gt_dc) / c_gt_dc,
        # 逐帧深度比(诊断)
        "s_per_frame_rgb": s_pf_rgb.tolist(),
        "s_per_frame_int": s_pf_int.tolist(),
        "s_per_frame_dc": s_pf_dc.tolist(),
        # 累计轨迹长度
        "cum_gt": cum(gt_cam).tolist(),
        "cum_model": cum(rgb_pos).tolist(),
        "cum_model_int": cum(intr_pos).tolist(),
        "cum_model_dc": cum(dc_pos).tolist(),
    }

    print(f"GT path {L_gt:.4f} m (robot {M['L_gt_robot']:.4f}) | metric RGB={rgb['metric']:.3f} "
          f"int={intr['metric']:.3f} dc={dc['metric']:.3f}"
          f" | c_gt rgb={c_gt_rgb:.3f} int={c_gt_int:.3f} dc={c_gt_dc:.3f}")
    print(f"[model {e_model*100:5.2f}%] [model_int {e_model_int*100:5.2f}%] [model_dc {e_model_dc*100:5.2f}%]"
          + ("   (skipped: 近静止)" if skipped else ""))

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(M, f, indent=2)

    # 仅在需要逐集出图、且非近静止时，保存绘图数据并渲染该集的 6 张图。
    if do_plots and not skipped:
        np.savez_compressed(
            os.path.join(out_dir, "plotdata.npz"),
            frame_idx=np.arange(n) * interval,
            pred_d_rgb=rgb["pred_depth"][:n].astype(np.float32),
            pred_d_int=intr["pred_depth"][:n].astype(np.float32),
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
    keys = ["episode", "n_frames", "L_gt",
            "L_model", "L_model_int", "L_model_dc",
            "metric_rgb", "metric_int", "metric_dc",
            "s_depth_rgb", "s_depth_int", "s_depth_dc",
            "c_gt_rgb", "c_gt_int", "c_gt_dc",
            "e_model", "e_model_int", "e_model_dc",
            "scale_err_model", "scale_err_model_int", "scale_err_model_dc"]
    rows = [{k: M.get(k) for k in keys} for M in results]

    def agg(key):
        vals = [r[key] for r in rows if r[key] is not None and np.isfinite(r[key])]
        if not vals:
            return None
        return {"mean": float(np.mean(vals)), "median": float(np.median(vals)), "std": float(np.std(vals))}

    variants = ("model", "model_int", "model_dc")
    win_counts = {v: 0 for v in variants}
    for M in results:
        w = min(variants, key=lambda k: M[f"e_{k}"])
        win_counts[w] += 1

    agg_keys = (
        # 主评分(轨迹长度误差)
        "e_model", "e_model_int", "e_model_dc",
        # 纯尺度误差
        "scale_err_model", "scale_err_model_int", "scale_err_model_dc",
        # 诊断量
        "s_depth_rgb", "s_depth_int", "s_depth_dc",
        "c_gt_rgb", "c_gt_int", "c_gt_dc",
    )

    summary = {
        "n_episodes": len(rows),
        "win_counts": win_counts,
        "agg": {k: agg(k) for k in agg_keys},
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
    # 新的统一入口:由 InternNavSequenceLoader 自动识别单 rosbag / 多 rosbag 父目录 / InternData-N1。
    ap.add_argument("--dataset_root", type=str, default="",
                    help="数据集根目录。可直接指向单个 rosbag、含多个 rosbag_* 的父目录,"
                         "或 InternData-N1 嵌套布局。未指定时回落到 --input_root 或 --rosbag_dir。")
    # 向后兼容别名:沿用旧脚本/CI 的调用方式;若未给 --dataset_root,这两个等价于它。
    ap.add_argument("--rosbag_dir", type=str,
                    default=r"G:/vln_real_data/lerobot_data/20260601/rosbag_20260529_155555",
                    help="[deprecated alias] 单个 rosbag 目录;等价于 --dataset_root。") # 该rosbag目录已经过lerobot_data_builder.py预处理
    ap.add_argument("--input_root", type=str, default="",
                    help="[deprecated alias] 含多个 rosbag_* 的根目录;等价于 --dataset_root。")
    ap.add_argument("--episode", type=str, default="episode_000",
                    help="episode 名;用 'all' 处理 dataset_root 下全部 episode。")
    ap.add_argument("--out_dir", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_out","20260604"))
    ap.add_argument("--condit_intr_path", type=str, default="",
                    help="可选:CLI JSON 内参(含 head_camera_intrinsic),覆盖各 trajectory 自身的 meta/info.json。"
                         "与 tools/run_intern_nav_occ.py 的同名参数语义一致。")
    ap.add_argument("--engine", type=str, default="intern_nav",
                    choices=["normal", "intern_nav"],
                    help="数据处理流程选择:"
                         "'intern_nav' 走 tools/run_intern_nav_occ.py 风格 (InternNavSequenceLoader + "
                         "InternNavDataGenerator),识别单/多 rosbag 与 InternData-N1;"
                         "'normal' 走 tools/run_normal_data_occ.py 风格 (手写 rosbag 发现 + "
                         "SimpleVideoDataGenerator),仅识别 lerobot rosbag。")
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

    # 解析数据根目录:--dataset_root > --input_root > --rosbag_dir。
    dataset_root = args.dataset_root or args.input_root or args.rosbag_dir
    if not dataset_root:
        print("[error] 未指定数据集根目录。请提供 --dataset_root / --input_root / --rosbag_dir 之一。")
        return
    if not os.path.isdir(dataset_root):
        print(f"[error] 数据集根目录不存在: {dataset_root}")
        return

    # CLI 内参覆盖(对所有 episode 生效);留空则在 process_episode 内读各自的 meta/info.json。
    cli_intrinsics_np = _load_intrinsics_from_json(args.condit_intr_path)
    if args.condit_intr_path and cli_intrinsics_np is None:
        print(f"[Warning] --condit_intr_path 提供但不可加载;将回退到每个 trajectory 自身的 meta/info.json: "
              f"{args.condit_intr_path}")

    # 按 engine 选择数据发现策略。
    if args.engine == "intern_nav":
        jobs = _build_jobs_intern_nav(dataset_root, args.out_dir, args.episode)
        if jobs is None:
            print(f"[error] InternNavSequenceLoader 在 {dataset_root} 下未发现有效轨迹。")
            return
    else:  # normal
        jobs = _build_jobs_normal(dataset_root, args.out_dir, args.episode)

    if not jobs:
        print(f"[error] engine={args.engine} 下未发现 episode='{args.episode}'。"
              f"用 --episode all 处理全部,或检查路径/名称。")
        return
    print(f"[plan] 共 {len(jobs)} 个 episode 待处理 (engine={args.engine})。")

    # 单集默认出逐集图；批量默认只出汇总图(除非 --per_episode_plots)。
    do_plots = args.per_episode_plots or (len(jobs) == 1)

    # 模型只加载一次。
    config_path = os.path.join(_PROJECT_ROOT, "L3ROcc", "configs", "config.yaml")
    model_dir = os.path.join(_PROJECT_ROOT, "ckpt")
    os.makedirs(args.out_dir, exist_ok=True)
    gen_cls = InternNavDataGenerator if args.engine == "intern_nav" else SimpleVideoDataGenerator
    gen = gen_cls(config_path, args.out_dir, model_dir, model_type="pi3x")
    # Windows 预加载的 state dict 已被 load_state_dict 复制进模型，释放它(~5.4GB)给推理腾内存。
    if sys.platform == "win32":
        globals().pop("_PI3X_SD", None)
        import gc
        gc.collect()
    print(f"[cfg] engine={args.engine}  generator={gen_cls.__name__}  "
          f"interval={gen.interval}  device={gen.device}  amp={gen.amp_dtype}")

    results = []
    for rb, ep, odir, label, rgb_path, depth_path in jobs:
        try:
            M = process_episode(gen, rb, ep, odir, label, args, do_plots, cli_intrinsics_np,
                                rgb_path=rgb_path, depth_path=depth_path)
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
            for m in ("e_model", "e_model_int", "e_model_dc"):
                a = s["agg"][m]
                if a:
                    print(f"  {m:12s} mean={a['mean']*100:5.2f}%  median={a['median']*100:5.2f}%")
            print(f"  win counts: {s['win_counts']}")
            render(args.out_dir, "summary")
        else:
            print("[summary] 无有效 episode，跳过汇总。")


if __name__ == "__main__":
    main()
