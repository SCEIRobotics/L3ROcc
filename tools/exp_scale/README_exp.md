# Pi3X 尺度 & 条件输入对比实验（exp_scale_compare）

验证：**在 Pi3X 的多模态条件输入中加入哪种，能让 metric 头给出的绝对尺度最准？**

- 主脚本：[`exp_scale_compare.py`](exp_scale_compare.py) —— 推理、算指标、存数据
- 绘图脚本：[`exp_plot.py`](exp_plot.py) —— 读数据出图（独立进程）

---

## 1. 核心思路

三个对比变体（均使用 Pi3X，均直接信任 `res["metric"]` 作为绝对尺度）：

| 变体名 | RGB | 标定内参 | 深度图 | 备注 |
|---|---|---|---|---|
| **`model`**     | ✓ | ✗ | ✗ | RGB-only 基线 |
| **`model_int`** | ✓ | ✓ | ✗ | 标定内参作为 conditioning |
| **`model_dc`**  | ✓ | ✓ | ✓ | 传感器深度 + 内参作为 conditioning |

三变体各自独立推理，用同一套指标与同一份 GT 比较。

**真值（GT）**：机器人里程计轨迹（parquet `observation.state`），经手眼标定换算到相机坐标系，与三变体相互独立。

---

## 2. 输入数据

脚本通过 `InternNavSequenceLoader`（`intern_nav` engine，默认）或手写 `discover_*`（`normal` engine）发现轨迹，支持两种 parquet schema：

### A. lerobot rosbag（Unitree 真实数据）

| 文件 | 内容 | 用途 |
|---|---|---|
| `videos/chunk-000/observation.images.RGB/<ep>.mp4` | RGB 视频 | Pi3X 输入 |
| `videos/chunk-000/observation.images.depth/<ep>.mkv` | gray16le 公制深度（mm） | `model_dc` 深度条件 |
| `data/chunk-000/<ep>.parquet` → `observation.state` | 里程计位姿（x,y,z + 四元数） | **真值轨迹** |
| `meta/info.json` | `head_camera_intrinsic`、`head_camera_extrinsic.t_cam2gripper` | 标定内参 / 手眼标定（脚本兼容旧名 `t_cam2robot`） |

`observation.state` 共 14 维：`x,y,z, vx,vy,vz, q_w,q_x,q_y,q_z, roll,pitch,yaw,yaw_speed`；`state[0:3]` 是身体中心，经 `t_cam2gripper` 换算才是相机中心。

### B. InternData-N1（合成，3D-Front + ZED）

| 文件 | 内容 | 用途 |
|---|---|---|
| `videos/chunk-000/observation.video.trajectory/*.mp4` 或 `observation.images.RGB/*.mp4` | RGB 视频 | Pi3X 输入 |
| `videos/chunk-000/observation.video.depth/*.mkv` 或 `observation.images.depth/*` | 深度视频 | `model_dc` 条件 |
| `data/chunk-000/<ep>.parquet` → `action` 列 | 每行 4×4 SE(3) 矩阵，相机在世界系的真位姿 | **真值轨迹（无里程计噪声）** |
| `data/chunk-000/<ep>.parquet` → `observation.camera_intrinsic` 列 | 每行 (3,3) 矩阵 | 标定内参（无独立 `meta/info.json`） |

N1 无"身体中心"概念，GT 直接取 `action` 平移列（`L_gt_robot` 与 `L_gt` 数值相等）。

---

## 3. 处理流程

```
load_images_as_tensor   # interval=10 下采样、统一缩放到 pixel_limit
  ├─ RGB    → imgs                                       (N,3,H,W)
  ├─ depth (gray16le ÷1000)                              → conditions["depths"]      (1,N,H,W) 米
  ├─ intrinsics 按 resize 比例同步缩放                    → conditions["intrinsics"]  (1,N,3,3)
  └─ K_rescaled (numpy, 元数据, run_pi3x 前自动过滤)      → conditions["K_rescaled"]  (3,3)
        │
run_pi3x(None)                              # model       (仅 RGB)
run_pi3x({intrinsics, depths=None})         # model_int   (RGB + 标定内参)
run_pi3x(完整 conditions)                    # model_dc    (RGB + 内参 + 深度)
        │
load_gt_camera_positions   →  GT 相机轨迹（里程计 + 手眼）
        │
轨迹长度误差打分 + 诊断量 → metrics.json + plotdata.npz
        │
exp_plot.py（子进程） → 6 张图（单集）/ summary.png（汇总）
```

---

## 4. 指标定义

每次推理返回（取自 Pi3X 的 `res`）：

| 字段 | 形状 | 含义 |
|---|---|---|
| `cam_pos` | (N,3) | 相机中心轨迹，已乘 `metric`，单位：米 |
| `pred_depth` | (N,H,W) | 逐像素预测深度，已乘 `metric`，单位：米 |
| `metric` | 标量 | Pi3X 内部"归一化 → 米"增益（≈ 0.30，跨场景/数据集固定，不是 per-scene 尺度估计） |

评分与诊断量（三变体共用，`<v>` 为 `rgb / int / dc`）：

```
e_<v>          = |L_<v> − L_gt| / L_gt           # 主评分：轨迹长度相对误差（越小越好）
c_gt_<v>       = umeyama_scale(<v>_pos, gt_cam)   # 对齐到 GT 的理想尺度（理想 ≈ 1.0）
scale_err_<v>  = |1.0 − c_gt_<v>| / c_gt_<v>     # 纯尺度误差（剥离轨迹形状误差）
s_depth_<v>    = median(D_sensor / D_pred_<v>)    # 诊断：模型深度 vs 传感器深度（理想 ≈ 1.0）
```

> **只看 `c_gt_*` / `e_model_*` / `scale_err_*`，不要把 `metric_*`（≈ 0.30）当 scale 结论读**——后者是 Pi3X 训练分布的固定内部增益，与场景无关。

---

## 5. GT 轨迹计算

`load_gt_camera_positions()` 根据 parquet 中的列名自动切换两种 schema，最终均返回与 RGB 帧对齐（同 `interval` 下采样）的相机中心序列 `cam_pos (n_keep, 3)`。

### A. lerobot rosbag（Unitree 真实数据）

数据来源：`observation.state` 列（14 维），完全不依赖视觉/深度，来自机器人本体里程计。

```python
# parquet: data/chunk-000/<ep>.parquet → "observation.state"
state      = df["observation.state"]          # (T, 14)
p_body     = state[:, 0:3]                    # 身体中心世界坐标 x,y,z（Unitree sportmodestate.position）
quat       = state[:, 6:10]                   # 身体姿态四元数 (w,x,y,z) → R_world_body
t_cam2body = info.json["head_camera_extrinsic.t_cam2gripper"]   # 身体系下相机原点（手眼标定，兼容旧名 t_cam2robot）

R_wb    = quat_wxyz_to_R(quat)                           # (T, 3, 3)
cam_pos = np.einsum("tij,j->ti", R_wb, t_cam2body) + p_body  # (T, 3) 世界系相机中心
L_gt    = Σ ‖cam_pos[i+1] − cam_pos[i]‖                 # 相机中心折线总长
```

- `t_cam2gripper` 沿用 OpenCV `calibrateHandEye` 命名，在四足/移动机器人语境下即身体中心偏移。
- `info.json` 缺失或缺该键时，GT 退化为身体中心轨迹（打 warn）。
- `L_gt_robot`（直接用 `p_body` 折线长，忽略手眼）作对照；若与 `L_gt` 接近说明手眼换算正确。

### B. InternData-N1（合成数据）

数据来源：`action` 列，每行是渲染管线写入的相机在世界系的真实位姿（4×4 SE(3) 矩阵），无里程计噪声，无独立身体中心。

```python
# parquet: data/chunk-000/<ep>.parquet → "action"
actions = np.stack(df["action"].values)   # (T, 4, 4) SE(3) 矩阵
cam_pos = actions[:, :3, 3]              # 直接取平移列，即世界系相机中心
# body_pos = cam_pos（N1 无独立身体中心，L_gt_robot 与 L_gt 数值相等）
L_gt = Σ ‖cam_pos[i+1] − cam_pos[i]‖
```

---

## 6. 运行方式

> 在项目根目录运行，模型只加载一次，批量时跨 episode 复用。

```bash
# 单集
python tools/exp_scale/exp_scale_compare.py --dataset_root <rosbag目录> --episode episode_000

# 批量
python tools/exp_scale/exp_scale_compare.py --dataset_root <rosbag或父目录或N1 group> --episode all
```

**只重新出图**：单集 `exp_plot.py <episode目录> episode`；汇总 `exp_plot.py <exp_out根目录> summary`。

### 主要参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataset_root` | 见脚本 | 统一入口：单个 rosbag / 含多个 `rosbag_*` 的父目录 / InternData-N1 嵌套布局，自动识别 |
| `--engine` | `intern_nav` | `intern_nav` 走 InternNavSequenceLoader；`normal` 走 discover_* + SimpleVideoDataGenerator |
| `--episode` | `episode_001` | episode 名；`all` 处理全部 episode |
| `--condit_intr_path` | 空 | CLI JSON 内参覆盖 |
| `--min_motion` | `0.3` | GT 相机轨迹 < 此值（米）的近静止集跳过 |
| `--per_episode_plots` | 关 | 批量时也为每集出 6 图 |
| `--cpu` | 关 | 强制 CPU（显存 < ~8 GB 时用） |
| `--pixel_limit` | `255000` | 每帧最大像素数；CPU 上调小可加速 |
| `--conf_thr` | `0.1` | 有效像素的最小 Pi3X 置信度（仅用于 `s_depth_*` 诊断） |
| `--dmin` / `--dmax` | `0.25` / `6.0` | 可信传感器深度区间（米，仅用于 `s_depth_*` 诊断） |

---

## 7. 输出

### 每个 episode（写到 `exp_out/<episode>/`）

- `metrics.json` —— 所有标量与逐帧/累计曲线数据（含 `skipped` 标志）
- `plotdata.npz` + 6 张图（单集或 `--per_episode_plots` 时生成）：
  1. `1_headline_error.png` —— 三变体轨迹长度误差（**主结论**）
  2. `2_correction_factor.png` —— 三变体 metric（c=1）vs 各自理想 `c_gt`
  3. `3_per_frame_scale.png` —— 三变体逐帧 `D_sensor/D_pred` 比值
  4. `4_cumulative_length.png` —— 三变体累计轨迹长度 vs 真值
  5. `5_depth_scatter.png` —— 三变体预测深度 vs 传感器深度散点
  6. `6_depth_error_maps.png` —— 采样帧深度误差热力图（基于 RGB-only）

### 跨 episode 汇总（批量时写到 `exp_out/`）

- `summary.json` —— `n_episodes`、各变体误差的 `mean/median/std`、`win_counts`、逐集 `rows`
- `summary.csv` —— 每行一个 episode 的关键指标
- `summary.png` —— 三面板：误差均值±标准差柱状图 / 最优次数 / `c_gt_*` 分布

### `metrics.json` 关键字段

| 字段 | 含义 |
|---|---|
| `L_gt` / `L_gt_robot` | 真值相机 / 机器本体轨迹长度（米） |
| `L_model` / `L_model_int` / `L_model_dc` | 三变体轨迹长度 |
| `metric_rgb` / `metric_int` / `metric_dc` | Pi3X 内部增益标量（≈ 0.30，诊断用） |
| `c_gt_rgb` / `c_gt_int` / `c_gt_dc` | Umeyama 理想尺度（≈ 1.0 为准） |
| `e_model` / `e_model_int` / `e_model_dc` | 轨迹长度相对误差（**主评分**） |
| `scale_err_model` / `_int` / `_dc` | 纯尺度误差（`c_gt` 衍生） |
| `s_depth_rgb` / `_int` / `_dc` | 预测深度 vs 传感器深度全局中位比（诊断） |
| `s_per_frame_rgb` / `_int` / `_dc` | 逐帧深度比（诊断） |
| `cum_gt` / `cum_model` / `cum_model_int` / `cum_model_dc` | 累计轨迹长度 |

---

## 8. 实验结果

### 8.1 Unitree rosbag（32 集，真实 VLN 数据）

GT 来自 Unitree 腿部里程计 + IMU + 手眼 `t_cam2gripper`。

| Variant | Mean `e_*` | Median `e_*` | Median `c_gt_*` | Median `metric_*` | Win count |
|---|---:|---:|---:|---:|---:|
| `model` — RGB only | 13.34% | 9.64% | 0.93 | 0.297 | 5 |
| `model_int` — RGB + K | 12.99% | 9.29% | 0.93 | 0.296 | 10 |
| `model_dc` — RGB + K + depth | **12.31%** | **9.27%** | 0.94 | 0.297 | **17** |

**主要观察**：
- 三变体差异极小（mean < 1%，median < 0.4%）；`metric_*` 对 conditioning 几乎不响应。
- `c_gt_*` ≈ 0.93：Pi3X 在真实场景下系统性高估轨迹长度 ~7%。
- Mean ≫ Median：存在少量高误差长尾（近静止集/大偏航段被放大）。

### 8.2 InternData-N1 / 3D-Front（21 集，干净合成 GT）

GT 直接取 parquet `action` 列的 4×4 SE(3) 矩阵平移分量（无里程计噪声）。

| Variant | Mean `e_*` | Median `e_*` | Median `c_gt_*` | Median `metric_*` |
|---|---:|---:|---:|---:|
| `model` — RGB only | 2.58% | 2.24% | 1.003 | 0.308 |
| `model_int` — RGB + K | 2.58% | 2.24% | 1.003 | 0.308 |
| `model_dc` — RGB + K + depth | 4.11% | 2.94% | 1.005 | 0.306 |

**主要观察**：
- 干净 GT 下 Pi3X 尺度残差仅 ~2-3%（`c_gt` ≈ 1.00）。
- `model` 与 `model_int` 结果完全相同，说明 N1 路径下内参未真正传入 Pi3X（待查）。
- `model_dc` 略差于 `model`，因 N1 深度视频编码与 gray16le 解码不匹配，深度数值不可信，污染了 conditioning。N1 上的 `s_depth_*`（≈ 13）与 `model_dc` 结果当前不可读。
- `metric_*` 跨数据集均 ≈ 0.30，与变体无关，印证其为固定内部增益。

### 8.3 综合结论

| GT 质量 | 数据 | n | Median `e_model` | 解读 |
|---|---|---:|---:|---|
| 干净合成 GT | InternData-N1 / 3D-Front | 21 | **2.2%** | Pi3X 本身尺度残差 ≈ 2-3% |
| 真实噪声 GT | Unitree rosbag | 32 | 9.6% | 模型 ~3% + GT 噪声 ~5-7% 叠加 |

**可用**（精度 3-10%）：`cam_pos` 路径长度与几何形状、`pred_depth`、`c_gt_*` / `e_model_*`。

**不可直接读为 scale 结论**：`metric_*` 标量（固定 ≈ 0.30，与场景无关）；"加内参 conditioning 有效"的结论（两数据集上差异均 < 0.4% 或为零，尚无统计支撑）。

**实操指南**：把 `cam_pos / pred_depth` 当 **3-5% 精度的弱公制**，需更高精度时再事后做一次尺度对齐（用已知距离/物体/独立 GT）；报告只引用 `c_gt_*` 或 `e_model_*`，不引用 `metric_*`。