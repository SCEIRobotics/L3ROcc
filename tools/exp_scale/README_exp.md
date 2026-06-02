# Pi3X 尺度对比实验说明（exp_scale_compare）

验证一个问题：对于 Pi3X 的三维重建，**哪个“绝对尺度(scale)”更准** ——
是模型自己预测的尺度（`metric` 头），还是用相机真实深度图推算出来的尺度？

- 主脚本：[`exp_scale_compare.py`](exp_scale_compare.py) —— 跑推理、算指标、存数据
- 绘图脚本：[`exp_plot.py`](exp_plot.py) —— 读数据出图（独立进程，避开 Windows 下 MKL 崩溃）

---

## 1. 核心思路

待比较的两个“尺度来源”：

| 名称 | 含义 | 修正系数 c |
|---|---|---|
| **model** | 信任 Pi3X 的 `metric` 头输出（仅 RGB 推理） | `c = 1.0` |
| **depth** | 用传感器深度对齐：`s_depth = median(D_sensor / D_pred)` | `c = s_depth` |
| **model_dc** | 把深度图作为**条件输入**再跑一次 Pi3X | （单独重建） |

> `model` 与 `depth` 共用**同一份仅 RGB 的几何**，只是换绝对尺度 —— 这样能干净地隔离“尺度”这一个变量。

**真值(GT)**：机器人里程计轨迹（parquet 的 `observation.state`），经手眼标定换算到相机坐标系。
它与「模型 metric 头」「深度传感器」都**相互独立**，因此可无循环依赖地公正裁判。

---

## 2. 输入数据

默认从一个 LeRobot rosbag 目录读取（`--rosbag_dir`），按 `--episode` 定位：

| 文件 | 内容 | 用途 |
|---|---|---|
| `videos/chunk-000/observation.images.RGB/<ep>.mp4` | RGB 视频 | Pi3X 输入 |
| `videos/chunk-000/observation.images.depth/<ep>.mkv` | gray16le 公制深度(mm) | 深度条件 / 算 depth scale |
| `data/chunk-000/<ep>.parquet` → `observation.state` | 里程计位姿(x,y,z + 四元数…) | **真值轨迹** |
| `meta/info.json` | `head_camera_intrinsic`、`head_camera_extrinsic.t_cam2gripper` | 内参、手眼标定 |

`observation.state` 共 14 维：`x,y,z, vx,vy,vz, q_w,q_x,q_y,q_z, roll,pitch,yaw,yaw_speed`

---

## 3. 处理流程

```
load_images_as_tensor          # 按 interval=10 下采样取帧、统一缩放到 pixel_limit
  ├─ RGB  → imgs (N,3,H,W)
  ├─ depth(gray16le ÷1000) → conditions["depths"] (N,H,W) 单位:米
  └─ intrinsics 按缩放比例同步调整
        │
run_pi3x(仅 RGB)   → 基准重建：camera_poses / local_points(Z=预测深度) / conf / metric
run_pi3x(带深度条件) → model_dc 变体
        │
depth_scale_ratios  → s_depth = median(D_sensor/D_pred)，仅统计有效像素
load_gt_camera_positions → 由里程计+手眼算出真值相机轨迹
        │
打分(轨迹长度误差) → metrics.json + plotdata.npz
        │
exp_plot.py(子进程) → 6 张图
```

**有效像素**定义（`depth_scale_ratios`）：预测/传感器深度均有限 ∧ 传感器深度 ∈ `[dmin,dmax]`
∧ 预测深度 > 0 ∧ 置信度 > `conf_thr`，逐帧取中位数，再汇总所有帧取全局中位数。

---

## 4. 三种 scale 详细计算流程（输入 → 计算 → 输出）

三者共用前置的两次推理结果：

```
rgb = run_pi3x(imgs)              # 仅 RGB
dc  = run_pi3x(imgs, conditions) # 深度条件(depths+intrinsics)
```

每次推理返回（取自 Pi3X 的 res，已搬到 CPU）：

| 字段 | 形状 | 含义 |
|---|---|---|
| `cam_pos` | (N,3) | 相机中心轨迹 = `camera_poses[:, :3, 3]`，**已乘 metric** → 公制(米) |
| `pred_depth` | (N,H,W) | 逐像素预测深度 = `local_points[...,2]`，**已乘 metric** → 公制(米) |
| `conf` | (N,H,W) | 置信度 = `sigmoid(conf)` |
| `metric` | 标量 | metric 头预测的全局绝对尺度 |

> 关键前提：在 [`pi3x.py`](../../third_party/pi3/pi3/models/pi3x.py) 的 `forward_head` 里，
> `local_points` 与 `camera_poses` 的平移**都乘了同一个 `metric` 标量**。
> 所以“相机轨迹”与“预测深度”天然处于同一尺度，对整套重建乘一个系数即可同时缩放点云与轨迹。

---

### 4.1 model —— 信任模型自身尺度

| | |
|---|---|
| **输入** | `rgb["cam_pos"]`（仅 RGB 重建的相机轨迹，已含 metric） |
| **计算** | 修正系数 `c = 1.0`（直接用模型输出，不做任何缩放）<br>`L_model = path_length(rgb_pos)` |
| **输出** | `metric_rgb = rgb["metric"]`<br>`L_model`，`e_model = |L_model − L_gt| / L_gt` |

本质：完全采信 Pi3X 的 `metric` 头从 RGB 推出的绝对尺度。

---

### 4.2 depth —— 用传感器深度推算尺度

| | |
|---|---|
| **输入** | `rgb["pred_depth"]`、`sensor_d`（传感器公制深度）、`rgb["conf"]` |
| **计算** | `depth_scale_ratios()`：在**有效像素**上逐像素求比值 `r = D_sensor / D_pred`<br>→ 逐帧中位数 `s_per_frame_rgb`<br>→ 全部帧汇总取全局中位数 **`s_depth`** |
| **应用** | 对仅 RGB 重建整体乘 `s_depth`；轨迹长度 `L_depth = s_depth × L_model` |
| **输出** | `s_depth`、`L_depth`、`e_depth = |s_depth·L_model − L_gt| / L_gt` |

有效像素：`D_pred、D_sensor 有限 ∧ D_sensor∈[dmin,dmax] ∧ D_pred>0 ∧ conf>conf_thr`，
单帧有效像素 ≥ 50 才计入。

> 因为 `pred_depth` 已含 metric，所以 `s_depth` 是**叠加在模型输出之上的“残差修正系数”**：
> 模型若已是完美公制，则 `s_depth ≈ 1`。本实验里 `s_depth≈0.95`，即模型深度比传感器约大 5%。

---

### 4.3 model_dc —— 深度作为条件输入

| | |
|---|---|
| **输入** | `imgs` + `conditions`（`depths`=传感器深度、`intrinsics`=缩放后内参） → 第二次推理 `dc` |
| **计算** | 模型在深度条件下**直接输出**带尺度的重建；`dc["cam_pos"]` 已是公制<br>`L_model_dc = path_length(dc_pos)` |
| **输出** | `metric_dc = dc["metric"]`、`L_model_dc`、`e_model_dc = |L_model_dc − L_gt| / L_gt` |

本质：让深度图在网络内部参与推断，由模型自己融合出尺度（而非事后乘系数）。

---

### 4.4 参照量与评分（三者共用）

```
c_gt = umeyama_scale(rgb_pos, gt_cam)   # 把仅 RGB 重建对齐到真值所需的"理想尺度"
scale_err_model = |1.0     − c_gt| / c_gt
scale_err_depth = |s_depth − c_gt| / c_gt
```

- `e_*`（轨迹长度相对误差，对真值）是**主评分**；越小越准。
- `scale_err_*`（修正系数 vs 理想系数 `c_gt`）剥离轨迹形状误差，只看纯尺度。
- 三者越接近 `c_gt`，说明该尺度越接近真值要求。

> 小结：`model` 与 `depth` 共用**同一份仅 RGB 几何**，只在“乘哪个系数”上不同（1.0 vs `s_depth`），
> 因此能干净隔离“尺度”这一个变量；`model_dc` 则是另起一份深度条件重建，用于对照“深度作为输入”是否更优。

---

## 5. “实际距离”怎么算（真值锚点）

完全**不依赖视觉/深度**，来自机器人本体里程计：

```python
# load_gt_camera_positions()
p_gripper = state[:, 0:3]                 # 夹爪世界坐标 x,y,z
quat      = state[:, 6:10]                # 四元数(w,x,y,z) → R_world_gripper
t_c2g     = info.json["...t_cam2gripper"] # 相机原点在夹爪坐标系下的位置(手眼标定)

# 夹爪轨迹 → 相机中心轨迹：
cam_pos = R_world_gripper @ t_c2g + p_gripper

# 真值距离 = 相机中心折线总长：
L_gt = Σ ‖cam_pos[i+1] − cam_pos[i]‖      # path_length()
```

- **重建侧距离** `L_model = Σ‖camera_poses 平移[i+1] − [i]‖`，再乘各自尺度。
- **误差** `= |c × L_model − L_gt| / L_gt`。
- `L_gt_gripper`（忽略手眼、直接用夹爪 xyz）作对照；若与 `L_gt` 接近即说明手眼换算正确。
- `umeyama_scale(rgb_pos, gt_cam)` 给出“理想尺度 `c_gt`”（重建对齐到真值所需的最优缩放），作参照。

---

## 6. 运行方式

> 可在项目根目录运行（脚本会自动向上定位项目根）。服务器(Linux)上自动用 bf16+Flash-Attention、
> 模型只加载一次；批量时跨 episode 复用同一份模型。

**服务器 — 单集（推荐，全分辨率 GPU）：**
```bash
python tools/exp_scale/exp_scale_compare.py --rosbag_dir <rosbag目录> --episode episode_000
```

**服务器 — 批量（一个 rosbag 的全部 episode）：**
```bash
python tools/exp_scale/exp_scale_compare.py --rosbag_dir <rosbag目录> --episode all
```

**服务器 — 批量（遍历多个 rosbag）：**
```bash
python tools/exp_scale/exp_scale_compare.py --input_root <含多个 rosbag_* 的根目录> --episode all
```

**本机 6GB 显卡 / 无 GPU（CPU 回退，需降分辨率）：**
```bash
python tools/exp_scale/exp_scale_compare.py --episode episode_000 --cpu --pixel_limit 40000
```

**只重新出图：** 单集 `exp_plot.py <episode目录> episode`；汇总 `exp_plot.py <exp_out根目录> summary`。

### 主要参数
| 参数 | 默认 | 说明 |
|---|---|---|
| `--rosbag_dir` | 见脚本 | 单个 rosbag 目录（未给 `--input_root` 时用） |
| `--input_root` | 空 | 含多个 `rosbag_*` 的根目录，遍历其下所有 rosbag |
| `--episode` | `episode_001` | episode 名；用 `all` 处理该 rosbag 下全部 episode |
| `--min_motion` | `0.3` | 真值相机轨迹 < 此值(米)的近静止集跳过，不计入汇总 |
| `--per_episode_plots` | 关 | 批量时也为每集出 6 图（默认仅单集出图，批量只出汇总图） |
| `--cpu` | 关 | 强制 CPU（显存 <~8GB 时用；导入阶段即生效） |
| `--pixel_limit` | `255000` | 每帧最大像素数；CPU 上调小可加速（注意力显存随帧数×token 数平方增长） |
| `--conf_thr` | `0.1` | 有效像素的最小 Pi3X 置信度 |
| `--dmin` / `--dmax` | `0.25` / `6.0` | 可信传感器深度区间(米) |

---

## 7. 输出

### 每个 episode（写到 `exp_out/<episode>/`，多 rosbag 时为 `exp_out/<rosbag>/<episode>/`）
- `metrics.json` —— 所有标量与逐帧/累计曲线数据（含 `skipped` 标志）
- `plotdata.npz` + 6 张图（仅在单集或 `--per_episode_plots` 且非近静止时生成）：
  1. `1_headline_error.png` —— 三种方法的轨迹长度误差（**主结论**）
  2. `2_correction_factor.png` —— 各修正系数 vs 理想系数 `c_gt`
  3. `3_per_frame_scale.png` —— 逐帧 `D_sensor/D_pred` 比值（看漂移与偏置）
  4. `4_cumulative_length.png` —— 累计轨迹长度 vs 真值
  5. `5_depth_scatter.png` —— 预测深度 vs 传感器深度散点（含 y=x）
  6. `6_depth_error_maps.png` —— 采样帧的深度误差热力图

### 跨 episode 汇总（批量时写到 `exp_out/`）
- `summary.json` —— `n_episodes`、各方法误差的 `mean/median/std`、各方法“最优次数” `win_counts`、逐集 `rows`
- `summary.csv` —— 每行一个 episode 的关键指标，便于表格查看
- `summary.png` —— 三面板：误差均值±标准差柱状图 / 最优次数 / 逐集 `s_depth` vs `c_gt` 散点（越靠 y=x 越准）

### metrics.json 关键字段
| 字段 | 含义 |
|---|---|
| `L_gt` / `L_gt_gripper` | 真值相机 / 夹爪轨迹长度(米) |
| `L_model` / `L_model_dc` / `L_depth` | 三种重建的轨迹长度 |
| `metric_rgb` / `metric_dc` | Pi3X metric 头标量 |
| `s_depth` | 深度推算的全局尺度 |
| `c_gt_rgb` | Umeyama 理想尺度 |
| `e_model` / `e_depth` / `e_model_dc` | 各方法的轨迹长度相对误差 |
| `scale_err_model` / `scale_err_depth` | 纯尺度误差（修正系数 vs 理想系数） |
| `s_per_frame_rgb` / `s_per_frame_dc` | 逐帧尺度比值 |

---

## 8. 已有结果（episode_000，CPU @ 168×224，仅供参考）

| 尺度来源 | 重建长度 | 轨迹长度误差 | 纯尺度误差 |
|---|---|---|---|
| model（RGB metric 头） | 1.79 m | 21.6% | 16.9% |
| **depth（RGB+传感器）** | 1.70 m | **15.5%** ✅ | 11.1% |
| model_dc（深度条件） | 1.70 m | 15.6% | — |

- 真值轨迹 **1.47 m**；模型自身尺度**偏大约 17~21%**。
- 用深度（后处理重缩放 **或** 作条件输入）都能降到约 15.5%，二者基本持平。
- **关键现象**（见图 3）：模型深度比传感器大约 5%（比值≈0.95，稳定无漂移），
  但轨迹真值要求的修正是 **0.855** —— 即**深度传感器与里程计本身对尺度也有约 11% 的分歧**。
  以里程计为准时 depth 更接近真值，但两者都未完全命中。

> ⚠️ 注意：上述为 6GB 显卡限制下的**低分辨率 CPU 结果**，模型 metric 头可能被低分辨率不公平地拖累。
> 定论请在服务器全分辨率下复跑（去掉 `--cpu`）。单一近似直线 episode，建议多 episode 复核。

---

## 9. 跨平台说明（Windows 规避已内置；Linux 走原生快路径）

1. **模型加载（已集中到 `L3ROcc/base.py:_load_pretrained_model`）** —— Linux 直接原生
   `from_pretrained`；Windows 上 `from_pretrained` 会段错误（先建模型再加载权重，破坏 safetensors
   mmap），故 Windows 分支先把权重加载到 CPU、再惰性导入模型并 `load_state_dict`。数据生成与本验证
   脚本共用此逻辑——验证脚本**不再**自带 monkeypatch。
2. **显存** —— fp32 模型≈5.44GB。服务器大显存直接 GPU 全分辨率（自动 bf16+Flash）；本机 6GB 用
   `--cpu`（内部设 `CUDA_VISIBLE_DEVICES=-1`）+ 调小 `--pixel_limit`。
3. **绘图（`exp_plot.py`）** —— Linux 进程内直接调用；Windows 必须独立子进程并设
   `MKL_THREADING_LAYER=SEQUENTIAL` / `KMP_DUPLICATE_LIB_OK=TRUE` / `OMP_NUM_THREADS=1`，
   否则 torch 的 MKL 驻留同进程会让 matplotlib 崩溃（`0xc06d007f`）。同类 `np.linalg.svd` 崩溃也是
   `umeyama_scale` 改用闭式公式（不做 SVD）的原因。
4. **数据生成（`tools/run_normal_data_occ.py`）** —— `--video_path` 默认空 → 走 `--input_root`
   批量；要单文件时显式传 `--video_path`。服务器上用 CLI 覆盖 `--input_root`/`--output_root` 为 Linux 路径。
