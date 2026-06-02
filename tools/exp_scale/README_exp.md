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

## 4. “实际距离”怎么算（真值锚点）

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

## 5. 运行方式

**服务器 / 大显存 GPU（推荐，全分辨率）：**
```bash
python tools/exp_scale_compare.py \
    --rosbag_dir G:/vln_real_data/lerobot_data/20260601/rosbag_20260529_155555 \
    --episode episode_000
```

**本机 6GB 显卡 / 无 GPU（CPU 回退，需降分辨率）：**
```bash
python tools/exp_scale_compare.py --episode episode_000 --cpu --pixel_limit 40000
```

**只重新出图（已有 metrics.json + plotdata.npz）：**
```bash
python tools/exp_plot.py tools/exp_out/episode_000
```

### 主要参数
| 参数 | 默认 | 说明 |
|---|---|---|
| `--rosbag_dir` | 见脚本 | rosbag 根目录 |
| `--episode` | `episode_001` | 要处理的 episode |
| `--cpu` | 关 | 强制 CPU（显存 <~8GB 时用；导入阶段即生效） |
| `--pixel_limit` | `255000` | 每帧最大像素数；CPU 上调小可加速（注意力显存随帧数×token 数平方增长） |
| `--conf_thr` | `0.1` | 有效像素的最小 Pi3X 置信度 |
| `--dmin` / `--dmax` | `0.25` / `6.0` | 可信传感器深度区间(米) |

---

## 6. 输出（写到 `tools/exp_out/<episode>/`）

- `metrics.json` —— 所有标量与逐帧/累计曲线数据
- `plotdata.npz` —— 绘图所需的深度图等数组
- 6 张图：
  1. `1_headline_error.png` —— 三种方法的轨迹长度误差（**主结论**）
  2. `2_correction_factor.png` —— 各修正系数 vs 理想系数 `c_gt`
  3. `3_per_frame_scale.png` —— 逐帧 `D_sensor/D_pred` 比值（看漂移与偏置）
  4. `4_cumulative_length.png` —— 累计轨迹长度 vs 真值
  5. `5_depth_scatter.png` —— 预测深度 vs 传感器深度散点（含 y=x）
  6. `6_depth_error_maps.png` —— 采样帧的深度误差热力图

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

## 7. 已有结果（episode_000，CPU @ 168×224，仅供参考）

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

## 8. Windows 环境注意事项（脚本里已内置规避）

1. **`Pi3X.from_pretrained` 段错误** —— 它先建模型再加载权重，破坏 safetensors mmap。
   规避：在导入模型模块**之前**先把权重加载到 CPU，再 `load_state_dict`。（该 bug 同样影响正式流水线）
2. **6GB 显卡装不下** —— fp32 模型≈5.44GB，且本机 torch 无 Flash-Attention，注意力会显存溢出。
   用 `--cpu`（内部设 `CUDA_VISIBLE_DEVICES=-1`）或上服务器。
3. **matplotlib/numpy MKL 崩溃**（`0xc06d007f`）—— 绘图放在独立子进程并设
   `MKL_THREADING_LAYER=SEQUENTIAL` / `KMP_DUPLICATE_LIB_OK=TRUE` / `OMP_NUM_THREADS=1`。
   同类 `np.linalg.svd` 崩溃也是 `umeyama_scale` 改用闭式公式（不做 SVD）的原因。
