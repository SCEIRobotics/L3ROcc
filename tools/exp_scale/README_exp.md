# Pi3X Scale & Conditioning Study — `exp_scale_compare`

> 在三种输入组合下评估 Pi3X 的 metric 头给出的绝对尺度准确性：
> 1) 仅 RGB，2) RGB + 标定内参，3) RGB + 标定内参 + 传感器深度图。
> 真值来自机器人里程计轨迹（与三个变体均独立）。

---

# Pi3X 尺度 & 条件输入对比实验（exp_scale_compare）

验证一个问题：**在 Pi3X 的多模态条件输入中加入哪种,能让 metric 头给出的"绝对尺度(scale)"最准？**

- 主脚本：[`exp_scale_compare.py`](exp_scale_compare.py) —— 跑推理、算指标、存数据
- 绘图脚本：[`exp_plot.py`](exp_plot.py) —— 读数据出图(独立进程,避开 Windows 下 MKL 崩溃)

---

## 1. 核心思路

三个对比变体(均使用 Pi3X 模型,均直接信任 `res["metric"]` 作为绝对尺度):

| 变体名 | RGB | 标定内参 | 深度图 | 备注 |
|---|---|---|---|---|
| **`model`**     | ✓ | ✗ | ✗ | 旧"RGB-only"基线;模型完全自己推断 |
| **`model_int`** | ✓ | ✓ | ✗ | 只把标定内参作为 conditioning |
| **`model_dc`**  | ✓ | ✓ | ✓ | 把传感器深度也作为 conditioning(完整多模态) |

> 三个变体都是**独立的 Pi3X 重建**(分别推理一次),互不共享几何;
> 评分用同一套指标(轨迹长度相对误差),与同一份 GT 比较。

**真值(GT)**：机器人里程计轨迹(`parquet` 的 `observation.state`),经手眼标定换算到相机坐标系。
它与三个变体的 metric 头都**相互独立**,可以无循环依赖地公正裁判。

---

## 2. 输入数据

默认从一个 LeRobot rosbag 目录读取(`--rosbag_dir`),按 `--episode` 定位：

| 文件 | 内容 | 用途 |
|---|---|---|
| `videos/chunk-000/observation.images.RGB/<ep>.mp4` | RGB 视频 | Pi3X 输入 |
| `videos/chunk-000/observation.images.depth/<ep>.mkv` | gray16le 公制深度(mm) | `model_dc` 的深度条件 / 诊断量 `s_depth_*` |
| `data/chunk-000/<ep>.parquet` → `observation.state` | 里程计位姿(x,y,z + 四元数…) | **真值轨迹** |
| `meta/info.json` | `head_camera_intrinsic`、`head_camera_extrinsic.t_cam2gripper` | 标定内参 / 手眼标定(`gripper` 沿用 OpenCV calibrateHandEye 命名,在四足/移动机器人语境下即身体中心;脚本回退兼容旧名 `t_cam2robot`) |

`observation.state` 共 14 维：`x,y,z, vx,vy,vz, q_w,q_x,q_y,q_z, roll,pitch,yaw,yaw_speed`

---

## 3. 处理流程

```
load_images_as_tensor   # 按 interval=10 下采样取帧、统一缩放到 pixel_limit
  ├─ RGB    → imgs                                       (N,3,H,W)
  ├─ depth (gray16le ÷1000)                              → conditions["depths"]      (1,N,H,W) 米
  ├─ intrinsics 按 resize 比例同步缩放                    → conditions["intrinsics"]  (1,N,3,3)
  └─ K_rescaled (numpy, 元数据,运行 Pi3X 前 pop)          → conditions["K_rescaled"]  (3,3)
        │
run_pi3x(None)                              # model       (仅 RGB)
run_pi3x({intrinsics 单独, depths=None})    # model_int   (RGB + 标定内参)
run_pi3x(完整 conditions)                    # model_dc    (RGB + 内参 + 深度)
        │
load_gt_camera_positions   →  GT 相机轨迹(里程计 + 手眼)
        │
轨迹长度误差打分 + 诊断量(s_depth_*、c_gt_*) → metrics.json + plotdata.npz
        │
exp_plot.py(子进程) → 6 张图(单集) / summary.png(汇总)
```

`s_depth_*`(诊断,可选)：每个变体的预测深度对传感器深度的中位比 `median(D_sensor / D_pred)`,
取在有效像素上。理想 metric 模型该值应 ≈ 1.0。

---

## 4. 三变体计算细节(输入 → 计算 → 输出)

每次推理返回(取自 Pi3X 的 `res`)：

| 字段 | 形状 | 含义 |
|---|---|---|
| `cam_pos` | (N,3) | 相机中心轨迹 = `camera_poses[:, :3, 3]`,**已乘 metric** → 公制(米) |
| `pred_depth` | (N,H,W) | 逐像素预测深度 = `local_points[...,2]`,**已乘 metric** → 公制(米) |
| `conf` | (N,H,W) | 置信度 = `sigmoid(conf)` |
| `metric` | 标量 | metric 头预测的全局绝对尺度 |

> 关键前提：在 [`pi3x.py`](../../third_party/pi3/pi3/models/pi3x.py) 的 `forward_head` 里,
> `local_points` 与 `camera_poses` 的平移**都乘了同一个 `metric` 标量**,
> 所以"相机轨迹"与"预测深度"天然处于同一尺度。

---

### 4.1 `model` —— 仅 RGB

| | |
|---|---|
| **输入** | `imgs`(无 conditioning) |
| **计算** | `rgb = run_pi3x(imgs)`;`L_model = path_length(rgb["cam_pos"])` |
| **输出** | `metric_rgb`、`L_model`、`e_model = |L_model − L_gt| / L_gt` |

本质：完全采信 Pi3X 在 RGB-only 下的 metric 头(无任何外部约束)。

---

### 4.2 `model_int` —— RGB + 标定内参

| | |
|---|---|
| **输入** | `imgs` + `conditions_int = {intrinsics, depths=None}` |
| **计算** | `intr = run_pi3x(imgs, conditions_int)`;`L_model_int = path_length(intr["cam_pos"])` |
| **输出** | `metric_int`、`L_model_int`、`e_model_int = |L_model_int − L_gt| / L_gt` |

本质：把"resize 后的标定内参"作为 Pi3X 条件输入(深度不给),让模型在已知相机几何的前提下做重建。
预期效果：消除 RGB-only 下因焦距未知导致的尺度偏置(初步实验显示 RGB-only 焦距系统性偏小 2-3%)。

---

### 4.3 `model_dc` —— RGB + 内参 + 深度

| | |
|---|---|
| **输入** | `imgs` + 完整 `conditions = {intrinsics, depths}` |
| **计算** | `dc = run_pi3x(imgs, conditions)`;`L_model_dc = path_length(dc["cam_pos"])` |
| **输出** | `metric_dc`、`L_model_dc`、`e_model_dc = |L_model_dc − L_gt| / L_gt` |

本质：让深度图与内参共同在网络内部参与推断，由模型融合出尺度，但深度图的质量对重建结果影响大。

---

### 4.4 评分与诊断量(三者共用)

```
e_<variant>      = |L_<variant> − L_gt| / L_gt        # 主评分:轨迹长度相对误差
c_gt_<variant>   = umeyama_scale(<variant>_pos, gt)   # 该变体对齐到 GT 的理想尺度
scale_err_<v>    = |1.0 − c_gt_<v>| / c_gt_<v>        # 纯尺度误差(剥离轨迹形状误差)
s_depth_<v>      = median(D_sensor / D_pred_<v>)      # 诊断:模型深度 vs 传感器深度
```

- `e_*` —— **主评分**,越小越准
- `c_gt_*` —— 若 ≈ 1.0,说明该变体 metric 头本身就接近 GT
- `s_depth_*` —— 诊断量,若与 1.0 偏离,说明该变体存在系统深度偏差

---

## 5. "实际距离"怎么算(GT锚点)

完全**不依赖视觉/深度**,来自机器人本体里程计：

```python
# load_gt_camera_positions()
p_body    = state[:, 0:3]                  # 身体中心世界坐标 x,y,z (Unitree sportmodestate.position)
quat      = state[:, 6:10]                 # 四元数(w,x,y,z) → R_world_body
t_c2b     = info.json["head_camera_extrinsic.t_cam2gripper"]  # 身体系下相机原点 (手眼标定; 兼容旧名 t_cam2robot)

# 身体中心轨迹 → 相机中心轨迹:
cam_pos = R_world_body @ t_c2b + p_body

# GT距离 = 相机中心折线总长:
L_gt = Σ ‖cam_pos[i+1] − cam_pos[i]‖      # path_length()
```

- **重建侧距离** `L_<variant> = Σ‖camera_poses 平移[i+1] − [i]‖`
- **误差** `= |L_<variant> − L_gt| / L_gt`
- `L_gt_robot`(忽略手眼、直接用机器本体 xyz)作对照;若与 `L_gt` 接近即说明手眼换算正确
- `umeyama_scale(<v>_pos, gt_cam)` 给出该变体的理想尺度 `c_gt_<v>`(对齐到真值所需的最优缩放),作参照

---

## 6. 运行方式

> 可在项目根目录运行(脚本会自动向上定位项目根)。服务器(Linux)上自动用 bf16+Flash-Attention、
> 模型只加载一次;批量时跨 episode 复用同一份模型。

**服务器 — 单集(推荐,全分辨率 GPU)：**
```bash
python tools/exp_scale/exp_scale_compare.py --rosbag_dir <rosbag目录> --episode episode_000
```

**服务器 — 批量(一个 rosbag 的全部 episode)：**
```bash
python tools/exp_scale/exp_scale_compare.py --rosbag_dir <rosbag目录> --episode all
```

**服务器 — 批量(遍历多个 rosbag)：**
```bash
python tools/exp_scale/exp_scale_compare.py --input_root <含多个 rosbag_* 的根目录> --episode all
```

**小内存显卡(eg., 6G) / 无 GPU(CPU 回退,需降分辨率)：**
```bash
python tools/exp_scale/exp_scale_compare.py --episode episode_000 --cpu --pixel_limit 40000
```

**只重新出图：** 单集 `exp_plot.py <episode目录> episode`;汇总 `exp_plot.py <exp_out根目录> summary`。

### 主要参数
| 参数 | 默认 | 说明 |
|---|---|---|
| `--rosbag_dir` | 见脚本 | 单个 rosbag 目录(未给 `--input_root` 时用) |
| `--input_root` | 空 | 含多个 `rosbag_*` 的根目录,遍历其下所有 rosbag |
| `--episode` | `episode_001` | episode 名;用 `all` 处理该 rosbag 下全部 episode |
| `--min_motion` | `0.3` | 真值相机轨迹 < 此值(米)的近静止集跳过,不计入汇总 |
| `--per_episode_plots` | 关 | 批量时也为每集出 6 图(默认仅单集出图,批量只出汇总图) |
| `--cpu` | 关 | 强制 CPU(显存 <~8GB 时用;导入阶段即生效) |
| `--pixel_limit` | `255000` | 每帧最大像素数;CPU 上调小可加速(注意力显存随帧数×token 数平方增长) |
| `--conf_thr` | `0.1` | 有效像素的最小 Pi3X 置信度(仅用于 `s_depth_*` 诊断) |
| `--dmin` / `--dmax` | `0.25` / `6.0` | 可信传感器深度区间(米)(仅用于 `s_depth_*` 诊断) |

---

## 7. 输出

### 每个 episode(写到 `exp_out/<episode>/`,多 rosbag 时为 `exp_out/<rosbag>/<episode>/`)
- `metrics.json` —— 所有标量与逐帧/累计曲线数据(含 `skipped` 标志)
- `plotdata.npz` + 6 张图(仅在单集或 `--per_episode_plots` 且非近静止时生成)：
  1. `1_headline_error.png` —— 三变体的轨迹长度误差(**主结论**)
  2. `2_correction_factor.png` —— 三变体 metric(c=1) vs 各自理想 c_gt
  3. `3_per_frame_scale.png` —— 三变体逐帧 `D_sensor/D_pred` 比值(看漂移与偏置)
  4. `4_cumulative_length.png` —— 三变体累计轨迹长度 vs 真值
  5. `5_depth_scatter.png` —— 三变体预测深度 vs 传感器深度散点(含 y=x)
  6. `6_depth_error_maps.png` —— 采样帧的深度误差热力图(基于 RGB-only)

### 跨 episode 汇总(批量时写到 `exp_out/`)
- `summary.json` —— `n_episodes`、各变体误差的 `mean/median/std`、各变体"最优次数" `win_counts`、逐集 `rows`
- `summary.csv` —— 每行一个 episode 的关键指标,便于表格查看
- `summary.png` —— 三面板：误差均值±标准差柱状图 / 最优次数 / 三变体的 `c_gt_*` 分布(越靠近 1.0 越准)

### metrics.json 关键字段

| 字段 | 含义 |
|---|---|
| `L_gt` / `L_gt_robot` | 真值相机 / 机器本体轨迹长度(米) |
| `L_model` / `L_model_int` / `L_model_dc` | 三变体的轨迹长度 |
| `metric_rgb` / `metric_int` / `metric_dc` | 三变体的 Pi3X metric 头标量，表示：是网络"自我标定到米"的乘数;它学多大都行,只要乘完之后的结果对齐 GT 即可。|
| `c_gt_rgb` / `c_gt_int` / `c_gt_dc` | 三变体各自 Umeyama 理想尺度，表示：是"网络声称的米"距离"真实的米"还差多少倍。|
| `s_depth_rgb` / `s_depth_int` / `s_depth_dc` | 三变体预测深度 vs 传感器深度的全局中位比(诊断) |
| `e_model` / `e_model_int` / `e_model_dc` | 三变体的轨迹长度相对误差(**主评分**) |
| `scale_err_model` / `scale_err_model_int` / `scale_err_model_dc` | 纯尺度误差(metric c=1 vs `c_gt_*`) |
| `s_per_frame_rgb` / `s_per_frame_int` / `s_per_frame_dc` | 逐帧深度比(诊断) |
| `cum_gt` / `cum_model` / `cum_model_int` / `cum_model_dc` | 累计轨迹长度 |

---

## 8. 跨平台说明(Windows 规避已内置;Linux 走原生快路径)

1. **模型加载(已集中到 `L3ROcc/base.py:_load_pretrained_model`)** —— Linux 直接原生
   `from_pretrained`;Windows 上 `from_pretrained` 会段错误(先建模型再加载权重,破坏 safetensors
   mmap),故 Windows 分支先把权重加载到 CPU、再惰性导入模型并 `load_state_dict`。数据生成与本验证
   脚本共用此逻辑——验证脚本**不再**自带 monkeypatch。
2. **显存** —— fp32 模型≈5.44GB。**三次**推理对显存累计开销 ≈ 单次的 1.5-2×(activations 释放后峰值不叠加,
   但运行时间几乎线性增长)。服务器大显存直接 GPU 全分辨率;本机 6GB 用
   `--cpu`(内部设 `CUDA_VISIBLE_DEVICES=-1`)+ 调小 `--pixel_limit`。
3. **绘图(`exp_plot.py`)** —— Linux 进程内直接调用;Windows 必须独立子进程并设
   `MKL_THREADING_LAYER=SEQUENTIAL` / `KMP_DUPLICATE_LIB_OK=TRUE` / `OMP_NUM_THREADS=1`,
   否则 torch 的 MKL 驻留同进程会让 matplotlib 崩溃(`0xc06d007f`)。同类 `np.linalg.svd` 崩溃也是
   `umeyama_scale` 改用闭式公式(不做 SVD)的原因。
4. **K_rescaled 元数据透传** —— `load_images_as_tensor` 在 `conditions` 中追加了 `K_rescaled`(numpy 3×3,
   resize 分辨率下的标定内参)。它不是 Pi3X 的 kwarg,`run_pi3x` 内部在 `**conditions` splat 前会先
   过滤掉这个键。

---

## 9. 与上一版实验的差异

| 项 | 旧实验(三变体) | 新实验(三变体) |
|---|---|---|
| 第一组 | `model`(RGB only) | `model`(RGB only)—— 保留 |
| 第二组 | `depth`(RGB 重建 × `s_depth = median(D_sensor/D_pred)`) | **`model_int`(RGB + 标定内参)**—— 全新 |
| 第三组 | `model_dc`(RGB + 内参 + 深度,作为 conditioning) | `model_dc`(同) |
| 评分变量 | `e_model` / `e_depth` / `e_model_dc` | `e_model` / `e_model_int` / `e_model_dc` |
| 主问题 | "哪种 scale 修正更准:模型 metric vs 传感器深度" | **"哪种 conditioning 让 metric 最准:RGB-only vs +intr vs +intr+depth"** |
| 重建数 | 2 次推理(RGB / depth-cond) | **3 次推理**(RGB / intr / intr+depth) |

旧的 `depth` 变体是"对 RGB-only 几何乘上一个由深度推算的尺度",属于事后修正。
新的 `model_int` 是"把内参直接喂给 Pi3X 在网络内部约束 metric 头",属于输入端修正。
研究问题从"哪种 scale 修正策略最好"切换到"在多模态条件输入中,加入哪些信号最有助于 metric 头"。

---

## 10. 实验结果

> ⚠️ 上一版的 32 集结果(`model=13.34%` / `depth=21.80%` / `model_dc=12.71%`)针对的是旧三变体,
> 与本版的 `model` / `model_int` / `model_dc` **不可直接对照**。新一版结果待重跑后补充。

**建议结果模板(待填)：**

| Variant | Mean Error | Median Error | Std Dev | Win Count |
|---|---|---|---|---|
| `model` — RGB only | TBD | TBD | TBD | TBD |
| `model_int` — RGB + intrinsic | TBD | TBD | TBD | TBD |
| `model_dc` — RGB + intrinsic + depth | TBD | TBD | TBD | TBD |

需要关注的关键观察点:
1. **`model_int` 相对 `model` 的提升量** —— 验证 MVP 实验的发现是否在整集尺度仍然成立(RGB-only fx/fy 系统偏小 2-3%,导致重建几何被压扁、轨迹长度被低估)
2. **`model_dc` 相对 `model_int` 的提升量** —— 验证传感器深度作为 conditioning 是否再带来增益,以及该增益是否值得开启 depth 输入的成本(数据采集 / 存储 / 标定)
3. **`s_depth_*` 是否朝 1.0 移动** —— 三变体的预测深度对传感器深度的偏置是否随 conditioning 增加而减小
