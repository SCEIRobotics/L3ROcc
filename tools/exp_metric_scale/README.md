# Pi3X metric scale 真实世界准确性验证（不依赖 GT 位姿对齐）

验证 Pi3X `metric_head` 输出的绝对尺度在真实世界里到底准不准、能不能直接用，**全程不使用 GT
odom 位姿对齐/纠正**（即不走 `intern_vln_env.align_with_gt_scale`，不读 parquet 的
`observation.state` / `action` 轨迹）。只用与 odom 无关的独立公制参照来**测量**误差。

## 背景

`metric_head`（`third_party/pi3/pi3/models/pi3x.py:399`）输出一个全局标量 `metric`（log 空间，
`.exp()` 后得到），统一缩放 `points` / `camera_poses` 平移 / `local_points`。已有观察：跨数据集、
无论内参/深度是否参与输入，`metric` 都≈0.3。

> **关键概念区分**：`metric` 标量（≈0.3）**不是**真实世界尺度比。它只是作用在归一化几何上的内部
> 乘子，其数值本身不代表准不准。真正衡量真实尺度准确性的是 **预测深度 / 传感器深度 的比值**
> （`scale_ratio`，≈1.0 才说明准）。自检中即出现 `metric`=0.30 但 `scale_ratio`≈0.92 的情况
> —— 说明该序列重建尺度其实只偏小约 8%，相当准。

## 三条互补证据链

| 方法 | 角色 | 做法 |
|---|---|---|
| **A 逐像素深度比值** | 主，定量 | 模型只喂 RGB(+内参)，**不喂深度**；与 16-bit 公制传感器深度逐像素比 `r=pred/sensor`。`median(r)`≈1 即准。传感器深度仅作参照、绝不入模型、绝不碰 odom。 |
| **B 已知物理尺寸** | 辅，零 GT | 导出重建点云 PLY + 自动地面 RANSAC 距离统计，在点云查看器里量取已知尺寸结构（层高/门高/桌面 0.75m/机器人自身尺寸）做交叉验证。 |
| **C metric 标量分布** | 诊断 | 跨序列统计 `metric` 标量分布。若恒≈0.3（CV 很小）而方法 A 的真实 ratio 随场景明显变化 ⇒ `metric_head` 已塌缩为常数、不可把该标量当真实公制尺度的指示。 |

输入条件只测 **仅RGB** 与 **RGB+内参**（喂深度作输入=作弊，会污染尺度验证）。

## 用法

```bash
# 单个 rosbag / 含多 rosbag_* 的父目录 / InternData-N1 嵌套布局，--input 自动识别
python tools/exp_metric_scale/eval_metric_scale.py \
    --input <数据集根> --episode all \
    --cond rgb rgb_intr --export-ply \
    --out-dir tools/exp_metric_scale/exp_out

# 单集快速验证
python tools/exp_metric_scale/eval_metric_scale.py \
    --input <某 rosbag 目录> --episode episode_000 --export-ply

# 自检（上界对照）：把传感器深度喂进模型，ratio 应明显趋近 1、AbsRel 骤降、δ1.25→1
python tools/exp_metric_scale/eval_metric_scale.py \
    --input <rosbag> --episode episode_000 --cheat-depth
```

常用参数：`--intrinsics K.npy`（3x3，最高优先级覆盖）、`--conf-thr`(默认 0.1)、
`--dmin/--dmax`(传感器深度可信区间，默认 0.1~10m；批处理脚本默认 0.25~6.0)、`--interval`(默认用 config)、`--cpu`。
内参优先级：CLI npy > parquet `observation.camera_intrinsic` > `meta/info.json head_camera_intrinsic` >
None(让 Pi3X 自反算)。

## 实采深度（lerobot/ZED）反光空洞与异常值的鲁棒过滤（默认开）

实采深度在反光/镜面处会出现**空洞**（值=0）与**在量程内但物理错误**的异常值，直接逐像素比会被污染。
方法 A 默认开启三项鲁棒过滤，并**同时输出未过滤的 `*_raw` 对照**：

1. **空洞边界腐蚀** `--hole-erode`(默认 1px)：对有效 sensor 掩码做二值腐蚀，剔反光空洞周边飞点。
2. **sensor 深度边缘过滤** `--sensor-rtol`(默认 0.1)：对传感器深度做 `depth_edge`（带 mask 忽略空洞），
   剔深度突变像素（与原有"仅对预测深度"的边缘过滤互补）。
3. **MAD 离群剔除** `--mad-k`(默认 3.0)：对 `log(pred/sensor)` 按 `中位数 ± k·1.4826·MAD` 剔离群比值，
   再算 `absrel`/`δ1.25`（专治镜面在量程内的错值，因其影响 mean 类指标）。

`--no-robust` 关闭整套、回到旧行为（headline 即等于 raw）。上游加载已把 0/65535 置 0 且用
`INTER_NEAREST` resize（不会在空洞边界插值出假深度），本节是在其之上的进一步过滤。

## 产出

- `exp_out/<episode>/metrics.json` —— 每条件：
  - headline（鲁棒）：`scale_ratio_median`、`scale_ratio_iqr`、`log_ratio_std`、`correction_factor`、
    `absrel`、`delta1_25`、`scale_ratio_median_of_frames`、`metric_scalar`、`valid_pixels`、逐帧 ratio；
  - 反光诊断：`hole_frac`(空洞/反光严重度)、`valid_frac`、`outlier_frac`(MAD 剔除比)；
  - raw 对照：`scale_ratio_median_raw`、`absrel_raw`、`delta1_25_raw`、`valid_pixels_raw`；
  - `--export-ply` 时另有 `ground_plane`（地面 RANSAC 平面、inlier 比、地面以上高度跨度 p5–p95）。
- `exp_out/<episode>/pcd_<cond>.ply` —— 重建点云（方法 B 量取用）。
- `exp_out/summary.json` + `summary.png` —— 方法 C 的 metric 分布、方法 A 的 ratio 汇总（鲁棒 + raw 对照
  + `hole_frac/valid_frac/outlier_frac` 均值），以及「metric 标量 vs 真实 ratio」散点。

## 结果判读

- `scale_ratio(pred/sensor)` median **≈1.0** ⇒ 真实尺度准，metric scale 可直接用；**<1** 偏小、
  **>1** 偏大。
- `metric` 标量数值（常≈0.3）**本身不是**准确性指标。
- `metric` 标量 CV 很小（近常数）但 `scale_ratio` 仍随场景明显变化 ⇒ `metric_head` 塌缩为常数，
  解释了「跨数据集都 0.3」，此时不可把 metric 标量当真实公制尺度。
- `correction_factor` = 把 `pred` 乘上它即匹配传感器（=`median(sensor/pred)`）；`absrel`/`delta1_25`
  是**尺度校正后**的形状误差（剥离尺度，看相对深度本身好不好）。
- 反光诊断：`hole_frac` 高 ⇒ 该序列反光/空洞重、需重点关注；正常情况下鲁棒后 `absrel ≤ absrel_raw`，
  而 headline `scale_ratio_median` 相对 `_raw` 仅微动（说明剔的是离群/飞点而非信号）。

## 与 `tools/exp_scale/` 的区别

`tools/exp_scale/exp_scale_compare.py` 用 **GT odom 轨迹长度**给尺度打分（依赖 `observation.state`/
`action` + 手眼标定）。本实验**刻意不用任何 GT odom**，只用传感器深度/已知物理尺寸作独立参照，
专门回答「metric scale 的真实世界绝对准确性」这一问题。
