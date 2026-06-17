# Pi3X metric scale 真实尺度准确性验证（GT-free）

验证 Pi3X `metric_head` 输出的绝对尺度在真实世界准不准、能否直接用，**全程不使用 GT odom
位姿对齐**（不走 `align_with_gt_scale`，不读 parquet 的 `observation.state`/`action`），只用与
odom 无关的独立参照来**测量**误差。

## 关键概念

`metric_head` 输出一个全局标量 `metric`（≈0.3，跨数据集近常数），统一缩放
`points`/`camera_poses`/`local_points`。但 **`metric` 标量本身不是真实尺度比**，只是作用在归一化
几何上的内部乘子。真正衡量准确性的是 **预测深度 / 传感器深度 的比值** `scale_ratio`（≈1.0 才准）。

## 三条证据链

| 方法 | 角色 | 做法 |
|---|---|---|
| **A 逐像素深度比值** | 主（定量） | 模型只喂 RGB(+内参)、**不喂深度**；与 16-bit 公制传感器深度逐像素比 `r=pred/sensor`，`median(r)`≈1 即准。传感器深度仅作参照，绝不入模型。 |
| **B 已知物理尺寸** | 辅（零 GT） | 导出重建点云 PLY + 地面 RANSAC，在查看器量取已知结构（层高/门高/桌面 0.75m）交叉验证。 |
| **C metric 标量分布** | 诊断 | 统计 `metric` 标量分布；若恒≈0.3 而方法 A 的真实 ratio 随场景变化 ⇒ `metric_head` 已塌缩为常数，不可当真实尺度。 |

输入只测 **仅RGB** 与 **RGB+内参**（喂深度作输入=作弊，污染验证）。默认开启实采深度的鲁棒过滤
（空洞腐蚀 / 深度边缘 / MAD 离群剔除），并同时输出未过滤的 `*_raw` 对照；`--no-robust` 关闭。

## 用法

```bash
# 单 rosbag / 多 rosbag_* 父目录 / InternData-N1 布局，--input 自动识别
python tools/exp_metric_scale/eval_metric_scale.py \
    --input <数据集根> --episode all --cond rgb rgb_intr --export-ply \
    --out-dir tools/exp_metric_scale/exp_out

# 自检（上界对照）：把传感器深度喂进模型，ratio 应趋近 1
python tools/exp_metric_scale/eval_metric_scale.py --input <rosbag> --episode episode_000 --cheat-depth
```

常用参数：`--intrinsics K.npy`（最高优先级）、`--conf-thr`(默认 0.1)、`--dmin/--dmax`(深度可信区间)、
`--cpu`。内参优先级：CLI npy > parquet `observation.camera_intrinsic` > `meta/info.json` > None。

## 产出与判读

- `exp_out/<episode>/metrics.json`：`scale_ratio_median`/`_iqr`、`absrel`、`delta1_25`、`metric_scalar`、
  反光诊断 `hole_frac`/`valid_frac`/`outlier_frac`、`*_raw` 对照；`--export-ply` 另含 `ground_plane`。
- `exp_out/<episode>/pcd_<cond>.ply`、`exp_out/summary.{json,png}`（含 metric 标量 vs 真实 ratio 散点）。

判读：`scale_ratio(pred/sensor)` median **≈1.0** ⇒ 尺度准；**<1** 偏小、**>1** 偏大。`metric` 标量
数值本身不是准确性指标。`correction_factor=median(sensor/pred)`，乘到 `pred` 即匹配传感器。

> 结论（实采 LeRobot/ZED）：`pred/sensor` median ≈ **0.92**（偏小约 8%，相当准），RGB-only metric head
> 可直接用；如需补偿可在 config 设 `metric_scale_correction ≈ 1.08`。
