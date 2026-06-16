# exp_coord_align — 坐标系对齐可视化对比实验

把 `tools/run_intern_nav_occ.py` Pipeline 中三套数据放到同一个可比的 **base 坐标系**直接叠加对比，
**全程不调用** `align_to_world` / `align_with_gt_scale`，用于诊断 Pi3X 重建本身（尺度 / 朝向 / 漂移）
相对 N1 GT 的真实差距。

## 坐标链路（均无优化对齐）

| 数据 | 源坐标系 | 变换链 |
|---|---|---|
| ① Pi3X `points`（点云场景） | 模型 world（锚定帧0相机, OpenCV RDF） | `inv(camera_poses[0])` → 帧0相机系 → `R(T_cam2base)` → base |
| ② Pi3X `camera_poses`（重建轨迹） | camera→world SE(3)，已 metric 缩放 | 同上（取每帧相机中心） |
| ③ N1 `action`（GT 轨迹） | GT camera→world 4×4 | `inv(action[0])` → 帧0相机系 → `R(T_cam2base)` → base |

- 参考系：各自**帧0相机**。因此两条轨迹都从 base 系原点 (0,0,0) 出发，可直接叠加。
- `T_cam2base` 取自 parquet `observation.camera_extrinsic`（或手眼 `R_cam2gripper`），**仅用旋转**（与生产 `convert_pointcloud_camera_to_base` 一致）。

复用的现有函数：`InternNavDataGenerator.pcd_reconstruction` / `get_gt_poses` /
`convert_pointcloud_world_to_camera` / `convert_pointcloud_camera_to_base`，`write_ply`。

## 翻转根因 + C 修正（OpenCV↔OpenGL，重要）

实验发现 Pi3X 轨迹与 N1 `action` 在 base 系**整体绕 X 轴翻转**（需绕 X 轴 180° 才重合，且 Z 朝下）。根因：

- Pi3X `camera_poses` 是 **OpenCV** 相机系 (X-right, Y-down, Z-forward)。
- N1 `action` / `observation.camera_extrinsic`(=`T_cam2base`) 是 **3D-Front 渲染相机** 的 **OpenGL/Blender**
  约定 (X-right, **Y-up, Z-back**)；实测 `action[0]` 相机 Y 轴指世界上。
- 两者相差标准 **OpenCV↔OpenGL 换基 `C = diag(1,-1,-1)`**（翻转相机 Y、Z = 绕相机 X 轴 180°，det=+1）。
  10 条轨迹 Kabsch 实测确认：`R_fix_base = diag(1,-1,-1)`，应用后残差 0.4%~5%、轨迹长度比≈1（Pi3X 重建本身很准）。

修正：在 world→camera 之后、camera→base 之前对 Pi3X 相机系点应用 `C`（`p_render = C @ p_opencv`）。
本实验对 Pi3X 数据应用 `C`（fixed），N1 `action` 不应用；同时输出未修正(raw)与“经验 R_fix 自检”做护栏。
**生产管线已同步修复**：`L3ROcc/base.py` 把 `C` 折进 `compute_sequence_data` 的 `T_cam2base` 旋转
(`R_eff = R_c2b @ C`)，点云/轨迹/可见性射线一致换基（常量 `R_OPENCV_TO_OPENGL`，实验复用同一常量）。

## 用法

```bash
# 默认路径
bash tools/run_exp_bash/run_exp_coord_align.sh
# 自定义路径 + 只跑一集
bash tools/run_exp_bash/run_exp_coord_align.sh <dataset_root> <out_dir> EPISODE=episode_000000
# 直接调脚本
python tools/exp_coord_align/exp_coord_align.py --dataset_root <root> --episode all
```

环境变量：`EPISODE` / `MODEL_TYPE`(pi3|pi3x) / `USE_DEPTH` / `NO_INTRINSIC` / `CPU`。

## 输出（`out_dir/<label>/`）

- `scene_base.ply` — base 系点云（带颜色，已应用 `C`）。
- `traj_pi3x_base.ply`（蓝，修正后）/ `traj_pi3x_base_raw.ply`（青，修正前/翻转）/ `traj_gt_base.ply`（红）— 已加密成线，可在 meshlab/open3d 叠加。
- `compare_traj.png` — 3D + XY/XZ/YZ 三视图叠加 GT、Pi3X-fixed(实线)、Pi3X-raw(虚线)，标注帧0原点。
- `metrics.json` / `metrics.txt` — 帧数、轨迹总长比(尺度比)、点云 bbox；**修正前后** 误差
  `pos_err_*_raw` / `pos_err_*_fixed`、`endpoint_err_raw/_fixed`；以及 `rfix_selfcheck`
  （经验 `R_fix_base_diag`、`C_emp_diag`、`c_matches_empirical`、`resid_after_Rfix`）。
- 根目录 `summary.json` — 所有轨迹度量汇总。

## 结果解读

- 两条轨迹起点必重合于原点；**尺度比 `scale_ratio_pi3x_over_gt` 接近 1** 说明 Pi3X metric_head 尺度准。
- `pos_err_mean_raw`（数米/翻转态）→ `pos_err_mean_fixed`（应骤降至重建误差量级）证明 `C` 即修复；
  `rfix_selfcheck.c_matches_empirical=true` 且 `R_fix_base_diag≈[1,-1,-1]` 为护栏验证。
- meshlab 里 `traj_pi3x_base.ply`(fixed) 应与 `traj_gt_base.ply` 基本重合且 Z 朝上，`*_raw.ply` 仍翻转。
- 残差 `pos_err_mean_fixed` 即 Pi3X 重建相对 GT 的真实漂移（`align_to_world` 实际在修正的部分）。
