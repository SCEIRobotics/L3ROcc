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

## 相机约定换基 C（按数据集区分 OpenCV/OpenGL，重要）

Pi3X `camera_poses` 是 **OpenCV** 相机系 (X-right, Y-down, Z-forward)。外参 `T_cam2base` 的相机约定
**随数据集不同**，对 Pi3X 是否需要换基也不同：

| 数据集 | 外参来源 | 相机约定 | 需要的 C | 现象 |
|---|---|---|---|---|
| **InternData-N1** | parquet `observation.camera_extrinsic`（3D-Front 渲染） | **OpenGL** (Y-up, Z-back) | `diag(1,-1,-1)` | 不换基则 base 系 Pi3X 与 GT 绕 X 轴翻转、Z 朝下 |
| **Lerobot**（实采） | `meta/info.json` 手眼 `R_cam2gripper` | **OpenCV** (Y-down，与 Pi3X 同) | **identity（不翻转）** | 直接对齐；误施加翻转反而错 |

- N1 经 10 条轨迹 Kabsch 实测确认 `R_fix_base = diag(1,-1,-1)`，应用后残差 0.4%~5%、长度比≈1（Pi3X 本身很准）。
- Lerobot GT 相机 Y 轴世界 z 分量 ≈ −0.905（朝下）= OpenCV，故 `C=identity`。
- 约定 `extrinsic_convention`(opengl/opencv/None) 由 `InternNavSequenceLoader.get_trajectory_info` 按外参来源产出并透传。

修正：world→camera 之后、camera→base 之前对 Pi3X 相机系点应用**该数据集期望的 C**（`p = C @ p_opencv`）。
本实验对 Pi3X 应用期望 C（fixed），GT 不换基；同时输出未修正(raw) 与“经验 R_fix 自检”护栏
（Kabsch 反求 `C_emp` 与该数据集期望 C 对比，防止用错常量）。
**生产管线已同步修复**：`L3ROcc/base.py compute_sequence_data` 按 `extrinsic_convention` **仅对 opengl** 把
`C=R_OPENCV_TO_OPENGL` 折进 `T_cam2base` 旋转（点云/轨迹/可见性射线一致换基），opencv/None 不翻转。

## 重力倾斜纠偏（`--no_gravity_align` 关闭，默认开）

Pi3X 重建在 base 系常有轻微 z 倾斜（地面不水平，Lerobot 上实测约 **4.7°**）。**仅参照** `align_to_world`：
复用生产 `InternNavDataGenerator._gravity_align_to_z`（地面 RANSAC → 法向 → 绕 base 原点 Rodrigues 旋到 +Z），
**只纠倾斜**，不做 align_to_world 的尺度/yaw/原点等其余步骤。

- 仅对 Pi3X **fixed** 数据（`scene_base.ply` + `traj_pi3x_base.ply`）、且有真实 base 外参（`T_cam2base≠None`）时生效。
- GT 与 raw **不**纠偏：`traj_pi3x_base_raw.ply` 保留倾斜态作 before 对照，GT 为重力基准。
- `metrics.json` 记 `gravity_applied` / `gravity_tilt_before_deg`(≈4.7) / `gravity_tilt_after_deg`(≈0) / `gravity_ground_z`。
- 关闭：`--no_gravity_align`（或 bash `GRAVITY_ALIGN=false`）。

## 用法

```bash
# 默认路径
bash tools/run_exp_bash/run_exp_coord_align.sh
# 自定义路径 + 只跑一集
bash tools/run_exp_bash/run_exp_coord_align.sh <dataset_root> <out_dir> EPISODE=episode_000000
# 直接调脚本
python tools/exp_coord_align/exp_coord_align.py --dataset_root <root> --episode all
```

环境变量：`EPISODE` / `MODEL_TYPE`(pi3|pi3x) / `USE_DEPTH` / `NO_INTRINSIC` / `GRAVITY_ALIGN` / `CPU`。

## 输出（`out_dir/<label>/`）

- `scene_base.ply` — base 系点云（带颜色，已应用该数据集期望 C **+ 重力倾斜纠偏**）。
- `traj_pi3x_base.ply`（蓝，修正后**+纠偏**）/ `traj_pi3x_base_raw.ply`（青，未换基/未纠偏对照）/ `traj_gt_base.ply`（红）— 已加密成线，可在 meshlab/open3d 叠加。
- `compare_traj.png` — 3D + XY/XZ/YZ 三视图叠加 GT、Pi3X-fixed(实线)、Pi3X-raw(虚线)，标注帧0原点。
- `metrics.json` / `metrics.txt` — `extrinsic_convention`、帧数、轨迹总长比(尺度比)、点云 bbox；**修正前后** 误差
  `pos_err_*_raw` / `pos_err_*_fixed`；`rfix_selfcheck`（`expected_convention`、`expected_C_diag`、
  经验 `R_fix_base_diag`、`C_emp_diag`、`c_matches_empirical`、`resid_after_Rfix`）；
  以及重力纠偏 `gravity_applied` / `gravity_tilt_before_deg` / `gravity_tilt_after_deg` / `gravity_ground_z`。
- 根目录 `summary.json` — 所有轨迹度量汇总。

## 结果解读

- 两条轨迹起点必重合于原点；**尺度比 `scale_ratio_pi3x_over_gt` 接近 1** 说明 Pi3X metric_head 尺度准。
- **N1（opengl）**：`pos_err_mean_raw`（数米/翻转态）→ `_fixed`（骤降）；`R_fix_base_diag≈[1,-1,-1]`。
- **Lerobot（opencv）**：`R_fix_base_diag≈[1,1,1]`，`pos_err_mean_fixed ≈ _raw`（无需翻转即对齐）；
  `traj_pi3x_base.ply` 应直接与 `traj_gt_base.ply` 重合、Z 朝上。
- 两者都应 `rfix_selfcheck.c_matches_empirical=true`（护栏：经验换基与该数据集期望 C 一致）。
- 残差 `pos_err_mean_fixed` 即 Pi3X 重建相对 GT 的真实漂移（`align_to_world` 实际在修正的部分）。
