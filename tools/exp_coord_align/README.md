# exp_coord_align — 坐标系对齐可视化对比（GT-free）

把 `run_intern_nav_occ.py` Pipeline 的三套数据放到同一 **base 坐标系**直接叠加对比，**全程不调用**
`align_to_world` / `align_with_gt_scale`，用于诊断 Pi3X 重建本身（尺度 / 朝向 / 漂移）相对 N1 GT 的真实差距。

## 坐标链路（均无优化对齐）

| 数据 | 源坐标系 | 变换链 |
|---|---|---|
| ① Pi3X `points`（点云） | 模型 world（锚定帧0相机, OpenCV） | `inv(camera_poses[0])` → 帧0相机系 → `R(T_cam2base)` → base |
| ② Pi3X `camera_poses`（重建轨迹） | camera→world，已 metric 缩放 | 同上（取每帧相机中心） |
| ③ N1 `action`（GT 轨迹） | GT camera→world 4×4 | `inv(action[0])` → 帧0相机系 → `R(T_cam2base)` → base |

- 参考系为各自**帧0相机**，故两条轨迹都从原点出发可直接叠加。`T_cam2base` **仅用旋转**（与生产
  `convert_pointcloud_camera_to_base` 一致）。复用：`pcd_reconstruction`/`get_gt_poses`/`convert_*`。

## 相机约定换基 C（按数据集区分，重要）

Pi3X `camera_poses` 是 **OpenCV** (X-right,Y-down,Z-forward)。外参 `T_cam2base` 的约定随数据集不同：

| 数据集 | 外参来源 | 约定 | 需要的 C | 不换基的现象 |
|---|---|---|---|---|
| **InternData-N1** | parquet `observation.camera_extrinsic`（3D-Front 渲染） | **OpenGL** | `diag(1,-1,-1)` | base 系 Pi3X 与 GT 绕 X 翻转、Z 朝下 |
| **Lerobot** | `meta/info.json` 手眼 `R_cam2gripper` | **OpenCV** | **identity** | 直接对齐；误施翻转反而错 |

约定 `extrinsic_convention`(opengl/opencv) 由 `InternNavSequenceLoader.get_trajectory_info` 产出并透传。
**生产管线已同步**：`base.py compute_sequence_data` 仅对 opengl 把 `C` 折进 `T_cam2base` 旋转。

## 重力倾斜纠偏（默认开，`--no_gravity_align` 关）

Pi3X 重建在 base 系常有轻微 z 倾斜（Lerobot 实测约 **4.7°**）。复用 `_gravity_align_to_z`
（地面 RANSAC → 法向 → 绕 base 原点 Rodrigues 旋到 +Z），**只纠倾斜**。仅对 Pi3X fixed 数据且有真实
base 外参时生效；GT 与 raw 不纠偏作对照。`metrics.json` 记 `gravity_tilt_before/after_deg`。

## 用法

```bash
python tools/exp_coord_align/exp_coord_align.py --dataset_root <root> --episode all
# 环境变量：EPISODE / MODEL_TYPE(pi3|pi3x) / USE_DEPTH / NO_INTRINSIC / GRAVITY_ALIGN / CPU
```

## 输出（`out_dir/<label>/`）

- `scene_base.ply`（base 系点云，已换基+纠偏）、`traj_pi3x_base.ply`(蓝,修正后) /
  `traj_pi3x_base_raw.ply`(青,未换基对照) / `traj_gt_base.ply`(红)、`compare_traj.png`(3D+三视图)。
- `metrics.json/.txt`：`extrinsic_convention`、轨迹长度比、`pos_err_*_raw` / `_fixed`、
  `rfix_selfcheck`(经验换基 vs 期望 C 一致性护栏)、重力纠偏指标；根目录 `summary.json` 汇总。

## 判读

- 起点必重合；**尺度比 ≈ 1** ⇒ Pi3X metric_head 尺度准。
- N1(opengl)：`pos_err_*_raw`(翻转态/数米) → `_fixed`(骤降)，`R_fix_base_diag≈[1,-1,-1]`。
- Lerobot(opencv)：`R_fix_base_diag≈[1,1,1]`，`_fixed≈_raw`（无需翻转），轨迹直接重合、Z 朝上。
- 残差 `pos_err_mean_fixed` 即 Pi3X 相对 GT 的真实漂移。
