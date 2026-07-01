<h1 align="center">🌌 L3ROcc: Local 3D Reconstruction with Occupancy</h1>

<p align="center">
    <a href="https://arxiv.org/abs/2507.13347" target="_blank">
    <img src="https://img.shields.io/badge/Engine-&pi;³-00AEEF?style=plastic&logo=arxiv&logoColor=white" alt="Paper">
    </a>
    <a href="#" target="_blank">
    <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=plastic&logo=python&logoColor=white" alt="Python">
    </a>
    <a href="#" target="_blank">
    <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=plastic&logo=pytorch&logoColor=white" alt="PyTorch">
    </a>
</p>

<div align="center">
  <a href="https://youtu.be/oqntFdGxhwg" target="_blank">
    <img src="https://github.com/user-attachments/assets/5a9204d8-88c0-4f09-bf31-ac6a0a690326" width="100%" alt="L3ROcc Demo">
  </a>
  <p><i>Left: RGB Input | Middle: 3D Point Cloud Fusion | Right: 4D Occupancy Grid</i></p>
  <p>
    <b>🎥 <a href="https://youtu.be/oqntFdGxhwg">Watch Full Demo on YouTube</a></b>
  </p>
</div>

`L3ROcc` is a high-performance visual geometry framework designed to transform standard RGB video sequences into high-precision **3D Point Clouds**, **3D Occupancy Grids**, and **4D Temporal Observation Data**.
This project employs **$\pi^3$ (Permutation-Equivariant Visual Geometry Learning)** as its foundational reconstruction engine, and implements a fully automated pipeline for data labeling and alignment that is customized for navigation learning tasks. All processed data adheres to the **LeRobotDataset v2.1** specification. In practical testing, processing a 16-second video segment using this pipeline requires roughly 15 seconds to produce occupancy (occ) and mask data.

## 📢 What's New (Latest Updates)
- **Decoupled Multi-Optional Inputs:** RGB (required), calibrated intrinsics (optional), and depth (optional) are now fully decoupled at every pipeline entry point (CLI / generator constructor / `pcd_reconstruction`). Any combination is valid — RGB-only, RGB+K, RGB+depth, RGB+K+depth — and the three flags `--use_depth` / `--use_intrinsic` / `--condit_intr_path` toggle them independently.
- **Unified Model Interface (`--model_type`):** Both backbones now share a single CLI flag — `--model_type pi3` (RGB-only forward) or `--model_type pi3x` (consumes optional K/depth conditioning at the model layer). The legacy `use_multimodal` parameter has been removed project-wide. When Pi3 is selected with a calibrated K, the K still overrides the DLT-estimated intrinsic saved to Parquet (post-processing path is model-agnostic).
- **GT-Free Metric Scale:** The pipeline trusts Pi3X's `metric_head` for absolute scale (validated against sensor depth and known physical sizes — ~8% under-scale on real data), applying only an optional `metric_scale_correction` factor. The previous GT/Sim3 scale alignment is no longer used. See [Experiments](#-experiments).
- **Base Frame Occupancy & z-Deskew (LeRobot):** Per-frame occupancy is anchored to the **robot base coordinate system** via a per-dataset camera-convention change. An optional z-axis deskew (`--use_z_deskew true`) removes the slight ground tilt and stores the per-frame deskew rotation (`R_deskew`) so the correction is reversible downstream.
- **Smart Data Integrity Checks:** Introduced strict file existence validation. The pipeline now verifies all expected output artifacts for a trajectory before skipping, preventing incomplete or corrupted data generation during batch processing.
- **Enhanced Code Robustness:** Refactored the Object-Oriented structure to standardize subclass method overrides and decoupled absolute paths into relative paths for seamless open-source deployment.

## ✨ Key Features
* **End-to-End Reconstruction**: Directly predicts affine-invariant camera poses and metric-scale point clouds from RGB video streams.
* **Flexible Model Selection**: Two backbone options under a single `--model_type` flag — `pi3` (RGB-only, lightweight) or `pi3x` (accepts optional intrinsic / depth conditioning at the model layer). Pi3X with `--use_depth true` further sharpens absolute scale alignment when high-quality depth is available.
* **Decoupled Optional Inputs**: Calibrated camera intrinsics (`--use_intrinsic true` / `--condit_intr_path`) and depth (`--use_depth true` / `--condit_depth_path`) are independent — supply one, both, or neither. Calibrated intrinsics always override the DLT-estimated K saved to Parquet, regardless of the chosen backbone.
* **Automated Voxelization**: Converts unstructured point clouds into structured Occupancy Grids.
* **Visibility Analysis**: Performs real-time ray casting based on intrinsic and extrinsic parameters of camera to compute visible regions (Visible Masks) and occlusion relationships.
* **4D Data Serialization**:
    * **Sparse OCC**: Utilizes Sparse CSR matrices to store temporal occupancy, significantly reducing disk usage.
    * **Packed Mask**: Implements bit-packing (via `np.packbits`) for visibility masks to optimize storage efficiency.
* **Multi-Dataset Adaptation**: Built-in generators for `SimpleVideo` (single video), [`InternData-N1`](https://huggingface.co/datasets/InternRobotics/InternData-N1) (large-scale), and **LeRobot rosbag** real-robot data (layout auto-detected).
* **Professional Visualization**: Mayavi-based 3D rendering tools for generating side-by-side comparison videos of point clouds, trajectories, and occupancy.

## 💡 Future Work 
- [ ] **Semantic Point Cloud**: Integrate semantic segmentation and instance segmentation to enhance reconstruction quality.
- [x] **Multi-modal Fusion**: Pi3X accepts decoupled, optional depth and intrinsic conditioning (`--model_type pi3x --use_depth true --use_intrinsic true`) for improved absolute scale accuracy. See [Experiments](#-experiments) for benchmark results.


## 🚀 Quick Start

### 1. Clone & Install Dependencies
#### (1). Clone the Repository
```bash
git clone --recursive https://github.com/SCEIRobotics/L3ROcc.git
cd L3ROcc
```
#### (2). Install Python Dependencies
##### i. For Production (Generating OCC data for InternData-N1/LeRobot):
Python 3.10+ is recommended. Install the following dependencies:
```bash
conda create -n <env> python=3.10 -y
conda activate <env>
pip install -e .
pip install -e third_party/pi3
```
##### ii. For Visualization (Rendering dynamic videos & 3D inspection):
3D rendering requires a GUI environment. Please set up this environment on your local computer (Windows/macOS/Linux), not on the remote server：
```bash
# Run on your local machine
conda create -n <env> python=3.8 -y
conda activate <env>
pip install -r requirements_visual.txt
conda install -c conda-forge mayavi
```
(Note: Ensure you have a working OpenGL environment for Mayavi rendering.)

### 2. Model Checkpoints
This project supports two model variants. Place the weights under `ckpt/` at the project root:

```
ckpt/
├── pi3/                   # Base Pi3 weights (RGB-only)
│   ├── model.safetensors
│   └── config.json
└── pi3x/                  # Pi3X weights (RGB-only, or RGB + optional intrinsic / depth conditioning)
    ├── model.safetensors
    └── config.json
```

| Model | Model-level inputs | HuggingFace ID | Direct Download |
|-------|--------------------|----------------|-----------------|
| **Pi3** | RGB only | [`yyfz233/Pi3`](https://huggingface.co/yyfz233/Pi3) | [model.safetensors](https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors) |
| **Pi3X** | RGB + optional intrinsic + optional depth | [`yyfz233/Pi3X`](https://huggingface.co/yyfz233/Pi3X) | [model.safetensors](https://huggingface.co/yyfz233/Pi3X/resolve/main/model.safetensors) |

> **Note on calibrated intrinsics**: even when `--model_type pi3` is selected (Pi3.forward only consumes RGB), `--use_intrinsic true` is still honored by the **post-processing path** — the rescaled calibrated K replaces the model's back-calculated K in the saved Parquet column `observation.camera_intrinsic_occ`.

> If automatic download from Hugging Face is slow, use the direct download links above and place the files in the corresponding subdirectory.

### 3. Run Example （The pipeline supports three primary modes）:

The CLI exposes three independent input toggles that can be freely combined:

| Flag | Default | Effect |
|---|---|---|
| `--use_intrinsic true\|false` | `true` | Load calibrated K from `--condit_intr_path` (info.json). Used as Pi3X conditioning **and** to override the saved Parquet K. |
| `--use_depth true\|false` | `true` | Load depth from `--condit_depth_path` (mkv/mp4) as Pi3X conditioning. Silently ignored at the model layer when `--model_type pi3`. |
| `--model_type {pi3,pi3x}` | `pi3x` | Backbone selection. Replaces the previous `use_multimodal` flag. |

#### Mode A: Generate Visualized Dynamic Video
Use this to create side-by-side comparison videos from your own footage with history frames.

```bash
# Variant A1: Pi3X RGB-only (lightest, recommended)
python tools/run_normal_data_occ.py --video_path data/examples/office.mp4 \
    --save_dir data/examples/outputs/ \
    --model_type pi3x --use_depth false --use_intrinsic false \
    --pcd_save true --mode visual --mesh false

# Variant A2: Pi3X RGB + depth conditioning (best metric scale when depth is high-quality)
python tools/run_normal_data_occ.py --video_path data/examples/office.mp4 \
    --condit_depth_path data/examples/depth.mkv \
    --save_dir data/examples/outputs/ \
    --model_type pi3x --use_depth true --use_intrinsic false \
    --pcd_save true --mode visual --mesh false
```

#### Mode B: Generate LeRobot-compatible Data
Use this to generate the standard dataset structure for model training. The same three input toggles apply.

```bash
# Pi3X RGB-only (recommended default; intrinsic disabled per the scale experiment)
python tools/run_normal_data_occ.py --video_path data/examples/office.mp4 \
    --save_dir data/examples/outputs/ \
    --model_type pi3x --use_depth false --use_intrinsic false \
    --pcd_save true --mode run --mesh false
```

#### Mode C: Batch Process InternData-N1 / LeRobot Datasets
Processes a full InternData-N1 directory **or** LeRobot rosbag data (layout auto-detected),
GT-free. `--dataset_root` may point at a single rosbag, a parent of many `rosbag_*`, or an
InternData-N1 tree. The optional LeRobot z-axis deskew is enabled with `--use_z_deskew true`.

It supports **breakpoint resumption**:
- `mask_sequence.npz` (final result file) exists: Skip and continue with the next trajectory.
- `--overwrite true`: Force reprocessing and overwrite existing files.
- Otherwise: Proceed with processing.

```bash
# InternData-N1 / LeRobot, RGB-only (DLT-estimated K), no depth, no z-deskew
python tools/run_intern_nav_occ.py --dataset_root data/examples/small_vln_n1/traj_data \
    --output_root data/examples/small_vln_n1_4/traj_data \
    --model_type pi3x --use_intrinsic false --use_depth false \
    --use_z_deskew false --pcd_save true --overwrite false --mesh false

# LeRobot rosbag with z-axis deskew (writes per-frame R_deskew to episode_000000.parquet)
python tools/run_intern_nav_occ.py --dataset_root data/examples/lerobot/rosbag_xxx \
    --output_root data/examples/lerobot_out \
    --model_type pi3x --use_intrinsic false --use_depth false \
    --use_z_deskew true --pcd_save true --overwrite false --mesh false
```

The InternData-N1 generator now supports the same decoupled RGB / intrinsic / depth
inputs as Mode A/B. Intrinsic priority: `--condit_intr_path` (external info.json) >
`observation.camera_intrinsic` column in parquet > DLT fallback.

> **World-fusion visualization (`--save_world_fusion true`)**: additionally writes per-frame fused `(N, 7)` `[x, y, z, r, g, b, label]` arrays to `<trajectory>/merge_npy_sequence_world/frame_XXXX_world.npy` for `tools/visual/npy_to_world_video.py`. Each frame fuses three labeled blocks — background point cloud (`label 0`), camera trajectory up to that frame (`label 1`), and temporally accumulated visible occupancy (`label 2`) — all rigidly placed into the **same GT-free frame** as `observation.camera_extrinsic_occ` (model world re-expressed by the convention change `M = C`; N1 `diag(1,-1,-1)`, LeRobot identity), so the fused occupancy, trajectory, and point cloud all overlay consistently. It **reuses the in-memory OCC / point cloud / poses (no second inference)** and is only emitted alongside (re)generation — pass `--overwrite true` to regenerate the fusion for already-processed trajectories. Default `false`.

```bash
# InternData-N1, RGB-only, also emit world-fusion frames for npy_to_world_video.py
python tools/run_intern_nav_occ.py --dataset_root data/examples/small_vln_n1/traj_data \
    --output_root data/examples/small_vln_n1_4/traj_data \
    --model_type pi3x --use_intrinsic false --use_depth false \
    --use_z_deskew false --pcd_save true --overwrite true --mesh false \
    --save_world_fusion true
```

**Model + input matrix:**

| `--model_type` | `--use_intrinsic` | `--use_depth` | Description |
|---|---|---|---|
| `pi3` | `false` | `false` | Pi3 RGB-only (lightweight). DLT-estimated K saved to Parquet. |
| `pi3` | `true` | `false` | Pi3 RGB-only at the model layer, but the calibrated K still **overrides** the saved Parquet K (post-processing path). |
| `pi3x` | `false` | `false` | Pi3X RGB-only with metric head. |
| `pi3x` | `true` | `false` | Pi3X with intrinsic conditioning. Sharpens DLT K and unlocks Pi3X intrinsic-aware encoder path. |
| `pi3x` | `true` | `true` | Pi3X full multimodal (recommended when high-quality depth is available — best scale accuracy). |

> `--use_intrinsic` and `--use_depth` are independent toggles. Combinations like `pi3x` + `use_depth=true` + `use_intrinsic=false` are valid but rarely useful, because Pi3X internally derives camera rays from K — supplying depth without K forces the model to fall back to its ray prior. CLI prints `[Warning]` / `[Info]` hints to flag suboptimal combinations.


## 🛠️ Pipeline Details

### 1. Data Generators

Located in `L3ROcc/generater/`, the project includes two core generators. Both share the unified `model_type={"pi3","pi3x"}` constructor flag and the decoupled `(condit_depth_path, intrinsics_np)` keyword arguments on `run_pipeline` / `visual_pipeline` / `single_frame_pipeline`:

* **SimpleVideoDataGenerator**: Best for individual videos; automatically builds standard directory structures including `meta/`, `videos/`, and `data/`.
* **InternNavDataGenerator**: Handles both **InternData-N1** and **LeRobot rosbag** layouts (auto-detected). Uses GT-free metric scale (trusts `metric_head` + optional `metric_scale_correction`) and a per-dataset camera-convention change to emit **base-frame** occupancy; an optional LeRobot z-axis deskew is available. Per-trajectory intrinsics auto-load from the source Parquet `observation.camera_intrinsic` (else `meta/info.json` `head_camera_intrinsic`); depth videos are auto-discovered under `observation.video.depth/` or `observation.images.depth/`.

### 2. Core Configuration

Parameters can be tuned in `L3ROcc/configs/config.yaml`:

* **`pc_range`**: Spatial clipping and perception range `[x_min, y_min, z_min, x_max, y_max, z_max]` in the **canonical robot base frame** — forward = +y, lateral = ±x, up = +z, origin at the per-frame camera. A single box (forward depth 5.6 m, lateral ±2 m, height **[−1.8, 0.6] m**) serves all datasets because both are normalized into this one frame: N1 (OpenGL) folds `C = diag(1, -1, -1)` into `T_cam2base`, and LeRobot (OpenCV, ROS-style forward = +x) additionally folds a +90° base yaw `R_BASE_CANON_OPENCV` to map its forward +x → +y (see `compute_sequence_data`). The **z range is intentionally asymmetric — 1.8 m down, 0.6 m up**: `camera → base` is **rotation-only** (the camera stays at the base origin `z = 0`, see `convert_pointcloud_camera_to_base`; the ray origin in `check_visual_occ` is likewise `(0, 0, 0)`), and the head camera is mounted above the floor looking forward-down, so the ground lies *below* the camera at `z = −(camera height)`. Extending 1.8 m downward captures the floor for camera mount heights up to ~1.8 m (real trajectories range ≈ 0.4–1.3 m); a too-shallow lower bound (e.g. the previous `z_min = −0.6`) clips the ground out of the occupancy for any camera mounted higher than 0.6 m. This keeps `occ_size` identical across datasets so downstream consumers (visualization, training) need no per-dataset shape handling.
* **`voxel_size`**: Base size for occupancy voxels (default 0.04m), directly related to the sparsity of the occupancy voxel map.
* **`occ_size`**: Number of voxel grids in each spatial dimension, derived from `(pc_range_max - pc_range_min) / voxel_size` with no independent configuration.
* **`metric_scale_correction`**: GT-free metric scale factor applied to the Pi3X output. Default `1.0` (trust `metric_head`); set ~`1.08` to compensate the measured ~8% under-scale on real data.
* **`interval`**: Frame sampling interval for video processing.
* **`history_len`**: Number of past frames to include in history (default: 10).
* **`history_step`**: Step size for history frame sampling (default: 2).


### 3.Dataset Structure & Contents 

#### (1). InternData-N1 Format
The following structure is generated under each trajectory directory (e.g., trajectory_1) to ensure compatibility with robotics learning frameworks:

```
trajectory_1/
├── data/
│   └── chunk-000/                 # Core Geometric Assets
│       ├── all_occ.npz            # Global scene occupancy grid
│       ├── origin_pcd.ply         # Downsampled global point cloud
│       └── episode_000000.parquet # Per-frame poses and intrinsics
├── meta/                          # Metadata & Statistics
│   ├── info.json                  # Dataset schema and feature definitions
│   ├── episodes.jsonl             # Episode metadata and metric scale factor
│   ├── episodes_stats.jsonl       # Feature statistics (min/max/mean/std)
│   └── tasks.jsonl                # Task descriptions
└── videos/
    └── chunk-000/                  # Temporal Sequences
        ├── observation.occ.mask/
        │   └── mask_sequence.npz   # Temporal visibility bitmask
        ├── observation.occ.view/
        │   └── occ_sequence.npz    # Temporal egocentric occupancy
        ├── observation.video.trajectory/
        │   └── reference.mp4       # Original RGB source video (REQUIRED input)
        └── observation.video.depth/        # OPTIONAL: depth video for --use_depth true
            └── reference.mkv               # alt: observation.images.depth/*.{mkv,mp4}
```

> The depth sub-directory is **optional**. When `--use_depth true` is passed but no depth file is discovered under either `observation.video.depth/` or `observation.images.depth/`, the trajectory still processes — the run simply logs a warning and degrades to RGB-only at the model layer.

> **LeRobot output**: outputs land under `<output_root>/<rosbag_*>/<episode_id>/` with the same `data/` + `videos/observation.occ.*` layout. When `--use_z_deskew true`, a fresh `data/chunk-000/episode_000000.parquet` is written holding the per-frame **`R_deskew`** (3×3 applied deskew rotation `D_i`); restore an un-deskewed OCC frame via `P_uncorrected = D_iᵀ · P_corrected`.

##### i. data/chunk-000/ (Core Geometric Assets)
- **all_occ.npz**: Occupancy point cloud (key `data`, shape `(M, 3)` float32) — the **occupied voxel centers in metres**, deduplicated to the `voxel_size` grid and clipped to `pc_range`. Despite the name it is **not** a world-frame global grid: it is the full reconstructed scene voxelized and expressed in the **last frame's ego base frame** (origin at the last camera), so it covers only the perception box around the last camera. The per-dataset base axes follow `pc_range` (N1: x=lateral, y=forward+, z=up).
- **origin_pcd.ply**: The reconstructed global point cloud (voxel-downsampled) with RGB color, in the **π³ model world frame** (OpenCV, scaled by `metric_scale_correction`).
- **episode_000000.parquet**: A structured data table containing per-frame high-level features:
  - **`observation.camera_intrinsic_occ`**: 3x3 intrinsic matrix at the **model-input resolution**. Population depends on `--use_intrinsic`:
    - **`--use_intrinsic true`** (with a valid `--condit_intr_path` or per-trajectory parquet K): the calibrated K is rescaled to model input size and written here, overriding any model-side estimate. Works for both `pi3` and `pi3x` (post-processing path is backbone-agnostic).
    - **`--use_intrinsic false`** (or no calibration available): the K is back-estimated from local geometry via Least Squares / DLT on the model's `local_points`.
  - **`observation.camera_extrinsic_occ`**: 4x4 camera extrinsics (cam→world) predicted by the π³ backbone, **GT-free**. The model-world poses are kept as-is and only re-expressed under the per-dataset camera convention via a rigid coordinate-system change (basis-change conjugation `aligned = C · P · C`): InternData-N1 (OpenGL) applies `C = diag(1, -1, -1)`, LeRobot (OpenCV) leaves the poses unchanged (`C = I`). No frame-0 anchoring, no GT, no Sim3/Kabsch optimization — the model's own relative motion and GT-free metric scale (`metric_scale_correction`) are preserved. This keeps the field consistent with the rest of the GT-free pipeline (same frame as `origin_pcd.ply` and the occupancy).

##### ii. meta/ (Metadata & Statistics)
- **info.json**: Defines the dataset schema, including the data types and shapes for observation.camera_extrinsic_occ and observation.camera_intrinsic_occ.
- **episodes.jsonl**: Contains episode-level constants, most notably the metric scale factor (`metric_scale_correction`, GT-free) applied to the reconstruction.
- **episodes_stats.jsonl**: Automatically calculates the statistical distribution (min, max, mean, std) for all observation vectors.
- **tasks.jsonl**: Provides task descriptions and objectives for the dataset.

##### iii. videos/chunk-000/ (Temporal Sequences)
- **observation.occ.mask/mask_sequence.npz**: A time-series of visibility masks. It uses an optimized Bit-packing format to store which voxels are currently visible within the camera's frustum.
- **observation.occ.view/occ_sequence.npz**: A time-series of egocentric occupancy data. Each frame represents the occupied voxels in the **robot base (ego) coordinate system**, stored as a Sparse CSR Matrix to minimize storage overhead.
- **observation.video.trajectory/reference.mp4**: The original RGB video sequence used as input for reconstruction.

#### (2). Visual Format

Outputs generated by the `visual_pipeline` are tailored for rendering and manual inspection:

| Directory/File | Description |
|---------------|-------------|
| `merge_npy_sequence_cam.npy` | Files per frame in Camera Coordinates, merging initial PCD and visible OCC. |
| `merge_npy_sequence_world.npy` | Files per frame in World Coordinates, used for rendering dynamic fused videos. |
| `merge_ply_sequence_cam.ply` | Files per frame in Camera Coordinates for 3D inspection (e.g., MeshLab). |
| `merge_ply_sequence_world.ply` | Files per frame in World Coordinates for 3D inspection. |
| `occ_only_cam_npy.npy` | Files per frame containing only visible OCC in Camera Coordinates for rendering. |
| `occ_only_cam_ply.ply` | Files per frame containing only visible OCC for 3D inspection. |

## 🔬 Experiments

Two **GT-free** studies validate the reconstruction without any odometry/GT pose alignment. They
back the pipeline's choices: trust Pi3X's metric scale, and rigidly place the output in the base
frame (with an optional z-deskew).

### 1. Metric Scale Accuracy (RGB-only metric head)
Measures whether Pi3X's `metric_head` absolute scale is directly usable, against **independent**
references (16-bit sensor depth, known physical sizes) — never GT odometry.

- On real (LeRobot / ZED) data the reconstruction-to-sensor depth ratio `pred/sensor` median ≈ **0.92**
  (~8% under-scale, i.e. quite accurate). The internal `metric` scalar (≈0.3 across datasets) is an
  internal multiplier, **not** a real-world scale indicator.
- Depth conditioning helps only marginally and adds variance, so **RGB-only metric head is the
  reliable default**. Optionally set `metric_scale_correction ≈ 1.08` to compensate the ~8%.

> Methodology & usage: [`tools/exp_metric_scale/README.md`](tools/exp_metric_scale/README.md)

### 2. Coordinate Alignment (rigid, GT-free)
Overlays the Pi3X point cloud / trajectory and the N1 GT trajectory in a common **base frame** with
no optimization-based alignment, to measure Pi3X's true scale / orientation / drift gap.

- Pi3X base-frame output **rigidly aligns** to the N1 base frame; trajectory length ratio ≈ 1 confirms
  the metric scale is accurate. The camera-convention change is **per-dataset**: InternData-N1
  extrinsics are OpenGL and need `C = diag(1, -1, -1)`; LeRobot hand-eye is OpenCV and needs identity.
- Real LeRobot reconstructions show a slight base-frame ground tilt (~**4.7°**), removed by the z-axis
  deskew (frame-0 gravity fold + per-frame leveling) under `--use_z_deskew true`.

> Methodology & usage: [`tools/exp_coord_align/README.md`](tools/exp_coord_align/README.md)

---

## 📺 Visualization & Toolbox

A variety of scripts are provided in `tools/visual/` for visualization and analysis:

| Script | Description |
|--------|-------------|
| `visual_simple_frame_npy.py` | Interactive single-frame debugger. Loads individual voxel .npy files, supports interactive view rotation in Mayavi, and prints real-time camera pose parameters (Position/Focal/ViewUp) to determine the optimal fixed view for video rendering. |
| `visual_simple_frame_npz.py` | Fast sparse matrix viewer. Directly reads compressed .npz  or .npy files to quickly verify the integrity of generated occupancy data without decompressing the entire sequence. |
| `npy_to_world_video.py` | God's eye (World-View) fusion rendering. Generates third-person global reconstruction videos containing three key elements: true-color background point clouds, global camera trajectories , and accumulated occupancy grids . Input frames come from `visual_pipeline` (`merge_npy_sequence_world.npy`) or, for InternData-N1 / LeRobot batch runs, from `--save_world_fusion true` (`merge_npy_sequence_world/frame_XXXX_world.npy`). |
| `npy_to_occ_video.py` | Egocentric (First-Person) stylized rendering. Generates first-person videos with only local occupancy, using Morandi color palette for depth-gradient shading to showcase pure spatial geometric structures. |
| `video_composer_to_3.py` | 3-Panel panoramic composer. Horizontally stitches three video streams to generate the final demo video, typically including: original RGB input video, world-view fusion video, and local occupancy video. |

## 🙏 Acknowledgements

This project is built upon the following excellent works:

  * [π³](https://github.com/yyfz/Pi3)
  * [Occ3D](https://arxiv.org/pdf/2304.14365)
  * [SurroundOcc](https://github.com/weiyithu/SurroundOcc)
  * [InternData-N1](https://huggingface.co/datasets/InternRobotics/InternData-N1)

## 🐼 Core Contributors

**Nianjin Ye**<sup>1*</sup>([GitHub](https://github.com/CallMeFrozenBanana)), **Binling Huang**<sup>12*</sup>([GitHub](https://github.com/hbl-0624)),**Xi Yang**<sup>1</sup>([GitHub](https://github.com/kingkids)),**Hao Xu**<sup>3</sup>([GitHub](https://hxwork.github.io/))

<sup>1</sup>Sichuan Embodied Intelligence Robot Training Base     <sup>2</sup>UESTC     <sup>3</sup>CUHK     <sup>*</sup> (Equal Contribution)

## 📜 Citation

If you find this project useful for your research, please consider citing the foundational works mentioned in the **Acknowledgements**:

```bibtex
@misc{wang2025pi3,
      title={$\pi^3$: Scalable Permutation-Equivariant Visual Geometry Learning}, 
      author={Yifan Wang and Jianjun Zhou and Haoyi Zhu and Wenzheng Chang and Yang Zhou and Zizun Li and Junyi Chen and Jiangmiao Pang and Chunhua Shen and Tong He},
      year={2025},
      eprint={2507.13347},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.13347}, 
}

@article{tian2023occ3d,
  title={Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving},
  author={Tian, Xiaoyu and Jiang, Tao and Yun, Longfei and Wang, Yue and Wang, Yilun and Zhao, Hang},
  journal={arXiv preprint arXiv:2304.14365},
  year={2023}
}

@article{wei2023surroundocc, 
      title={SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving}, 
      author={Yi Wei and Linqing Zhao and Wenzhao Zheng and Zheng Zhu and Jie Zhou and Jiwen Lu},
      journal={arXiv preprint arXiv:2303.09551},
      year={2023}
}

@misc{interndata_n1,
  title={InternData-N1 Dataset},
  author={InternData-N1 Dataset contributors},
  howpublished={\url{https://huggingface.co/datasets/InternRobotics/InternData-N1}},
  year={2025}
}
```


## 📄 License
For academic use, this project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details. For commercial use, please contact the authors.
