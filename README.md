<h1 align="center">🌌 OccGen: Scalable 4D Occupancy Generation Pipeline</h1>

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
  <img src="assets/demo_V2.gif" width="100%" alt="OccGen Demo">
  <p><i>Left: RGB Input | Middle: 3D Point Cloud Fusion | Right: 4D Occupancy Grid</i></p>
</div>

`OccGen` is a high-performance visual geometry framework designed to transform standard RGB video sequences into high-precision **3D Point Clouds**, **3D Occupancy Grids**, and **4D Temporal Observation Data**.
The project utilizes **$\pi^3$ (Permutation-Equivariant Visual Geometry Learning)** as its underlying reconstruction engine and provides a fully automated data labeling and alignment pipeline tailored for robotics learning formats such as InternNav and LeRobot.

## ✨ Key Features
* **End-to-End Reconstruction**: Directly predicts camera poses, depth maps, and globally consistent point clouds from RGB video streams.
* **Automated Voxelization**: Converts unstructured point clouds into structured Occupancy Grids with dynamic voxel size calculation based on scene volume.
* **Visibility Analysis (Ray Casting)**: Performs real-time ray casting based on camera poses to compute visible regions (Visible Masks) and occlusion relationships.
* **4D Data Serialization**:
    * **Sparse OCC**: Utilizes Sparse CSR matrices to store temporal occupancy, significantly reducing disk usage.
    * **Packed Mask**: Implements bit-packing (via `np.packbits`) for visibility masks to optimize storage efficiency.
* **Multi-Dataset Adaptation**: Built-in generators for both `SimpleVideo` (single video) and `InternNav` (large-scale datasets).
* **Professional Visualization**: Mayavi-based 3D rendering tools for generating side-by-side comparison videos of point clouds, trajectories, and occupancy.


## 🚀 Quick Start

### 1. Clone & Install Dependencies
#### (1). Clone the Repository
```bash
git clone https://github.com/CallMeFrozenBanana/occgen.git
cd occgen
```
#### (2). Install Python Dependencies
##### i. For Production (Generating OCC data for InternNav/LeRobot):
```bash
pip install -r requirements.txt
```
##### ii. For Visualization (Rendering dynamic videos & 3D inspection):
Python 3.8+ is recommended. Install the following dependencies:
```bash
pip install -r requirements_visual.txt
```
(Note: Ensure you have a working OpenGL environment for Mayavi rendering.)

### 2.Model Checkpoints
Place the $\pi^3$ model weights (model.safetensors) and configuration files in the ckpt/ directory at the project root. If the automatic download from Hugging Face is slow, you can download the model checkpoint manually from [here](https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors).

### 3. Run Example （The pipeline supports three primary modes）:
#### Mode A: Generate Visualized Dynamic Video Use this to create side-by-side comparison videos from your own footage with history frames.
```bash
# Ensure you uncomment generator.visual_pipeline in the script
python tools/run_normal_data_occ.py
```
#### Mode B: Generate LeRobot-compatible OCC Data Use this to generate the standard dataset structure for model training.
```bash
# Ensure you uncomment generator.run_pipeline in the script
python tools/run_normal_data_occ.py
```
#### Mode C: Batch Process InternNav Dataset To process the full InternNav directory with scale alignment enabled.
```bash
python tools/run_intern_nav_occ.py 
```

## 🛠️ Pipeline Details

### 1. Data Generators

Located in `occ/generater/`, the project includes two core generators:

* **SimpleVideoDataGenerator**: Best for individual videos; automatically builds standard directory structures including `meta/`, `videos/`, and `data/`.
* **InternNavDataGenerator**: Designed for large-scale InternNav data enhancement; supports **Scale Alignment** using Sim3 to ensure reconstruction coordinates match ground truth.

### 2. Core Configuration

Parameters can be tuned in `occ/configs/config.yaml`:

* **`voxel_size`**: Base size for occupancy voxels (e.g., 0.02m).
* **`pc_range`**: Spatial clipping and perception range `[x_min, y_min, z_min, x_max, y_max, z_max]`.
* **`interval`**: Frame sampling interval for video processing.
* **`history_len`**: Number of past frames to include in history (default: 10).
* **`history_step`**: Step size for history frame sampling (default: 2).


### 3.Dataset Structure & Contents 

#### (1). InternNav Format
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
│   ├── episodes.jsonl             # Episode metadata and Sim3 scale factors
│   ├── episodes_stats.jsonl       # Feature statistics (min/max/mean/std)
│   └── tasks.jsonl                # Task descriptions
└── videos/
    └── chunk-000/                 # Temporal Sequences
        ├── observation.occ.mask/
        │   └── mask_sequence.npz  # Temporal visibility bitmask
        ├── observation.occ.view/
        │   └── occ_sequence.npz   # Temporal egocentric occupancy
        └── observation.video.trajectory/
            └── reference.mp4      # Original RGB source video
```

##### i. data/chunk-000/ (Core Geometric Assets)
- **all_occ.npz**: Stores the global occupancy grid of the entire scene in world coordinates.
- **origin_pcd.ply**: The initial global point cloud reconstructed from the video, optimized via voxel downsampling for efficient processing.
- **episode_000000.parquet**: A structured data table containing per-frame high-level features:
  - **Camera Intrinsics_occ**: 3x3 matrices re-estimated via Least Squares/DLT based on local geometry.
  - **Camera Extrinsics_occ**: 4x4 extrinsic matrices predicted by the π³ model and aligned to world coordinates.

##### ii. meta/ (Metadata & Statistics)
- **info.json**: Defines the dataset schema, including the data types and shapes for observation.camera_extrinsic_occ and observation.camera_intrinsic_occ.
- **episodes.jsonl**: Contains episode-level constants, most notably the Sim3 Scale Factor used to align the model's relative units to real-world metric scales.
- **episodes_stats.jsonl**: Automatically calculates the statistical distribution (min, max, mean, std) for all observation vectors.
- **tasks.jsonl**: Provides task descriptions and objectives for the dataset.

##### iii. videos/chunk-000/ (Temporal Sequences)
- **observation.occ.mask/mask_sequence.npz**: A time-series of visibility masks. It uses an optimized Bit-packing format to store which voxels are currently visible within the camera's frustum.
- **observation.occ.view/occ_sequence.npz**: A time-series of egocentric occupancy data. Each frame represents the occupied voxels in the current camera coordinate system, stored as a Sparse CSR Matrix to minimize storage overhead.
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

## 📺 Visualization & Toolbox

A variety of scripts are provided in `utils/visual/` for visualization and analysis:

| Script | Description |
|--------|-------------|
| `visual_simple_frame_npy.py` | Interactive OCC viewer with real-time Mayavi camera parameter extraction. |
| `npy_to_world_video.py` | Generates global-view videos featuring true-color point clouds, trajectories, and OCC. |
| `npy_to_occ_video.py` | Generates egocentric (first-person) occupancy rendering videos. |
| `video_composer_to_3.py` | Composes a 3-panel video (Original vs. World Fusion vs. Pure OCC). |

## 🙏 Acknowledgements

This project is built upon the following excellent works:

  * [π³](https://github.com/yyfz/Pi3)
  * [CUT3R](https://github.com/CUT3R/CUT3R)
  * [DUSt3R](https://github.com/naver/dust3r)

## 💡 Core Contributors

* **Nianjing Ye**<sup>1*</sup> ([GitHub](https://github.com/CallMeFrozenBanana))

* **Binling Huang**<sup>12*</sup> ([GitHub](https://github.com/hbl-0624))

<sup>1</sup>ChangHong Robotics     <sup>2</sup>UESTC     (<sup>*</sup> Equal Contribution)

## 📜 Citation

If you find our work useful, please consider citing:

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
```


## 📄 License
For academic use, this project is licensed under the 2-clause BSD License. See the [LICENSE](./LICENSE) file for details. For commercial use, please contact the authors.