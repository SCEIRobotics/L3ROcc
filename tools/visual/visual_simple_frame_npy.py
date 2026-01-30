import argparse
import os
import sys

import mayavi.mlab as mlab
import numpy as np
import yaml

# Add current directory and parent directory to sys.path for local module resolution
sys.path.append(".")
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from L3ROcc.configs.globals import get_global

# ==============================================================================
# Script Description
# ==============================================================================
# This script is designed for visualizing .npy voxel files.
# Requirement: Run locally using Python 3.8 (Mayavi dependency).
# Primary Use Case: Interactively adjusting the 3D view to extract the optimal
# camera perspective parameters (Position, Focal Point, View Angle) for the first frame.
# ==============================================================================

# Initialize Global Color Map
LABEL_COLORS = get_global("LABEL_COLORS") * 255
# Add Alpha channel (Opacity) set to 255 (Fully Opaque)
alpha = np.ones((LABEL_COLORS.shape[0], 1)) * 255
LABEL_COLORS = np.concatenate((LABEL_COLORS, alpha), axis=1)
LABEL_COLORS = LABEL_COLORS.astype(np.uint8)
FREE_LABEL = len(LABEL_COLORS)


def voxel2points(pred_occ, mask_camera=None, free_label=0):
    """
    Converts a voxel grid into a point cloud format.

    Args:
        pred_occ: The predicted occupancy grid.
        mask_camera: Optional boolean mask for the camera field of view.
        free_label: The label index representing empty space (to be filtered out).

    Returns:
        fov_voxels: Filtered point cloud data (N, 4) -> [x, y, z, label].
    """
    # Generate grid indices
    x = np.linspace(0, pred_occ.shape[0] - 1, pred_occ.shape[0])
    y = np.linspace(0, pred_occ.shape[1] - 1, pred_occ.shape[1])
    z = np.linspace(0, pred_occ.shape[2] - 1, pred_occ.shape[2])

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Stack coordinates with label data
    vv = np.stack([X, Y, Z, pred_occ], axis=-1)

    # Filter valid voxels (occupied space)
    valid_mask = pred_occ != free_label

    # Apply camera mask if provided
    if mask_camera is not None:
        valid_mask = np.logical_and(valid_mask, mask_camera)

    fov_voxels = vv[valid_mask].astype(np.float32)

    return fov_voxels


if __name__ == "__main__":
    # Toggle offscreen rendering (Set to False for interactive mode)
    offscreen = False
    mlab.options.offscreen = offscreen

    # ==========================================================================
    # 1. Configuration & Argument Parsing
    # ==========================================================================
    parse = ArgumentParser()
    parse.add_argument(
        "--visual_path",
        type=str,
        default="/outputs/occ.npy",
    )
    parse.add_argument(
        "--visual_save_dir",
        type=str,
        default="./outputs/visual",
    )
    args = parse.parse_args()

    visual_path = args.visual_path
    visual_save_dir = args.visual_save_dir
    config_path = "./L3ROcc/configs/config.yaml"

    # Load configuration
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    voxel_size = config.get("voxel_size", 0.2)
    pc_range = config.get("pc_range", [])
    occ_size = config.get("occ_size", [])

    # ==========================================================================
    # 2. Data Loading & Preprocessing
    # ==========================================================================
    if visual_path.endswith("npz"):
        data = np.load(visual_path)
        # Handle NPZ dictionary structure
        mask_camera = data["mask_camera"] if "mask_camera" in data else None
        semantics = data["semantics"]
        fov_voxels = voxel2points(semantics)
    else:
        # Handle standard NPY array
        fov_voxels = np.load(visual_path).astype(np.float32)

    # Ensure data is in point cloud format (N, 4)
    if len(fov_voxels.shape) == 3:
        fov_voxels = voxel2points(fov_voxels)

    # Filter valid labels if shape is (N, 4)
    if fov_voxels.shape[1] == 4:
        fov_voxels = fov_voxels[fov_voxels[..., 3] >= 0]

    # [Optional] Coordinate Denormalization (Currently Disabled)
    # Uncomment to convert voxel indices back to world coordinates
    # fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
    # fov_voxels[:, 0] += pc_range[0]
    # fov_voxels[:, 1] += pc_range[1]
    # fov_voxels[:, 2] += pc_range[2]

    # ==========================================================================
    # 3. Visualization
    # ==========================================================================
    figure = mlab.figure(size=(600, 600), bgcolor=(1, 1, 1))

    # Clamp labels to ensure they fit within the color map range
    fov_voxels[fov_voxels[:, 3] >= len(LABEL_COLORS), 3] = len(LABEL_COLORS) - 1

    # Render 3D points
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],  # Scalars for color mapping
        colormap="viridis",
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=len(LABEL_COLORS) - 1,
    )

    # Apply custom color lookup table (LUT)
    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = LABEL_COLORS

    if offscreen:
        if not os.path.exists(visual_save_dir):
            os.makedirs(visual_save_dir)
        save_path = os.path.join(visual_save_dir, "occ_visual.png")
        mlab.savefig(save_path)
        print(f"Saved snapshot to {save_path}")
    else:
        print("Rendering scene... Please adjust the view to the desired angle.")
        print("Close the window to output the camera parameters.")
        mlab.show()

        # ======================================================================
        # 4. Camera Parameter Extraction
        # ======================================================================
        # Output parameters after user interaction for use in other scripts
        cam = figure.scene.camera
        print("\n" + "=" * 50)
        print(" Captured Camera Parameters ")
        print("=" * 50)
        print(f"cam.position = {tuple(cam.position)}")
        print(f"cam.focal_point = {tuple(cam.focal_point)}")
        print(f"cam.view_angle = {cam.view_angle}")
        print(f"cam.view_up = {tuple(cam.view_up)}")
        print(f"cam.clipping_range = {tuple(cam.clipping_range)}")
        print("=" * 50 + "\n")
