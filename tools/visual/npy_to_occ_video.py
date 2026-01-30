import glob
import os
import re
import sys
from argparse import ArgumentParser

import cv2
import mayavi.mlab as mlab
import numpy as np
import yaml

# This script converts a sequence of occ_only_cam.npy files into a video.



def create_low_sat_gradient():
    """
    Creates a 256-color low-saturation gradient lookup table (LUT)
    transitioning from a 'near' color to a 'far' color.
    """
    # 1. Define colors (RGB 0-1) - Morandi palette
    # Near: Soft hazy white
    color_near = np.array([0.85, 0.83, 0.80, 1.0])
    # Far: Deep grey-blue
    color_far = np.array([0.25, 0.30, 0.35, 1.0])

    n_bins = 256
    custom_lut = np.zeros((n_bins, 4))

    for i in range(n_bins):
        ratio = i / float(n_bins - 1)
        custom_lut[i] = color_near * (1 - ratio) + color_far * ratio

    return (custom_lut * 255).astype(np.uint8)


LOW_SAT_LUT = create_low_sat_gradient()
# ==========================================


# Numerical sorting helper
def numerical_sort(value):
    numbers = re.compile(r"(\d+)")
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


# Convert voxel grid to point cloud coordinates
def voxel2points(pred_occ, mask_camera=None, free_label=0):
    x = np.linspace(0, pred_occ.shape[0] - 1, pred_occ.shape[0])
    y = np.linspace(0, pred_occ.shape[1] - 1, pred_occ.shape[1])
    z = np.linspace(0, pred_occ.shape[2] - 1, pred_occ.shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    vv = np.stack([X, Y, Z, pred_occ], axis=-1)
    valid_mask = pred_occ != free_label
    if mask_camera is not None:
        valid_mask = np.logical_and(valid_mask, mask_camera)
    fov_voxels = vv[valid_mask].astype(np.float32)
    return fov_voxels


if __name__ == "__main__":
    mlab.options.offscreen = True

    parse = ArgumentParser()
    parse.add_argument(
        "--input_dir",
        type=str,
        default="/outputs/office_1/occ_only_npy",
    )
    parse.add_argument(
        "--output_video",
        type=str,
        default="/outputs/office_1/occ_only.mp4",
    )
    parse.add_argument(
        "--config",
        type=str,
        default="/L3ROcc/configs/config.yaml",
    )

    args = parse.parse_args()
    input_dir = args.input_dir
    output_video = args.output_video
    config_path = args.config

    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    voxel_size = config.get("voxel_size", 0.05)

    files = sorted(glob.glob(os.path.join(input_dir, "*.npy")), key=numerical_sort)
    if not files:
        print(f"Error: No .npy files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(files)} frames. Generating video...")

    width, height = 800, 800
    figure = mlab.figure(size=(width, height), bgcolor=(1, 1, 1))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, 30.0, (width, height))

    # Camera position constant (used for distance calculation)
    CAM_POS_ARRAY = np.array(
        [-0.2231539051005686, -0.7217984195639319, -3.1510481045664047]
    )

    for i, file_path in enumerate(files):
        mlab.clf()

        if file_path.endswith("npz"):
            data = np.load(file_path)
            semantics = data["semantics"]
            fov_voxels = voxel2points(semantics)
        else:
            fov_voxels = np.load(file_path).astype(np.float32)

        if len(fov_voxels.shape) == 3:
            fov_voxels = voxel2points(fov_voxels)

        if len(fov_voxels.shape) == 2 and fov_voxels.shape[1] == 4:
            fov_voxels = fov_voxels[fov_voxels[..., 3] >= 0]

        if len(fov_voxels) == 0:
            print(f"Frame {i} is empty.")
            img_array = mlab.screenshot(figure=figure, mode="rgb", antialiased=True)
            writer.write(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            continue

        # fov_voxels[:, 0] = -fov_voxels[:, 0]

        # 1. Compute distance (used for coloring)
        dist_values = np.sqrt(
            (fov_voxels[:, 0] - CAM_POS_ARRAY[0]) ** 2
            + (fov_voxels[:, 1] - CAM_POS_ARRAY[1]) ** 2
            + (fov_voxels[:, 2] - CAM_POS_ARRAY[2]) ** 2
        )

        # 2. Render point cloud
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0],
            fov_voxels[:, 1],
            fov_voxels[:, 2],
            dist_values,
            scale_factor=voxel_size - 0.05 * voxel_size,
            mode="cube",
            opacity=1.0,
        )

        # [CRITICAL FIX]
        # Force disable data scaling to ensure all voxels remain uniform in size
        plt_plot_fov.glyph.scale_mode = "data_scaling_off"

        # Apply custom color lookup table (LUT)
        plt_plot_fov.module_manager.scalar_lut_manager.lut.table = LOW_SAT_LUT

        # 3. Setup camera: Optimal viewpoint parameters obtained from visual.py for the first frame
        cam = figure.scene.camera
        cam.position = [-0.2231539051005686, -0.7217984195639319, -3.1510481045664047]
        cam.focal_point = [
            0.029999852180480957,
            -0.30000001192092896,
            1.9500000774860382,
        ]
        cam.view_angle = 30.0
        cam.view_up = [0.004228976510311452, -0.9966070345601077, 0.08219814123800787]
        cam.clipping_range = [1.401581949682037, 9.823731225875328]
        cam.compute_view_plane_normal()
        figure.scene.render()

        img_array = mlab.screenshot(figure=figure, mode="rgb", antialiased=True)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        writer.write(img_bgr)

        if i % 10 == 0:
            print(f"Processed {i}/{len(files)}")

    writer.release()
    mlab.close(all=True)
    print(f"Video saved to: {output_video}")
