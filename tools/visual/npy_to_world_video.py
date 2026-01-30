import glob
import os
import re
import sys
from argparse import ArgumentParser

import cv2
import mayavi.mlab as mlab
import numpy as np
import yaml

# This script generates a video from .npy files containing the initial point cloud,
# Occupancy (OCC), and camera trajectory in the world coordinate system.


def numerical_sort(value):
    numbers = re.compile(r"(\d+)")
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


if __name__ == "__main__":
    mlab.options.offscreen = True

    parse = ArgumentParser()
    # Path to 'npy_sequence_world' directory (Expected data shape: N x 7)
    parse.add_argument(
        "--input_dir",
        type=str,
        default="/outputs/office_1/npy_sequence_world",
    )
    # Output path (Rename to prevent overwriting existing files)
    parse.add_argument(
        "--output_video",
        type=str,
        default="/outputs/office_1/real_color_world.mp4",
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

    voxel_size = 0.05
    if os.path.exists(config_path):
        with open(config_path, "r") as stream:
            config = yaml.safe_load(stream)
        voxel_size = config.get("voxel_size", 0.05)

    files = sorted(glob.glob(os.path.join(input_dir, "*.npy")), key=numerical_sort)
    if not files:
        print(f"Error: No .npy files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(files)} frames. Generating Real-Color video...")

    width, height = 800, 800
    figure = mlab.figure(size=(width, height), bgcolor=(1, 1, 1))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, 30.0, (width, height))

    for i, file_path in enumerate(files):
        mlab.clf()

        try:
            data = np.load(file_path)  # Shape: (N, 7) -> [x, y, z, r, g, b, label]
        except:
            continue

        if data.shape[1] < 7:
            print(
                f"Error: Frame {i} has shape {data.shape}, expected (N, 7). Did you run the new Section D code?"
            )
            continue

        # 1. Data Splitting
        # The label is located in the 6th column (index 6)
        mask_bg = data[:, 6] == 0
        mask_traj = data[:, 6] == 1
        mask_occ = data[:, 6] == 2

        # Background data: XYZ coordinates and RGB color
        bg_xyz = data[mask_bg, :3]
        bg_rgb = data[mask_bg, 3:6]  # R, G, B in [0, 1]

        pts_traj = data[mask_traj, :3]
        pts_occ = data[mask_occ, :3]

        # 2. Render Background (Supports True Color)
        if len(bg_xyz) > 0:
            N = len(bg_xyz)
            # Create scalar indices from 0 to N-1
            scalars = np.arange(N)

            # Render points using '2dvertex' mode
            pts = mlab.points3d(
                bg_xyz[:, 0],
                bg_xyz[:, 1],
                bg_xyz[:, 2],
                scalars,  # Pass indices
                mode="2dvertex",
                scale_factor=0.03,  # Placeholder scale
            )
            # Create a Lookup Table (LUT) of shape (N, 4) uint8, including the Alpha channel
            lut = np.zeros((N, 4), dtype=np.uint8)
            lut[:, :3] = (bg_rgb * 255).astype(np.uint8)  # Fill RGB values
            lut[:, 3] = 255  # Alpha = 255 (Opaque)

            # Force assignment of this custom LUT to the Mayavi object
            pts.module_manager.scalar_lut_manager.lut.number_of_colors = N
            pts.module_manager.scalar_lut_manager.lut.table = lut

            # Disable auto-scaling to prevent color misalignment
            pts.glyph.scale_mode = "scale_by_vector"

        # 3. Render OCC (Label 2) -> Dark Gray Cubes
        if len(pts_occ) > 0:
            occ_plot = mlab.points3d(
                pts_occ[:, 0],
                pts_occ[:, 1],
                pts_occ[:, 2],
                mode="cube",
                color=(0.4, 0.4, 0.4),  # Dark gray
                scale_factor=voxel_size - 0.005,
                opacity=1.0,
            )
            occ_plot.glyph.scale_mode = "data_scaling_off"

        # 4. Render Trajectory (Label 1) -> Blue Path + Red Head
        if len(pts_traj) > 0:
            # Historical trajectory
            if len(pts_traj) > 1:
                hist_traj = pts_traj[:-1]
                mlab.points3d(
                    hist_traj[:, 0],
                    hist_traj[:, 1],
                    hist_traj[:, 2],
                    mode="sphere",
                    color=(0.0, 0.0, 1.0),  # Pure blue
                    scale_factor=0.025,
                )

            # Current agent position (Head)
            curr_pos = pts_traj[-1]
            mlab.points3d(
                curr_pos[0],
                curr_pos[1],
                curr_pos[2],
                mode="sphere",
                color=(1.0, 0.0, 0.0),  # Pure red
                scale_factor=0.05,
            )

        # 5. Camera Configuration
        # Note: View parameters are obtained via visual_sample_frame.py
        # -------------------------------------------------
        cam = figure.scene.camera
        cam.position = [-5.682662716850507, -2.3581944446045062, 1.1414235902534458]
        cam.focal_point = [0.3545984131007137, 0.24273155461184365, -0.1329467104813029]
        cam.view_angle = 30.0
        cam.view_up = [0.24965504712039718, -0.826617329042394, -0.5043571638969063]
        cam.clipping_range = [1.851463337952073, 12.81058369015543]
        cam.compute_view_plane_normal()

        figure.scene.render()

        img_array = mlab.screenshot(figure=figure, mode="rgb", antialiased=True)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        writer.write(img_bgr)

        if i % 10 == 0:
            print(f"Processed {i}/{len(files)}")

    writer.release()
    mlab.close(all=True)
    print(f"Real-Color Fused Video saved to: {output_video}")
