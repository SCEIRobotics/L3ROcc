import glob
import os
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import yaml

# ==============================================================================
# Interactive viewer for world-frame fusion .npy frames
# (merge_npy_sequence_world/frame_*.npy, shape (N,7) = [x, y, z, r, g, b, label]).
#
# It renders EXACTLY like tools/visual/npy_to_world_video.py (true-color
# background, dark-gray OCC cubes, blue trajectory + red head) and uses the same
# metric world coordinates, so a camera picked here transfers faithfully to the
# video — unlike the voxel-index viewers (visual_simple_frame_npz/npy.py).
#
# Controls:  M  print MANUAL_CAM block   |   N/B  next/prev frame
#            Home/End  first/last frame  |   mouse/arrows  rotate & zoom
# ==============================================================================


def numerical_sort(value):
    numbers = re.compile(r"(\d+)")
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def render_world_frame(data, voxel_size):
    """Render one (N,7) world frame with the same styling as the video script."""
    mask_bg = data[:, 6] == 0
    mask_traj = data[:, 6] == 1
    mask_occ = data[:, 6] == 2

    bg_xyz = data[mask_bg, :3]
    bg_rgb = data[mask_bg, 3:6]  # R, G, B in [0, 1]
    pts_traj = data[mask_traj, :3]
    pts_occ = data[mask_occ, :3]

    # Background (true color via a per-point LUT)
    if len(bg_xyz) > 0:
        N = len(bg_xyz)
        scalars = np.arange(N)
        pts = mlab.points3d(
            bg_xyz[:, 0], bg_xyz[:, 1], bg_xyz[:, 2], scalars,
            mode="2dvertex", scale_factor=0.03,
        )
        lut = np.zeros((N, 4), dtype=np.uint8)
        lut[:, :3] = (bg_rgb * 255).astype(np.uint8)
        lut[:, 3] = 255
        pts.module_manager.scalar_lut_manager.lut.number_of_colors = N
        pts.module_manager.scalar_lut_manager.lut.table = lut
        pts.glyph.scale_mode = "scale_by_vector"

    # OCC (label 2) -> dark gray cubes
    if len(pts_occ) > 0:
        occ_plot = mlab.points3d(
            pts_occ[:, 0], pts_occ[:, 1], pts_occ[:, 2],
            mode="cube", color=(0.4, 0.4, 0.4),
            scale_factor=voxel_size - 0.005, opacity=1.0,
        )
        occ_plot.glyph.scale_mode = "data_scaling_off"

    # Trajectory (label 1) -> blue history + red head
    if len(pts_traj) > 0:
        if len(pts_traj) > 1:
            hist = pts_traj[:-1]
            mlab.points3d(
                hist[:, 0], hist[:, 1], hist[:, 2],
                mode="sphere", color=(0.0, 0.0, 1.0), scale_factor=0.025,
            )
        curr = pts_traj[-1]
        mlab.points3d(
            curr[0], curr[1], curr[2],
            mode="sphere", color=(1.0, 0.0, 0.0), scale_factor=0.05,
        )


if __name__ == "__main__":
    parse = ArgumentParser()
    default_input_dir = str(Path('/path/to/merge_npy_sequence_world'))
    parse.add_argument("--input_dir", type=str, default=default_input_dir)
    parse.add_argument("--config", type=str, default="/L3ROcc/configs/config.yaml")
    parse.add_argument(
        "--frame", type=int, default=-1,
        help="Frame index to show first; -1 = last frame (full trajectory).",
    )
    args = parse.parse_args()

    voxel_size = 0.05
    if os.path.exists(args.config):
        with open(args.config, "r") as stream:
            config = yaml.safe_load(stream)
        voxel_size = config.get("voxel_size", 0.05)

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")), key=numerical_sort)
    if not files:
        print(f"Error: No .npy files found in {args.input_dir}")
        sys.exit(1)
    total_frames = len(files)
    print(f"Found {total_frames} world frames in {args.input_dir}")

    figure = mlab.figure(size=(900, 800), bgcolor=(1, 1, 1))
    start = args.frame if args.frame >= 0 else total_frames - 1
    state = {
        "frame": max(0, min(start, total_frames - 1)),
        "initialized": False,
        "rendering": False,   # guard against re-entrant key-repeat events
    }

    def load_frame(idx):
        try:
            data = np.load(files[idx])
        except Exception as e:
            print(f"Frame {idx}: load failed: {e}")
            return None
        if data.ndim != 2 or data.shape[1] < 7:
            print(f"Frame {idx}: unexpected shape {getattr(data, 'shape', None)}")
            return None
        return data

    def render_frame(idx):
        # Preserve the user's viewpoint across frame changes (after first render).
        if state["initialized"]:
            cam = figure.scene.camera
            saved = (cam.position, cam.focal_point, cam.view_up,
                     cam.clipping_range, cam.view_angle)

        mlab.clf()
        data = load_frame(idx)
        if data is not None:
            render_world_frame(data, voxel_size)

        mlab.text(
            0.62, 0.90,
            "N  Next frame     B  Prev frame\n"
            "Home First        End  Last\n"
            "M  Print camera params\n"
            "Arrows / Mouse: rotate view",
            width=0.36,
        )
        mlab.text(0.01, 0.01, f"Frame: {idx} / {total_frames - 1}", width=0.25)

        if state["initialized"]:
            cam = figure.scene.camera
            (cam.position, cam.focal_point, cam.view_up,
             cam.clipping_range, cam.view_angle) = saved
            cam.compute_view_plane_normal()
        figure.scene.render()
        state["initialized"] = True
        state["rendering"] = False

    def print_camera():
        cam = figure.scene.camera
        az, el, _dist, _fp = mlab.view()
        roll = mlab.roll()
        print("\n" + "=" * 66)
        print("# Current camera — paste MANUAL_CAM into tools/visual/npy_to_world_video.py")
        print("# (A) Orientation-only (robust; distance/focal auto-fit per scene):")
        print("MANUAL_CAM = {")
        print(f'    "azimuth": {az:.3f}, "elevation": {el:.3f}, "roll": {roll:.3f},')
        print('    "distance": "auto", "focalpoint": "auto",')
        print("}")
        print("# (B) Absolute pose — EXACT, valid because this viewer renders the SAME")
        print("#     world-frame .npy data as the video:")
        print("# MANUAL_CAM = {")
        print(f'#     "position": {list(np.round(cam.position, 4))},')
        print(f'#     "focal_point": {list(np.round(cam.focal_point, 4))},')
        print(f'#     "view_up": {list(np.round(cam.view_up, 4))},')
        print(f'#     "view_angle": {round(float(cam.view_angle), 4)},')
        print(f'#     "clipping_range": {list(np.round(cam.clipping_range, 4))},')
        print("# }")
        print("=" * 66 + "\n")

    def on_key_press(obj, _evt):
        if state["rendering"]:          # drop key-repeat events during render
            return
        key = obj.GetKeySym()
        if key in ("m", "M"):           # m -> print current camera params
            print_camera()
            return
        f = state["frame"]
        if key in ("n", "N"):
            f = min(f + 1, total_frames - 1)
        elif key in ("b", "B"):
            f = max(f - 1, 0)
        elif key == "Home":
            f = 0
        elif key == "End":
            f = total_frames - 1
        else:
            return  # let VTK handle everything else (arrows = rotate view)
        if f != state["frame"]:
            state["frame"] = f
            state["rendering"] = True
            render_frame(f)

    figure.scene.interactor.add_observer("KeyPressEvent", on_key_press)

    render_frame(state["frame"])
    print("Window opened. Adjust the view, then press M to print MANUAL_CAM.")
    print("Navigate frames: N next  B prev  Home first  End last.")
    mlab.show()
