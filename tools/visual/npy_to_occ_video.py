import glob
import os
import re
import sys
from argparse import ArgumentParser

import cv2
import mayavi.mlab as mlab
import numpy as np
import scipy.sparse as sparse
import yaml

# This script converts occ sequence data (npy dir OR npz file) into a video.


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


# =============================================================================
# NPZ sequence loaders — returns a list of (N,3) point arrays, one per frame
# =============================================================================

def _grid_size_from_config(config):
    gs = config.get("occ_size", [200, 200, 16])
    return tuple(gs)


def _voxel_to_metric(indices, config):
    """Convert voxel grid indices (N,3) to metric coordinates using pc_range and voxel_size."""
    pc_range = np.array(config.get("pc_range", [-2, -5.6, -0.6, 2, 0, 1.8]), dtype=np.float32)
    voxel_size = float(config.get("voxel_size", 0.04))
    pc_min = pc_range[:3]
    return indices * voxel_size + pc_min + voxel_size * 0.5


def load_frames_sparse_csr(file_path, config):
    """Sparse CSR npz (occ_sequence.npz): shape (N_frames, H*W*D)."""
    mat = sparse.load_npz(file_path)
    total_frames, flat_dim = mat.shape
    grid_size = _grid_size_from_config(config)
    if np.prod(grid_size) != flat_dim:
        dim = int(round(flat_dim ** (1 / 3)))
        grid_size = (dim, dim, dim)
    print(f"   Grid: {grid_size}  flat_dim: {flat_dim}")
    frames = []
    for i in range(total_frames):
        row = mat[i]
        idx = row.indices
        if len(idx) == 0:
            frames.append(np.zeros((0, 3), dtype=np.float32))
            continue
        x, y, z = np.unravel_index(idx, grid_size, order="C")
        pts = np.stack([x, y, z], axis=1).astype(np.float32)
        frames.append(_voxel_to_metric(pts, config))
    return frames


def load_frames_packed(file_path, config):
    """Packed-bits npz (mask_sequence.npz): keys data / shape / mode."""
    raw = np.load(file_path)
    packed = raw["data"]          # (N_frames, packed_size)
    stored_shape = tuple(raw["shape"].tolist())
    H, W, D = stored_shape
    flat_len = H * W * D
    total_frames = packed.shape[0]
    frames = []
    for i in range(total_frames):
        unpacked = np.unpackbits(packed[i])[:flat_len].reshape(H, W, D)
        pts = np.argwhere(unpacked > 0).astype(np.float32)
        frames.append(_voxel_to_metric(pts, config))
    return frames


def load_frames_from_npz(file_path, config):
    """
    Auto-detect npz format and return list of (N,3) point arrays.
    Supports: sparse CSR, packed-bits, standard 4D/3D numpy arrays.
    """
    # 1. Try sparse CSR
    try:
        mat = sparse.load_npz(file_path)
        print(f"NPZ format: Sparse CSR  shape={mat.shape}")
        return load_frames_sparse_csr(file_path, config)
    except Exception:
        pass

    # 2. Try numpy npz
    raw = np.load(file_path)

    # 2a. Packed-bits
    if "mode" in raw and str(raw["mode"]) == "packed":
        print(f"NPZ format: Packed-bits")
        return load_frames_packed(file_path, config)

    # 2b. Standard array
    key = "data" if "data" in raw else ("arr_0" if "arr_0" in raw else None)
    if key is None:
        raise ValueError(f"Unknown npz keys: {list(raw.keys())}")
    arr = raw[key]
    print(f"NPZ format: Standard numpy  shape={arr.shape}")
    if arr.ndim == 4:                          # (N, H, W, D)
        return [np.argwhere(arr[i] > 0).astype(np.float32) for i in range(arr.shape[0])]
    if arr.ndim == 3:                          # single frame (H, W, D)
        return [np.argwhere(arr > 0).astype(np.float32)]
    raise ValueError(f"Unsupported array shape: {arr.shape}")


# =============================================================================
# Resolve input: directory of npy files  OR  single npz sequence file
# =============================================================================

def resolve_input(input_path, config):
    """
    Returns (frames, mode) where:
      - frames: list of np.ndarray (N,3) point arrays, one per frame
      - mode:   'npz' | 'npy'
    """
    if os.path.isfile(input_path) and input_path.lower().endswith(".npz"):
        print(f"Input: single npz sequence file → {input_path}")
        frames = load_frames_from_npz(input_path, config)
        return frames, "npz"

    # Directory: look for a sequence npz first, then fall back to npy files
    if os.path.isdir(input_path):
        for candidate in ("occ_sequence.npz", "mask_sequence.npz"):
            npz_path = os.path.join(input_path, candidate)
            if os.path.exists(npz_path):
                print(f"Input: npz sequence found in dir → {npz_path}")
                frames = load_frames_from_npz(npz_path, config)
                return frames, "npz"

        # Fall back to per-frame npy files
        npy_files = sorted(glob.glob(os.path.join(input_path, "*.npy")), key=numerical_sort)
        if not npy_files:
            print(f"Error: no .npz sequence or .npy files found in {input_path}")
            sys.exit(1)
        print(f"Input: {len(npy_files)} npy files in dir → {input_path}")
        return npy_files, "npy"

    print(f"Error: input path does not exist or is not a file/directory: {input_path}")
    sys.exit(1)


# =============================================================================
# Load one frame from npy (original logic, preserved)
# =============================================================================

def load_npy_frame(file_path):
    fov_voxels = np.load(file_path).astype(np.float32)
    if fov_voxels.ndim == 3:
        fov_voxels = voxel2points(fov_voxels)
    if fov_voxels.ndim == 2 and fov_voxels.shape[1] == 4:
        fov_voxels = fov_voxels[fov_voxels[:, 3] >= 0, :3]
    return fov_voxels


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path

    default_input = str(Path(r'G:\vln_collect_data\l3rocc_data\run\custom_videos\episode_000\trajectory_0\videos\chunk-000\observation.occ.view'))
    default_output = str(Path(r'G:\vln_collect_data\l3rocc_data\run\custom_videos\episode_000\trajectory_0\outputs\occ_sequence.avi'))

    parse = ArgumentParser()
    parse.add_argument(
        "--input",
        type=str,
        default=default_input,
        help="Directory of .npy files OR path to a single .npz sequence file.",
    )
    parse.add_argument(
        "--output_video",
        type=str,
        default=default_output,
    )
    parse.add_argument(
        "--config",
        type=str,
        default="L3ROcc\configs\config.yaml",
    )

    args = parse.parse_args()
    output_video = args.output_video
    config_path = args.config

    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    voxel_size = config.get("voxel_size", 0.05)

    frames_or_files, mode = resolve_input(args.input, config)
    total = len(frames_or_files)
    print(f"Found {total} frames. Generating video...")

    os.makedirs(os.path.dirname(os.path.abspath(output_video)), exist_ok=True)

    figure = mlab.figure(size=(800, 800), bgcolor=(1, 1, 1))

    locked_cam = None        # camera state locked after first non-empty frame
    writer = None            # initialized lazily on first screenshot (real dimensions)
    writer_size = None       # (w, h) stored at init time — don't rely on writer.get()

    def init_writer(img_bgr):
        """Create VideoWriter using actual screenshot dimensions."""
        h, w = img_bgr.shape[:2]
        out_path = output_video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        wr = cv2.VideoWriter(out_path, fourcc, 30.0, (w, h))
        if not wr.isOpened():
            # mp4v failed — fall back to XVID in an AVI container
            out_path = os.path.splitext(output_video)[0] + ".avi"
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            wr = cv2.VideoWriter(out_path, fourcc, 30.0, (w, h))
        print(f"VideoWriter: {w}x{h}  →  {out_path}")
        return wr, (w, h)

    for i in range(total):
        mlab.clf()

        if mode == "npz":
            fov_voxels = frames_or_files[i]
        else:
            fov_voxels = load_npy_frame(frames_or_files[i])

        if len(fov_voxels) == 0:
            print(f"Frame {i}: empty.")
            img_bgr = np.full((800, 800, 3), 255, dtype=np.uint8)  # white placeholder
        else:
            # 1. Compute distance from scene centroid for coloring
            centroid = fov_voxels.mean(axis=0)
            dist_values = np.linalg.norm(fov_voxels - centroid, axis=1)

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
            plt_plot_fov.glyph.scale_mode = "data_scaling_off"
            plt_plot_fov.module_manager.scalar_lut_manager.lut.table = LOW_SAT_LUT

            # 3. Camera: auto-fit on first non-empty frame, then lock viewpoint
            if locked_cam is None:
                mlab.view()                      # auto-fit to scene bounds
                figure.scene.render()
                cam = figure.scene.camera
                locked_cam = {
                    "position":       list(cam.position),
                    "focal_point":    list(cam.focal_point),
                    "view_up":        list(cam.view_up),
                    "view_angle":     cam.view_angle,
                    "clipping_range": list(cam.clipping_range),
                }
                print(f"Camera locked: pos={locked_cam['position']}")
            else:
                cam = figure.scene.camera
                cam.position       = locked_cam["position"]
                cam.focal_point    = locked_cam["focal_point"]
                cam.view_up        = locked_cam["view_up"]
                cam.view_angle     = locked_cam["view_angle"]
                cam.clipping_range = locked_cam["clipping_range"]
                cam.compute_view_plane_normal()
                figure.scene.render()

            img_rgb = mlab.screenshot(figure=figure, mode="rgb", antialiased=True)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Lazy-init writer with real frame dimensions
        if writer is None:
            writer, writer_size = init_writer(img_bgr)

        # Resize to writer dimensions if DPI scaling caused a mismatch
        wr_w, wr_h = writer_size
        if img_bgr.shape[1] != wr_w or img_bgr.shape[0] != wr_h:
            img_bgr = cv2.resize(img_bgr, (wr_w, wr_h))

        writer.write(img_bgr)

        if i % 10 == 0:
            print(f"Processed {i}/{total}")

    if writer is not None:
        writer.release()
    mlab.close(all=True)
    print(f"Video saved to: {output_video}")
