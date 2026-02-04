import os
import sys

import mayavi.mlab as mlab
import numpy as np
import scipy.sparse as sparse

# ==============================================================================
# Script Description
# ==============================================================================
# This script is designed for visualizing .npz voxel files.
# Requirement: Run locally using Python 3.8 (Mayavi dependency).
# ==============================================================================


def get_points_from_path(file_path, frame_index=0, config=None):
    """
    Universal data loader: Supports Sparse CSR, Optimized Packbits, and Legacy NPY formats.

    Args:
        file_path (str): Path to the .npz or .npy file.
        frame_index (int): The index of the frame to retrieve.
        config (dict): Configuration dictionary.

    Returns:
        np.ndarray: An (N, 3) array of point coordinates, or None if loading fails.
    """
    points = None

    # =========================================================================
    # 1. Attempt to load as Sparse CSR Matrix (Targeting OCC data)
    # =========================================================================
    try:
        sparse_mat = sparse.load_npz(file_path)
        print(f"Format: Sparse CSR Matrix | Shape: {sparse_mat.shape}")

        total_frames, flat_dim = sparse_mat.shape
        if frame_index >= total_frames:
            print(f"Frame {frame_index} out of bounds. Using last frame.")
            frame_index = -1

        # Extract single frame (sparse row)
        frame_row = sparse_mat[frame_index]
        flat_indices = frame_row.indices  # Get flat indices of non-zero elements

        # Infer spatial dimensions (assuming a cubic grid)
        dim_size = int(round(flat_dim ** (1 / 3)))
        if dim_size**3 == flat_dim:
            grid_size = (dim_size, dim_size, dim_size)
        else:
            print(f"Warning: Non-cubic grid. Assuming 400x400x400...")
            grid_size = config["occ_size"]

        print(f"   -> Extracting Frame {frame_index}, Inferred Grid: {grid_size}")
        x, y, z = np.unravel_index(flat_indices, grid_size, order="C")
        points = np.stack([x, y, z], axis=1)
        return points

    except Exception:
        # Not a sparse matrix; proceed to try other formats.
        pass

    # =========================================================================
    # 2. Attempt to load as Numpy .npz (Targeting Mask data)
    # =========================================================================
    try:
        raw_data = np.load(file_path)

        # --- Case A: Optimized Packbits (Streamed Format) ---
        if "mode" in raw_data and str(raw_data["mode"]) == "packed":
            print(f"Format: Optimized Packbits (Streamed)")

            packed_data = raw_data["data"]  # Shape: (N, Packed_Size)
            stored_shape = raw_data["shape"]  # Shape: (H, W, D)

            # Retrieve dimension information
            total_frames = packed_data.shape[0]
            H, W, D = stored_shape
            flat_len = H * W * D

            if frame_index >= total_frames:
                frame_index = -1

            print(
                f"   -> Extracting Frame {frame_index}/{total_frames} (On-the-fly Unpacking)"
            )

            # [Critical Optimization] Extract and unpack only the requested frame
            frame_packed = packed_data[frame_index]

            # Unpack bits
            frame_unpacked = np.unpackbits(frame_packed)

            # Truncate padding bits and reshape
            frame_bool = frame_unpacked[:flat_len].reshape(H, W, D)

            # Extract coordinates where value is > 0
            points = np.argwhere(frame_bool > 0)
            return points

        # --- Case B: Standard Data (Legacy or Uncompressed) ---
        else:
            # Identify data key
            if "data" in raw_data:
                data_source = raw_data["data"]
            elif "arr_0" in raw_data:
                data_source = raw_data["arr_0"]
            else:
                raise ValueError(f"Unknown keys: {list(raw_data.keys())}")

            print(f"Format: Standard Numpy | Shape: {data_source.shape}")

            # Handle 4D Sequences
            if data_source.ndim == 4:
                if frame_index >= data_source.shape[0]:
                    frame_index = -1
                grid_frame = data_source[frame_index]
                points = np.argwhere(grid_frame > 0)
            # Handle 3D Single Frame
            elif data_source.ndim == 3:
                points = np.argwhere(data_source > 0)
            # Handle Point Cloud List (N, 3)
            elif data_source.ndim == 2 and data_source.shape[1] == 3:
                points = data_source

            return points

    except Exception as e:
        print(f"Error loading file: {e}")
        import traceback

        traceback.print_exc()
        return None


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":

    # 1. Path to your .npz or .npy file
    FILE_PATH = "/trajectory_0/videos/chunk-000/observation.occ.mask/mask_sequence.npz"

    # 2. Frame index to visualize (0 for the first frame)
    FRAME_INDEX = 0

    # 3. Background color setting (True for Black, False for White)
    BG_BLACK = True

    # 4. Config
    config_path = "/L3ROcc/configs/config.yaml"
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    # ==========================================
    # Execution Logic
    # ==========================================

    print(f"Loading: {FILE_PATH}")
    points = get_points_from_path(FILE_PATH, FRAME_INDEX, config)

    if points is None or len(points) == 0:
        print("Result is empty (0 voxels). Check frame index or file path.")
        sys.exit()

    print(f"Plotting {len(points)} voxels...")
    print(
        f"   Bounds: X[{points[:,0].min()}:{points[:,0].max()}] "
        f"Y[{points[:,1].min()}:{points[:,1].max()}] "
        f"Z[{points[:,2].min()}:{points[:,2].max()}]"
    )

    # Configure colors
    bg_color = (0, 0, 0) if BG_BLACK else (1, 1, 1)
    fg_color = (1, 1, 1) if BG_BLACK else (0, 0, 0)

    # Initialize Mayavi figure
    mlab.figure(size=(1000, 800), bgcolor=bg_color, fgcolor=fg_color)

    # Automatically switch display mode for performance
    mode = "point" if len(points) > 500000 else "cube"
    scale_factor = 1.0

    # Render the point cloud
    mlab.points3d(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        mode=mode,  # 'cube' or 'point'
        color=(0, 1, 1),  # Cyan color
        scale_factor=scale_factor,
        scale_mode="none",
    )

    # Add axes and text info
    mlab.axes(xlabel="X", ylabel="Y", zlabel="Z", color=fg_color)
    mlab.text(
        0.01,
        0.01,
        f"Frame: {FRAME_INDEX}\nVoxels: {len(points)}\nMode: {mode}",
        width=0.3,
    )

    print("✨ Window Opened. Interact with the scene.")
    mlab.show()
