import os

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
# Helper: Detect total frames without loading all frame data
# =============================================================================
def get_total_frames(file_path):
    try:
        mat = sparse.load_npz(file_path)
        return mat.shape[0]
    except Exception:
        pass
    try:
        raw = np.load(file_path)
        if "mode" in raw and str(raw["mode"]) == "packed":
            return int(raw["data"].shape[0])
        key = "data" if "data" in raw else ("arr_0" if "arr_0" in raw else None)
        if key is not None:
            d = raw[key]
            return int(d.shape[0]) if d.ndim == 4 else 1
    except Exception:
        pass
    return 1


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    import yaml
    from pathlib import Path

    # 1. Path to your .npz or .npy file
    FILE_PATH = str(Path(r"G:\vln_collect_data\l3rocc_data\run\custom_videos\episode_000\trajectory_0\videos\chunk-000\observation.occ.mask\mask_sequence.npz"))

    # 2. Frame index to visualize (0 for the first frame)
    FRAME_INDEX = 0

    # 3. Background color setting (True for Black, False for White)
    BG_BLACK = True

    # 4. Config
    config_path = "L3ROcc\configs\config.yaml"
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    # ==========================================
    # Setup
    # ==========================================
    bg_color = (0, 0, 0) if BG_BLACK else (1, 1, 1)
    fg_color = (1, 1, 1) if BG_BLACK else (0, 0, 0)

    total_frames = get_total_frames(FILE_PATH)
    print(f"Loading: {FILE_PATH}  |  Total frames: {total_frames}")

    fig = mlab.figure(size=(1000, 800), bgcolor=bg_color, fgcolor=fg_color)
    state = {
        "frame": max(0, min(FRAME_INDEX, total_frames - 1)),
        "initialized": False,
        "rendering": False,   # guard against re-entrant calls from key-repeat
    }

    # ==========================================
    # Render a single frame (clear + redraw, keep camera)
    # ==========================================
    def render_frame(frame_idx):
        # Save camera only after the first render; on first call the camera
        # hasn't been fitted yet, so restoring it would produce a black screen.
        if state["initialized"]:
            cam = fig.scene.camera
            cam_pos, cam_focal, cam_up = cam.position, cam.focal_point, cam.view_up

        mlab.clf()

        points = get_points_from_path(FILE_PATH, frame_idx, config)
        if points is None or len(points) == 0:
            print(f"Frame {frame_idx}: empty (0 voxels).")
            mlab.text(0.3, 0.5, f"Frame {frame_idx}: no data", width=0.4)
        else:
            mode = "point" if len(points) > 500000 else "cube"
            mlab.points3d(
                points[:, 0], points[:, 1], points[:, 2],
                mode=mode,
                color=(0, 1, 1),  # Cyan
                scale_factor=1.0,
                scale_mode="none",
            )
            mlab.axes(xlabel="X", ylabel="Y", zlabel="Z", color=fg_color)
            print(
                f"Frame {frame_idx}/{total_frames - 1} | "
                f"{len(points)} voxels | mode={mode}"
            )
            mlab.text(
                0.01, 0.01,
                f"Frame: {frame_idx} / {total_frames - 1}\n"
                f"Voxels: {len(points)}  Mode: {mode}",
                width=0.3,
            )

        # On-screen navigation hint (top-right)
        mlab.text(
            0.62, 0.90,
            "N  Next frame     B  Prev frame\n"
            "Home First        End  Last\n"
            "Arrows / Mouse: rotate view",
            width=0.36,
        )

        if state["initialized"]:
            # Restore the user's viewing angle across frame changes
            cam = fig.scene.camera
            cam.position, cam.focal_point, cam.view_up = cam_pos, cam_focal, cam_up

        fig.scene.render()
        state["initialized"] = True
        state["rendering"] = False

    # ==========================================
    # Keyboard handler (frame navigation only; arrows left for view rotation)
    # ==========================================
    def on_key_press(obj, _evt):
        if state["rendering"]:          # drop key-repeat events during render
            return
        key = obj.GetKeySym()
        f = state["frame"]
        if key in ("n", "N"):           # n -> next frame
            f = min(f + 1, total_frames - 1)
        elif key in ("b", "B"):         # b -> previous frame
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

    fig.scene.interactor.add_observer("KeyPressEvent", on_key_press)

    # Initial render
    render_frame(state["frame"])
    print(
        "Window opened. Navigate frames:  N next  B prev  "
        "Home first  End last.  Arrow keys still rotate the view."
    )
    mlab.show()
