import os

import cv2
import numpy as np
from tqdm import tqdm

# This script combines two videos side-by-side and overlays an image
# in the top-left corner (Picture-in-Picture).


def combine_videos(
    path_left_video,
    path_right_video,
    path_pip_image,
    output_path,
    pip_scale=0.3,  # Scale factor for the PiP image relative to the left video
    margin=20,  # Margin from the top-left corner in pixels
):
    # 1. Verify file existence
    for p in [path_left_video, path_right_video, path_pip_image]:
        if not os.path.exists(p):
            print(f"Error: File not found -> {p}")
            return

    # 2. Initialize video streams
    cap_left = cv2.VideoCapture(path_left_video)
    cap_right = cv2.VideoCapture(path_right_video)

    # Retrieve video properties (Left video serves as the reference)
    fps = cap_left.get(cv2.CAP_PROP_FPS)
    w_left = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_left = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))

    # 3. Load and process the PiP image (Point Cloud Snapshot)
    pip_img = cv2.imread(path_pip_image)
    if pip_img is None:
        print("Error: Failed to read the PiP image.")
        return

    # Calculate target dimensions for PiP (maintaining aspect ratio)
    pip_h_orig, pip_w_orig = pip_img.shape[:2]
    target_pip_w = int(w_left * pip_scale)  # Target width is 30% of the left video
    target_pip_h = int(target_pip_w * (pip_h_orig / pip_w_orig))

    # Resize the PiP image
    pip_resized = cv2.resize(pip_img, (target_pip_w, target_pip_h))

    # Add a white border to the PiP image for better visibility (Optional)
    pip_resized = cv2.copyMakeBorder(
        pip_resized, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    pip_h, pip_w = pip_resized.shape[:2]

    # 4. Prepare output video writer
    # Read one sample frame from the right video to determine the final canvas width
    ret, frame_right_sample = cap_right.read()
    if not ret:
        return

    # Resize the height of the right video to match the left video for alignment
    h_right_orig, w_right_orig = frame_right_sample.shape[:2]
    scale_factor = h_left / h_right_orig

    # Note: Specific scaling factor (1.78) applied here. Verify if this hardcoding is necessary.
    w_right_new = int(w_right_orig * scale_factor * 1.78)

    # Reset the right video stream to the beginning
    cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Define final canvas dimensions
    canvas_w = w_left + w_right_new
    canvas_h = h_left

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (canvas_w, canvas_h))

    print(f"Processing started...")
    print(f"Left video dimensions: {w_left}x{h_left}")
    print(f"Right video resized to: {w_right_new}x{h_left}")
    print(f"Total canvas dimensions: {canvas_w}x{canvas_h}")

    # 5. Process frames loop
    # Use tqdm to display a progress bar
    pbar = tqdm(total=total_frames, unit="frame")

    while True:
        ret1, frame_left = cap_left.read()
        ret2, frame_right = cap_right.read()

        # Terminate if either video stream ends
        if not ret1 or not ret2:
            break

        # A. Process right frame: Resize height to match the left frame
        frame_right = cv2.resize(frame_right, (w_right_new, h_left))

        # B. Concatenate frames (Horizontal stacking)
        # axis=1 implies horizontal, axis=0 implies vertical
        canvas = np.concatenate((frame_left, frame_right), axis=1)

        # C. Overlay PiP image (Top-left corner)
        # Coordinate range: [y_start : y_end, x_start : x_end]
        y1, y2 = margin, margin + pip_h
        x1, x2 = margin, margin + pip_w

        # Ensure coordinates are within bounds
        if y2 < h_left and x2 < w_left:
            # Pixel-level overlay
            canvas[y1:y2, x1:x2] = pip_resized

        # D. Write frame to output
        writer.write(canvas)
        pbar.update(1)

    # 6. Release resources
    cap_left.release()
    cap_right.release()
    writer.release()
    pbar.close()
    print(f"Merging complete! Video saved to: {output_path}")


# ==========================================
# Configuration Paths
# ==========================================
if __name__ == "__main__":
    # Original Input Video (Left side)
    input_video_path = (
        "/Users/huangbinling/Documents/trae_projects/occgen/occgen/inputs/e1.mp4"
    )

    # OCC Generated Video (Right side)
    occ_video_path = "/Users/huangbinling/Documents/trae_projects/occgen/occgen/outputs/e1_02/occ_video_e04.mp4"

    # Initial Point Cloud Snapshot (PiP Image)
    pcd_image_path = (
        "/Users/huangbinling/Documents/trae_projects/occgen/occgen/snapshot01.png"
    )

    # Output Path
    output_video_path = "/Users/huangbinling/Documents/trae_projects/occgen/occgen/outputs/e1_02/final_demo.mp4"

    combine_videos(
        input_video_path,
        occ_video_path,
        pcd_image_path,
        output_video_path,
        pip_scale=0.3,  # PiP occupies 30% of the left video width
        margin=0,  # Margin in pixels
    )
