import os

import cv2
import numpy as np
from tqdm import tqdm

# This script stitches the RGB video, World-space video, and OCC video side-by-side.
# This is the final production script for video concatenation.


def combine_three_videos_crop_middle(
    path_input_video,  # Left: Original input video (Reference)
    path_world_video,  # Middle: World coordinate fusion video (To be cropped)
    path_occ_video,  # Right: OCC video
    output_path,
    crop_ratio=0.2,  # Cropping ratio for the middle video
):
    # 1. Verify file existence
    inputs = [path_input_video, path_world_video, path_occ_video]
    names = ["Input Video", "World Video", "OCC Video"]
    for p, n in zip(inputs, names):
        if not os.path.exists(p):
            print(f"Error: {n} does not exist -> {p}")
            return

    # 2. Initialize video capture streams
    cap_1 = cv2.VideoCapture(path_input_video)  # Left
    cap_2 = cv2.VideoCapture(path_world_video)  # Middle
    cap_3 = cv2.VideoCapture(path_occ_video)  # Right

    # Retrieve reference properties (based on the left input video)
    fps = cap_1.get(cv2.CAP_PROP_FPS)
    h1 = int(cap_1.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Reference height
    w1 = int(cap_1.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap_1.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- Process middle video (World) - Requires cropping ---
    ret2, frame2_sample = cap_2.read()
    if not ret2:
        return
    h2_raw, w2_raw = frame2_sample.shape[:2]

    # Calculate cropping margins
    # crop_ratio = 0.2 implies removing 20% from top, bottom, left, and right
    margin_h = int(h2_raw * crop_ratio)
    margin_w = int(w2_raw * crop_ratio)

    # Calculate dimensions after cropping (used for scaling calculation)
    h2_cropped_orig = h2_raw - 2 * margin_h
    w2_cropped_orig = w2_raw - 2 * margin_w

    # Calculate scaling factor (Align cropped height to h1)
    scale2 = h1 / h2_cropped_orig
    w2_new = int(w2_cropped_orig * scale2)  # Final width on the canvas

    cap_2.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset frame pointer

    # --- Process right video (OCC) - No cropping required ---
    ret3, frame3_sample = cap_3.read()
    if not ret3:
        return
    h3_orig, w3_orig = frame3_sample.shape[:2]
    scale3 = h1 / h3_orig
    w3_new = int(w3_orig * scale3)
    cap_3.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset frame pointer

    # Calculate total canvas width
    canvas_w = w1 + w2_new + w3_new
    canvas_h = h1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (canvas_w, canvas_h))

    print(f"Starting 3-screen merge (Middle video crop {crop_ratio*100:.0f}%)...")
    print(f"View 1 (Input) : {w1}x{h1}")
    print(
        f"View 2 (World) : Raw {w2_raw}x{h2_raw} -> Cropped {w2_cropped_orig}x{h2_cropped_orig} -> Scaled {w2_new}x{h1}"
    )
    print(f"View 3 (OCC)   : {w3_new}x{h1}")
    print(f"Final Canvas   : {canvas_w}x{canvas_h}")

    # 5. Processing loop
    pbar = tqdm(total=total_frames, unit="frame")

    while True:
        ret1, frame1 = cap_1.read()
        ret2, frame2 = cap_2.read()
        ret3, frame3 = cap_3.read()

        # Terminate loop if any video stream ends
        if not ret1 or not ret2 or not ret3:
            break

        # A. Process middle video: Crop first, then resize
        # Slicing syntax: image[y_start:y_end, x_start:x_end]
        # Using negative indices: -margin_h is equivalent to h2_raw - margin_h
        frame2_cropped = frame2[margin_h:-margin_h, margin_w:-margin_w]
        frame2 = cv2.resize(frame2_cropped, (w2_new, h1))

        # B. Process right video: Resize directly
        frame3 = cv2.resize(frame3, (w3_new, h1))

        # C. Concatenate three screens [ Left | Middle | Right ]
        canvas = np.concatenate((frame1, frame2, frame3), axis=1)

        writer.write(canvas)
        pbar.update(1)

    cap_1.release()
    cap_2.release()
    cap_3.release()
    writer.release()
    pbar.close()
    print(f"\nThree-screen (cropped) video saved to: {output_path}")


if __name__ == "__main__":
    base_dir = "/Users/huangbinling/Documents/trae_projects/occgen/occgen"

    # 1. Far Left: Original Video
    path_input = os.path.join(base_dir, "inputs/office.mp4")

    # 2. Middle: World Coordinate Fusion Video (To be cropped and magnified)
    # Using the True Color video generated previously
    path_world = os.path.join(base_dir, "outputs/office_1/real_color_world.mp4")

    # 3. Far Right: Pure OCC Video
    path_occ = os.path.join(base_dir, "outputs/office_1/occ_only.mp4")

    # 4. Output Path
    path_output = os.path.join(base_dir, "outputs/office_1/final_3screen_crop_demo.mp4")

    # Execute
    combine_three_videos_crop_middle(
        path_input,
        path_world,
        path_occ,
        path_output,
        crop_ratio=0.15,  # Crop 15% from top, bottom, left, and right
    )
