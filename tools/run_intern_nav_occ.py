import faulthandler
import traceback
import os

import numpy as np
import argparse

# Set environment variables to limit thread usage for numerical libraries
# This is often necessary to prevent CPU oversubscription in multi-process environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


from L3ROcc.dataset.intern_nav_adapter import InternNavSequenceLoader

# Import custom modules after setting up the path
from L3ROcc.generater.intern_vln_env import InternNavDataGenerator


def run_dataset_pipeline(args):
    """
    Main pipeline function to load trajectory data and generate OCC (Occupancy) data.
    """
    # ================= 1. Configuration Parameters =================
    # Project root directory (assumed to be the parent of the current script)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Root directory for the dataset
    dataset_root = args.dataset_root

    # Root directory for saving results
    output_root = args.output_root

    # Directory containing model checkpoints
    model_dir = os.path.join(project_root, "ckpt")

    # Path to the configuration file
    config_path = os.path.join(project_root, "L3ROcc", "configs", "config.yaml")

    # ================= 2. Initialization =================

    print(f"Initializing data loader, scanning path: {dataset_root} ...")
    loader = InternNavSequenceLoader(dataset_root)
    print(f"Scan complete. Found {len(loader)} trajectories.")

    print("Initializing OCC Generator...")
    # Initialize the generator. Note: save_dir is a temporary root here;
    # it will be updated for each specific trajectory in the loop.
    generator = InternNavDataGenerator(
        config_path=config_path, save_dir=output_root, model_dir=model_dir
    )

    # ================= 3. Start Processing Loop =================
    for i in range(len(loader)):
        try:
            # A. Retrieve information from the loader
            video_path, cam_intrinsics = loader.get_trajectory_info(i)

            if video_path is None:
                print(f"Skipping trajectory {i}: Video file not found.")
                continue

            # The DataGenerator requires 'input_path' to point directly to the video file
            # e.g., .../observation.video.trajectory/0.mp4
            input_path_for_gen = video_path

            # B. Construct the specific output path for this trajectory
            # Logic: output_root / group_name / scene_id / trajectory_id
            # We infer the directory structure from 'video_path'
            # Example video_path: .../traj_data/3dfront/scene_abc/traj_1/videos/...
            path_parts = video_path.split(os.sep)

            try:
                # Attempt to extract group, scene, and traj ID based on the 'traj_data' anchor
                start_idx = path_parts.index("traj_data") + 1
                # Extract parts like: 3dfront_d435i/00154.../trajectory_1
                relative_path = os.path.join(*path_parts[start_idx : start_idx + 3])
            except ValueError:
                # Fallback: If 'traj_data' is not in the path, use a simple index-based naming convention
                relative_path = f"trajectory_{i:06d}"

            # Combine output root with the inferred relative path
            current_save_dir = os.path.join(output_root, relative_path)

            # Create the output directory if it does not exist
            if not os.path.exists(current_save_dir):
                os.makedirs(current_save_dir)

            print(f"\n[{i+1}/{len(loader)}] Processing: {relative_path}")
            print(f"   Input: {input_path_for_gen}")
            print(f"   Output: {current_save_dir}")

            # C. Inject parameters into the Generator
            # 1. Override save_path (ensure results are saved to the specific sub-folder)
            generator.save_path = current_save_dir

            # 2. Inject real camera intrinsics (if available)
            if cam_intrinsics is not None:
                generator.camera_intric = cam_intrinsics.astype(np.float32)
            else:
                print("No intrinsics found in Parquet; using default values.")

            # 3. Clear history buffer (prevent state leakage from the previous trajectory)
            if hasattr(generator, "occ_history_buffer"):
                generator.occ_history_buffer.clear()

            # D. Run the core pipeline
            # 'pcd_save=True' enables the saving logic
            generator.run_pipeline(input_path_for_gen, pcd_save=True)

            print("Processing successful!")

        except Exception as e:
            print(f"Processing failed: {e}")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    # Enable fault handler to dump stack trace on segfaults
    parser = argparse.ArgumentParser(
        description="Run InternNav OCC Pipeline for video occupancy generation"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data/examples/small_vln_n1/traj_data/",
        help="Directory to load dataset",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/examples/small_vln_n1/traj_data/",
        help="Directory to save outputs",
    )

    args = parser.parse_args()

    faulthandler.enable()

    run_dataset_pipeline(args)
