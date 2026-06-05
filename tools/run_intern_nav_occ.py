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


def _parse_bool(v):
    """Accept 'true'/'false' (case-insensitive) as argparse bool values."""
    return str(v).strip().lower() == "true"


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

    # Whether to save files
    pcd_save = args.pcd_save

    # Whether to overwrite existing files
    overwrite = args.overwrite

    # Whether to use mesh instead of origin point cloud
    mesh = args.mesh

    use_depth = args.use_depth
    use_multimodal = args.model_type == "pi3x"

    # -------- Consistency check --------
    if use_depth and not use_multimodal:
        print("[Warning] use_depth=True but model_type='pi3'. "
              "Pi3 does NOT support depth conditioning — depth input will be ignored by the model.")
    if not use_depth and use_multimodal:
        print("[Info] model_type='pi3x' but use_depth=False. "
              "Pi3X will run in RGB-only mode (no depth / intrinsic conditions).")

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
        config_path=config_path,
        save_dir=output_root,
        model_dir=model_dir,
        use_multimodal=use_multimodal,
    )

    # ================= 3. Start Processing Loop =================
    for i in range(len(loader)):
        try:
            # A. Retrieve information from the loader
            video_path, depth_path, cam_intrinsics, cam_extrinsics = loader.get_trajectory_info(i)

            if video_path is None:
                print(f"Skipping trajectory {i}: Video file not found.")
                continue

            # The DataGenerator requires 'input_path' to point directly to the video file
            # e.g., .../observation.video.trajectory/0.mp4
            input_path_for_gen = video_path

            condit_depth_path = None
            intrinsics_np = None
            if use_depth:
                if depth_path is not None and os.path.isfile(depth_path):
                    condit_depth_path = depth_path
                else:
                    print(f"[Warning] No depth video found for trajectory {i}. "
                          "Reconstruction will proceed without depth conditioning.")

                if cam_intrinsics is not None and isinstance(cam_intrinsics, np.ndarray) and cam_intrinsics.shape == (3, 3):
                    intrinsics_np = cam_intrinsics.astype(np.float32)
                else:
                    print(f"[Warning] No valid 3x3 intrinsic matrix found for trajectory {i}. "
                          "Reconstruction will proceed without intrinsic conditioning.")

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
            if cam_intrinsics is not None and isinstance(cam_intrinsics, np.ndarray) and cam_intrinsics.shape == (3, 3):
                generator.camera_intric = cam_intrinsics.astype(np.float32)
            else:
                print("No intrinsics found in Parquet/info.json; using default values.")

            # 3. Clear history buffer (prevent state leakage from the previous trajectory)
            if hasattr(generator, "occ_history_buffer"):
                generator.occ_history_buffer.clear()

            # D. Run the core pipeline
            # 'pcd_save=True' enables the saving logic
            generator.run_pipeline(
                input_path_for_gen,
                condit_depth_path=condit_depth_path,
                intrinsics_np=intrinsics_np,
                pcd_save=pcd_save,
                overwrite=overwrite,
                mesh=mesh,
                T_cam2base=cam_extrinsics,
            )

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
        default="./data/traj_data/",
        help="Directory to load dataset",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./data/traj_data/",
        help="Directory to save outputs",
    )

    parser.add_argument(
        "--pcd_save",
        type=_parse_bool,
        default=True,
        metavar="true|false",
        help="Save files",
    )

    parser.add_argument(
        "--overwrite",
        type=_parse_bool,
        default=False,
        metavar="true|false",
        help="Overwrite existing files",
    )

    parser.add_argument(
        "--mesh",
        type=_parse_bool,
        default=False,
        metavar="true|false",
        help="Use mesh instead of origin point cloud",
    )

    parser.add_argument(
        "--use_depth",
        type=_parse_bool,
        default=False,
        metavar="true|false",
        help="Whether to feed depth data into the model. Default: true.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="pi3x",
        choices=["pi3", "pi3x"],
        help="Checkpoint to load: 'pi3x' (multimodal, supports depth) or 'pi3' (RGB-only).",
    )

    args = parser.parse_args()

    faulthandler.enable()

    run_dataset_pipeline(args)
