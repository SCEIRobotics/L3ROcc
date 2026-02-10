import os
import argparse
from L3ROcc.generater.normal_data_vln_env import SimpleVideoDataGenerator

# Set environment variables to limit thread usage for numerical libraries
# This is often necessary to prevent CPU oversubscription in multi-process environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def run_normal_data_pipeline(args):
    """
    Main function to execute the data generation pipeline for a single video.
    This sets up the configuration paths and triggers the generator.
    """

    # ================= 1. Configuration Parameters  =================
    # Project root directory (assumed to be the parent of the current script)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Root directory where the processed output will be saved
    save_dir = os.path.join(args.save_dir, args.mode)  # "data/examples/outputs/"

    # Directory containing pre-trained model checkpoints
    model_dir = os.path.join(project_root, "ckpt")

    # Path to the configuration file (YAML)
    config_path = os.path.join(project_root, "L3ROcc", "configs", "config.yaml")

    # ================= 2. Initialization  =================
    print(f"Initializing SimpleVideoDataGenerator with config: {config_path}")
    generator = SimpleVideoDataGenerator(config_path, save_dir, model_dir)

    # ================= 3. Execution  =================

    # [Option 1] visual_pipeline:
    # Generates files required specifically for visualization purposes.
    if args.mode == "visual":
        generator.visual_pipeline(args.video_path, pcd_save=True)

    # [Option 2] run_pipeline:
    # Generates files required for the LeRobot format and standard dataset structure.
    if args.mode == "run":
        generator.run_pipeline(args.video_path, pcd_save=True)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run normal data pipeline for video occupancy generation."
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="data/examples/office.mp4",
        help="Path to the video file to process.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data/examples/outputs/",
        help="Path to the directory where processed outputs will be saved.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="run",
        help="Mode to run the pipeline in. Options: 'visual' or 'run'.",
    )

    args = parser.parse_args()
    print("args: \n", args)
    run_normal_data_pipeline(args)
