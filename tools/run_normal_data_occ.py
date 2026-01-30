import os
import sys
import traceback

import numpy as np



from L3ROcc.base import DataGenerator
from L3ROcc.generater.normal_data_vln_env import SimpleVideoDataGenerator



def run_normal_data_pipeline():
    """
    Main function to execute the data generation pipeline for a single video.
    This sets up the configuration paths and triggers the generator.
    """

    # ================= 1. Configuration Parameters  =================
    # Project root directory (assumed to be the parent of the current script)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Root directory for input data
    input_path = "data/examples/"

    # Target video filename to process
    video_name = "office.mp4"

    # Root directory where the processed output will be saved
    save_dir = "data/examples/outputs/"

    # Directory containing pre-trained model checkpoints
    model_dir = os.path.join(project_root, "ckpt")

    # Path to the configuration file (YAML)
    config_path = os.path.join(project_root, "L3ROcc", "configs", "config.yaml")

    # ================= 2. Initialization  =================
    print(f"Initializing SimpleVideoDataGenerator with config: {config_path}")
    generator = SimpleVideoDataGenerator(config_path, save_dir, model_dir)

    # ================= 3. Execution  =================
    full_video_path = os.path.join(input_path, video_name)

    # [Option 1] visual_pipeline:
    # Generates files required specifically for visualization purposes.
    generator.visual_pipeline(full_video_path, pcd_save=True)

    # [Option 2] run_pipeline:
    # Generates files required for the LeRobot format and standard dataset structure.
    #generator.run_pipeline(full_video_path, pcd_save=True)


if __name__ == "__main__":
    # Set environment variables to limit thread usage for numerical libraries.
    # This prevents CPU oversubscription when running multiple processes or heavy computations.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    run_normal_data_pipeline()
