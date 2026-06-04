import os
import argparse
from L3ROcc.generater.normal_data_vln_env import SimpleVideoDataGenerator
import json
import numpy as np

# Set environment variables to limit thread usage for numerical libraries
# This is often necessary to prevent CPU oversubscription in multi-process environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _parse_bool(v):
    """Accept 'true'/'false' (case-insensitive) as argparse bool values."""
    return str(v).strip().lower() == "true"


def run_normal_data_pipeline(args):
    """
    Main function to execute the data generation pipeline for a single video.
    This sets up the configuration paths and triggers the generator.
    """

    # ================= 1. Configuration Parameters  =================
    # Project root directory (assumed to be the parent of the current script)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Root directory where the processed output will be saved
    save_dir = os.path.join(args.save_dir, args.mode)

    # Directory containing pre-trained model checkpoints (model_dir/pi3 and model_dir/pi3x)
    model_dir = os.path.join(project_root, "ckpt")

    # Path to the configuration file (YAML)
    config_path = os.path.join(project_root, "L3ROcc", "configs", "config.yaml")

    pcd_save       = args.pcd_save
    mesh           = args.mesh
    use_depth      = args.use_depth          # bool
    use_intrinsic  = args.use_intrinsic      # bool, decoupled from use_depth
    use_multimodal = args.model_type == "pi3x"

    # Quick verification: process only the first N frames to validate the whole data path fast.
    max_frames = args.quick_frames if args.quick else None
    if args.quick:
        print(f"[Quick] Fast verification enabled — only the first {args.quick_frames} "
              "trajectory frames will be processed.")

    # -------- Consistency check --------
    if use_depth and not use_multimodal:
        print("[Warning] use_depth=True but model_type='pi3'. "
              "Pi3 does NOT support depth conditioning — depth input will be ignored by the model.")
    if use_intrinsic and not use_multimodal:
        print("[Info] use_intrinsic=True but model_type='pi3'. "
              "Pi3 does NOT support intrinsic conditioning, but the calibrated K will still be "
              "saved to the dataset (overriding the model's back-calculated K).")
    if not use_depth and not use_intrinsic and use_multimodal:
        print("[Info] model_type='pi3x' with no depth/intrinsic. "
              "Pi3X runs in RGB-only mode; saved K will be the model's back-calculated estimate.")

    # ================= 2. Build depth / intrinsic inputs  =================
    # Depth and intrinsic are now decoupled — each is loaded only if its own flag is on.

    condit_depth_path = None
    if use_depth:
        condit_depth_path = args.condit_depth_path
        if not os.path.isfile(condit_depth_path):
            print(f"[Warning] Depth file not found: {condit_depth_path}. "
                  "Reconstruction will proceed without depth conditioning.")
            condit_depth_path = None

    intrinsics_np = None
    if use_intrinsic:
        if os.path.isfile(args.condit_intr_path) and args.condit_intr_path.endswith(".json"):
            with open(args.condit_intr_path, "r", encoding="utf-8") as f_intr:
                condit_intr = json.load(f_intr)
            intrinsics_np = np.array(condit_intr["head_camera_intrinsic"], dtype=np.float32)
        else:
            print(f"[Warning] Intrinsic JSON not found or invalid: {args.condit_intr_path}. "
                  "Reconstruction will proceed without intrinsic conditioning.")

    # ================= 3. Initialization  =================
    print(f"Initializing SimpleVideoDataGenerator  model={args.model_type}  "
          f"use_depth={use_depth}  config={config_path}")
    generator = SimpleVideoDataGenerator(
        config_path, save_dir, model_dir, use_multimodal=use_multimodal
    )

    # ================= 4. Execution  =================

    # [Option 1] visual_pipeline:
    # Generates files required specifically for visualization purposes.
    if args.mode == "visual":
        generator.visual_pipeline(
            args.video_path, condit_depth_path, intrinsics_np,
            pcd_save=pcd_save, max_frames=max_frames
        )

    # [Option 2] run_pipeline:
    # Generates files required for the LeRobot format and standard dataset structure.
    if args.mode == "run":
        generator.run_pipeline(
            args.video_path, condit_depth_path, intrinsics_np,
            pcd_save=pcd_save, mesh=mesh, T_cam2base=None, max_frames=max_frames
        )


if __name__ == "__main__":
    from pathlib import Path

    # 输入文件路径（默认值，用户可通过命令行参数覆盖）
    default_video_path = str(Path(r"G:\vln_real_data\lerobot_data\20260601\rosbag_20260529_155555\videos\chunk-000\observation.images.RGB\episode_001.mp4"))
    default_condit_depth_path = str(Path(r"G:\vln_real_data\lerobot_data\20260601\rosbag_20260529_155555\videos\chunk-000\observation.images.depth\episode_001.mkv"))
    default_condit_intr_path = str(Path(r"G:\vln_real_data\lerobot_data\20260601\rosbag_20260529_155555\meta\info.json"))
    default_save_dir = str(Path(r"G:\vln_real_data\l3rocc_data\20260601\rosbag_20260529_155555"))

    # 其他参数（默认值）
    default_use_depth = True       # 是否使用深度数据作为模型输入（条件），False 则不使用 depth conditioning
    default_use_intrinsic = True   # 是否使用真实标定内参（既作为 Pi3X 条件，也用于覆盖最终保存的 K）
    default_model_type = "pi3x"    # 'pi3x' (multimodal, supports depth) or 'pi3' (RGB-only)
    default_pcd_save = True        # 是否保存结果文件
    default_mesh = True           # 是否使用 Poisson 表面重建网格（否则输出原始点云）

    quick_verification = True     # 是否启用快速验证模式（仅处理前 N 帧以快速验证数据路径）

    parser = argparse.ArgumentParser(
        description="Run normal data pipeline for video occupancy generation."
    )

    # ---------- Batch Processing Inputs ----------
    # 服务器(Linux)上请用 CLI 覆盖为实际路径，例如：
    #   python tools/run_normal_data_occ.py --input_root /data/lerobot/20260601 --output_root /data/l3rocc/20260601
    parser.add_argument(
        "--input_root", type=str, default=str(Path(r"G:\vln_real_data\lerobot_data\20260601")),
        help="包含多个 rosbag_* 的根目录(批量模式)。服务器上用 CLI 覆盖为 Linux 路径。",
    )
    parser.add_argument(
        "--output_root", type=str, default=str(Path(r"G:\vln_collect_data\l3rocc_data\20260601")),
        help="批量模式输出根目录。服务器上用 CLI 覆盖为 Linux 路径。",
    )

    # ---------- Single Process Inputs (Fallback) ----------
    # 默认空 -> 走批量模式(遍历 --input_root)。要处理单个视频时显式传 --video_path。
    # 单文件示例(本机)：{default_video_path}
    parser.add_argument(
        "--video_path", type=str, default="",
        help="单个 RGB 视频路径。留空(默认)则走 --input_root 批量模式。",
    )
    parser.add_argument(
        "--condit_depth_path", type=str, default=default_condit_depth_path,
        help="Path to the depth video file (MKV gray16le or MP4). Used only when --use_depth true.",
    )
    parser.add_argument(
        "--condit_intr_path", type=str, default=default_condit_intr_path,
        help="Path to the camera intrinsic JSON file. Used only when --use_depth true.",
    )
    parser.add_argument(
        "--save_dir", type=str, default=default_save_dir,
        help="Path to the directory where processed outputs will be saved for single video.",
    )

    # ---------- Mode ----------
    parser.add_argument(
        "--mode", type=str, default="run", choices=["run", "visual"],
        help="'run': generate LeRobot dataset; 'visual': generate visualisation files.",
    )

    # ---------- Depth / Intrinsic / model selection ----------
    parser.add_argument(
        "--use_depth", type=_parse_bool, default=default_use_depth,
        metavar="true|false",
        help="Whether to feed depth data into Pi3X as conditioning.",
    )
    parser.add_argument(
        "--use_intrinsic", type=_parse_bool, default=default_use_intrinsic,
        metavar="true|false",
        help="Whether to load the calibrated K from info.json. When true: "
             "(1) K is fed to Pi3X as conditioning (if model_type='pi3x'); "
             "(2) the rescaled K replaces the DLT-estimated K in the saved Parquet. "
             "Decoupled from --use_depth so RGB+intrinsic-only is possible.",
    )
    parser.add_argument(
        "--model_type", type=str, default=default_model_type, choices=["pi3", "pi3x"],
        help="Checkpoint to load: 'pi3x' (multimodal, supports depth) or 'pi3' (RGB-only).",
    )

    # ---------- Output options ----------
    parser.add_argument(
        "--pcd_save", type=_parse_bool, default=default_pcd_save,
        metavar="true|false",
        help="Save result files. Default: true.",
    )
    parser.add_argument(
        "--mesh", type=_parse_bool, default=default_mesh,
        metavar="true|false",
        help="Use Poisson mesh instead of raw point cloud. Default: false.",
    )

    # ---------- Quick verification ----------
    parser.add_argument(
        "--quick", type=_parse_bool, default=quick_verification,
        metavar="true|false",
        help="Fast pipeline verification: only process the first --quick_frames trajectory "
             "frames so the whole data path runs quickly. Default: false.",
    )
    parser.add_argument(
        "--quick_frames", type=int, default=3,
        help="Number of trajectory frames to process when --quick true. Default: 3.",
    )

    args = parser.parse_args()
    print("args: \n", args)

    if args.video_path:
        # User specified a single file manually, just run it
        run_normal_data_pipeline(args)
    else:
        # Batch processing mode
        input_root = Path(args.input_root.strip())
        output_root = Path(args.output_root.strip())
        
        if not input_root.exists():
            print(f"Error: input_root {input_root} does not exist!")
            exit(1)
            
        # Iterate over rosbags
        for rosbag_dir in input_root.iterdir():
            if not rosbag_dir.is_dir() or not rosbag_dir.name.startswith("rosbag_"):
                continue
                
            rgb_dir = rosbag_dir / "videos" / "chunk-000" / "observation.images.RGB"
            depth_dir = rosbag_dir / "videos" / "chunk-000" / "observation.images.depth"
            meta_file = rosbag_dir / "meta" / "info.json"
            
            if not rgb_dir.exists():
                print(f"Skipping {rosbag_dir.name}: missing RGB directory {rgb_dir}")
                continue
                
            # Iterate over all episodes in this rosbag
            for video_file in rgb_dir.glob("*.mp4"):
                episode_name = video_file.stem  # e.g. "episode_000"
                print(f"\n[{rosbag_dir.name}] Outputting {episode_name} ...")
                
                # Determine associated paths
                args.video_path = str(video_file)
                args.condit_intr_path = str(meta_file) if meta_file.exists() else ""
                
                # Try finding depth video (.mkv or .mp4)
                depth_file = depth_dir / f"{episode_name}.mkv"
                if not depth_file.exists():
                    depth_file = depth_dir / f"{episode_name}.mp4"
                args.condit_depth_path = str(depth_file) if depth_file.exists() else ""
                
                # Set specific save directory for this video
                # Optimize save path: output_root / rosbag_name
                # The pipeline will then append `args.mode` 
                # (e.g. 'run') internally.
                args.save_dir = str(output_root / rosbag_dir.name)
                
                # Strip and clean paths before running to avoid Errno 22
                args.video_path = args.video_path.strip()
                args.save_dir = args.save_dir.strip()
                args.condit_intr_path = args.condit_intr_path.strip()
                args.condit_depth_path = args.condit_depth_path.strip()
                
                try:
                    run_normal_data_pipeline(args)
                except Exception as e:
                    print(f"Failed processing {video_file}: {e}")
                    import traceback
                    traceback.print_exc()

