import argparse
import faulthandler
import json
import os
import traceback

import numpy as np

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


def _load_intrinsics_from_json(json_path):
    """Read a 3x3 ``head_camera_intrinsic`` from an info.json. Returns None on failure."""
    if (
        not json_path
        or not os.path.isfile(json_path)
        or not json_path.endswith(".json")
    ):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f_intr:
            data = json.load(f_intr)
        return np.array(data["head_camera_intrinsic"], dtype=np.float32)
    except Exception as e:
        print(f"[Warning] Failed to read intrinsic JSON {json_path}: {e}")
        return None


def run_dataset_pipeline(args):
    """
    Main pipeline function to load trajectory data and generate OCC (Occupancy) data.
    """
    # ================= 1. Configuration Parameters =================
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dataset_root = args.dataset_root
    output_root = args.output_root

    pcd_save = args.pcd_save
    overwrite = args.overwrite
    mesh = args.mesh

    use_depth = args.use_depth
    use_intrinsic = args.use_intrinsic
    is_pi3 = args.model_type == "pi3"

    model_dir = os.path.join(project_root, "ckpt")
    config_path = os.path.join(project_root, "L3ROcc", "configs", "config.yaml")

    # -------- Consistency hints (mirror run_normal_data_occ.py) --------
    if use_depth and is_pi3:
        print(
            "[Warning] use_depth=True but model_type='pi3'. "
            "Pi3 does NOT support depth conditioning — depth input will be ignored by the model."
        )
    if use_intrinsic and is_pi3:
        print(
            "[Info] use_intrinsic=True but model_type='pi3'. "
            "Pi3 does NOT support intrinsic conditioning, but the calibrated K will still be "
            "saved to the dataset (overriding the model's back-calculated K)."
        )
    if not use_depth and not use_intrinsic and not is_pi3:
        print(
            "[Info] model_type='pi3x' with no depth/intrinsic. "
            "Pi3X runs in RGB-only mode; saved K will be the model's back-calculated estimate."
        )

    # ================= 2. Initialization =================
    print(f"Initializing data loader, scanning path: {dataset_root} ...")
    loader = InternNavSequenceLoader(dataset_root)
    print(f"Scan complete. Found {len(loader)} trajectories.")

    print(
        f"Initializing OCC Generator  model_type={args.model_type}  "
        f"use_depth={use_depth}  use_intrinsic={use_intrinsic}"
    )
    # save_dir is a temporary root here; it gets overridden per-trajectory below.
    generator = InternNavDataGenerator(
        config_path=config_path,
        save_dir=output_root,
        model_dir=model_dir,
        model_type=args.model_type,
    )

    # Optional external intrinsic JSON (overrides the loader-resolved K when present;
    # loader K comes from parquet observation.camera_intrinsic or, as fallback,
    # per-trajectory meta/info.json head_camera_intrinsic — lerobot v2.1 only has the latter)
    cli_intrinsics_np = (
        _load_intrinsics_from_json(args.condit_intr_path) if use_intrinsic else None
    )
    if use_intrinsic and args.condit_intr_path and cli_intrinsics_np is None:
        print(
            f"[Warning] --condit_intr_path provided but not loadable; "
            f"will fall back to the loader-resolved intrinsic "
            f"(parquet observation.camera_intrinsic or meta/info.json head_camera_intrinsic) if available."
        )

    # ================= 3. Start Processing Loop =================
    for i in range(len(loader)):
        try:
            # A. Retrieve info from the loader (returns depth_path between video and intrinsic)
            video_path, depth_path, cam_intrinsics, cam_extrinsics, cam_convention = (
                loader.get_trajectory_info(i)
            )

            if video_path is None:
                print(f"Skipping trajectory {i}: Video file not found.")
                continue

            input_path_for_gen = video_path

            # B. Construct the specific output path for this trajectory.
            # InternData-N1: output_root / <group> / <scene> / <trajectory_*>
            # lerobot rosbag: output_root / <rosbag_*> / <episode_id>
            path_parts = video_path.split(os.sep)
            try:
                start_idx = path_parts.index("traj_data") + 1
                relative_path = os.path.join(*path_parts[start_idx : start_idx + 3])
            except ValueError:
                rosbag_idx = next(
                    (j for j, p in enumerate(path_parts) if p.startswith("rosbag_")),
                    None,
                )
                if rosbag_idx is not None:
                    episode_id = os.path.splitext(path_parts[-1])[0]
                    relative_path = os.path.join(path_parts[rosbag_idx], episode_id)
                else:
                    relative_path = f"trajectory_{i:06d}"

            current_save_dir = os.path.join(output_root, relative_path)
            if not os.path.exists(current_save_dir):
                os.makedirs(current_save_dir)

            print(f"\n[{i + 1}/{len(loader)}] Processing: {relative_path}")
            print(f"   Input:  {input_path_for_gen}")
            print(f"   Output: {current_save_dir}")

            # C. Resolve depth and intrinsic for this trajectory
            condit_depth_path = None
            if use_depth:
                if depth_path and os.path.isfile(depth_path):
                    condit_depth_path = depth_path
                    print(f"   Depth:  {condit_depth_path}")
                else:
                    print(
                        f"   [Warning] use_depth=True but no depth video found "
                        f"under trajectory; proceeding without depth conditioning."
                    )

            # Intrinsic priority: CLI JSON override > loader (parquet observation.camera_intrinsic,
            # else meta/info.json head_camera_intrinsic) > None (DLT fallback)
            intrinsics_np = None
            if use_intrinsic:
                if cli_intrinsics_np is not None:
                    intrinsics_np = cli_intrinsics_np
                elif cam_intrinsics is not None:
                    intrinsics_np = cam_intrinsics.astype(np.float32)
                else:
                    print(
                        "   [Info] use_intrinsic=True but no intrinsic available; "
                        "DLT estimation will be used."
                    )

            # D. Inject per-trajectory state into the generator
            generator.save_path = current_save_dir
            if hasattr(generator, "occ_history_buffer"):
                generator.occ_history_buffer.clear()

            # E. Run the core pipeline with decoupled depth / intrinsic
            generator.run_pipeline(
                input_path_for_gen,
                condit_depth_path=condit_depth_path,
                intrinsics_np=intrinsics_np,
                pcd_save=pcd_save,
                overwrite=overwrite,
                mesh=mesh,
                T_cam2base=cam_extrinsics,
                extrinsic_convention=cam_convention,
            )

            print("Processing successful!")

        except Exception as e:
            print(f"Processing failed: {e}")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run InternNav OCC Pipeline for video occupancy generation"
    )

    # ---------- Dataset paths ----------
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="./data/traj_data/",
        help="Root directory to load InternData-N1 trajectories.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./data/traj_data/",
        help="Root directory to save generated outputs.",
    )

    # ---------- Mode / Model selection ----------
    parser.add_argument(
        "--model_type",
        type=str,
        default="pi3x",
        choices=["pi3", "pi3x"],
        help="Checkpoint to load: 'pi3x' (multimodal, supports depth) or 'pi3' (RGB-only).",
    )
    parser.add_argument(
        "--use_depth",
        type=_parse_bool,
        default=False,
        metavar="true|false",
        help="Feed depth into Pi3X as conditioning. InternData-N1 usually has no depth; "
        "default false. When true, the loader probes each trajectory for a depth video.",
    )
    parser.add_argument(
        "--use_intrinsic",
        type=_parse_bool,
        default=True,
        metavar="true|false",
        help="Use a calibrated K. When true: (1) the loader reads K from parquet's "
        "observation.camera_intrinsic if present, otherwise falls back to the "
        "per-trajectory meta/info.json 'head_camera_intrinsic' "
        "(lerobot v2.1 data does NOT write K to parquet, so only the info.json path applies), "
        "(2) --condit_intr_path (if given) overrides the loaded value for all trajectories, "
        "(3) the rescaled K replaces the DLT-estimated K in the saved output Parquet, "
        "(4) the K is also fed to Pi3X as conditioning when model_type='pi3x'.",
    )
    parser.add_argument(
        "--condit_intr_path",
        type=str,
        default="",
        help="Optional info.json path with 'head_camera_intrinsic'. When set, overrides "
        "the per-trajectory loaded intrinsic (parquet or meta/info.json) for ALL trajectories.",
    )

    # ---------- Output options ----------
    parser.add_argument(
        "--pcd_save",
        type=_parse_bool,
        default=True,
        metavar="true|false",
        help="Save result files. Default: true.",
    )
    parser.add_argument(
        "--overwrite",
        type=_parse_bool,
        default=True,
        metavar="true|false",
        help="Overwrite existing files even when artifacts look complete. Default: false.",
    )
    parser.add_argument(
        "--mesh",
        type=_parse_bool,
        default=False,
        metavar="true|false",
        help="Use Poisson mesh instead of raw point cloud. Default: false.",
    )

    args = parser.parse_args()
    print("args: \n", args)

    faulthandler.enable()

    run_dataset_pipeline(args)
