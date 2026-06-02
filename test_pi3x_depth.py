import torch
import numpy as np
import os
from L3ROcc.generater.normal_data_vln_env import SimpleVideoDataGenerator
from L3ROcc.utils import load_images_as_tensor
from third_party.pi3.pi3.utils.geometry import depth_edge
import json

project_root = r"f:/CH_Robot/3_project/code/L3ROcc"
config_path = os.path.join(project_root, "L3ROcc", "configs", "config.yaml")
save_dir = r"G:\test_out"
model_dir = os.path.join(project_root, "ckpt")

gen = SimpleVideoDataGenerator(config_path, save_dir, model_dir, use_multimodal=True)
print("Model loaded.")

v_path = r"G:\vln_real_data\lerobot_data\20260601\rosbag_20260529_155555\videos\chunk-000\observation.images.RGB\episode_000.mp4"
d_path = r"G:\vln_real_data\lerobot_data\20260601\rosbag_20260529_155555\videos\chunk-000\observation.images.depth\episode_000.mkv"
intr_path = r"G:\vln_real_data\lerobot_data\20260601\rosbag_20260529_155555\meta\info.json"

with open(intr_path, 'r') as f:
    intrinsics_np = np.array(json.load(f)["head_camera_intrinsic"], dtype=np.float32)

print("\n--- WITH DEPTH ---")
imgs, traj_len, conditions = load_images_as_tensor(v_path, interval=gen.interval, condit_depth_path=d_path, intrinsics_np=intrinsics_np, device=gen.device)
imgs = imgs.to(gen.device)[:2]
conditions = {k: v[:, :2] if v is not None else None for k, v in conditions.items()}

with torch.no_grad(), torch.amp.autocast("cuda", dtype=gen.amp_dtype):
    res = gen.model(imgs[None], **conditions)
conf = torch.sigmoid(res["conf"][..., 0])
masks = (conf > 0.1) & (~depth_edge(res["local_points"][..., 2], rtol=0.03))[0]
print("Metric:", res["metric"].item())
print("Conf mean:", conf.mean().item(), "Conf max:", conf.max().item())
print("Total points:", conf.numel(), "Masked points:", masks.sum().item(), f"({masks.sum().item()/conf.numel()*100:.2f}%)")

print("\n--- WITHOUT DEPTH ---")
conditions_no = {k: None for k in conditions}
with torch.no_grad(), torch.amp.autocast("cuda", dtype=gen.amp_dtype):
    res_no = gen.model(imgs[None], **conditions_no)
conf_no = torch.sigmoid(res_no["conf"][..., 0])
masks_no = (conf_no > 0.1) & (~depth_edge(res_no["local_points"][..., 2], rtol=0.03))[0]
print("Metric:", res_no["metric"].item())
print("Conf mean:", conf_no.mean().item(), "Conf max:", conf_no.max().item())
print("Total points:", conf_no.numel(), "Masked points:", masks_no.sum().item(), f"({masks_no.sum().item()/conf_no.numel()*100:.2f}%)")
