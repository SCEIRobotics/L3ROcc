import os
import cv2
import pdb
import numpy as np
import pandas as pd
import torch 
import open3d as o3d

def convert_pointcloud_world_to_camera(points_world, T_cw):
    """
    Transforms point cloud from World to Camera frame.
    Supports both Numpy and PyTorch Tensor (GPU).
    """
    # 1. Tensor Mode (GPU Optimized)
    if isinstance(points_world, torch.Tensor):
        if not isinstance(T_cw, torch.Tensor):
            T_cw = torch.tensor(
                T_cw, device=points_world.device, dtype=points_world.dtype
            )

        R_cw = T_cw[:3, :3]
        t_cw = T_cw[:3, 3]

        # Logic: P_cam = (P_world - t_cw) @ R_cw
        # Note: R_wc = R_cw.T. The formula is P_cam = (R_wc @ (P_world - t_cw).T).T
        # Which simplifies to: (P_world - t_cw) @ R_wc.T => (P_world - t_cw) @ R_cw
        points_camera = (points_world - t_cw) @ R_cw
        return points_camera

    # 2. Numpy Mode (Legacy)
    else:
        if len(points_world) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        R_cw = T_cw[:3, :3]
        t_cw = T_cw[:3, 3]
        R_wc = R_cw.T
        points_camera = (R_wc @ (points_world - t_cw).T).T
        return points_camera.astype(np.float32)


if __name__ == '__main__':
    pd_path = "data/examples/3dfront_zed/41102cdc-f833-4edf-8d5b-4dcd24607969/trajectory_59/data/chunk-000/episode_000000.parquet"
    df = pd.read_parquet(pd_path)
    print(df.keys())

    ply_path = "/mnt/data/yenianjin/project/L3ROcc/data/examples/3dfront_zed/41102cdc-f833-4edf-8d5b-4dcd24607969/trajectory_59/data/chunk-000/origin_pcd.ply"
    pcd = o3d.io.read_point_cloud(ply_path)
    o3d.visualization.draw_geometries([pcd])

    camera_extrinsic_occ = np.array([x for x in df["observation.camera_extrinsic_occ"][0]]) 
    pcd_array = np.asarray(pcd.points) 
    pcd_array_cam = convert_pointcloud_world_to_camera(pcd_array, camera_extrinsic_occ)
    pcd.points = o3d.utility.Vector3dVector(pcd_array_cam) 
    o3d.io.write_point_cloud("pcd_cam.ply", pcd)

    pdb.set_trace()
