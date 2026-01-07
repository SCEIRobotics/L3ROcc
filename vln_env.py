import torch
import os
import random
import yaml
import time
import glob
import numpy as np
import open3d as o3d
 
# from pi3.utils.geometry import homogenize_points
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3
from pi3.utils.basic import (
    load_images_as_tensor,
    load_depths_as_tensor,
    write_ply,
)  # Assuming you have a helper function
from occ.utils import create_mesh_from_map, preprocess, convert_pointcloud_world_to_camera, interpolate_extrinsics, plot_camera_poses, homogenize_points, pcd_to_voxels, voxels_to_pcd, voxel2points
 
def compute_rigid_transform(source_points, target_points):
    """
    计算源点集到目标点集的刚体变换（旋转矩阵R和平移向量t）
    
    参数:
        source_points: 源点集，形状为(n, 3)的numpy数组，n为点的数量
        target_points: 目标点集，形状为(n, 3)的numpy数组，与源点集一一对应
        
    返回:
        R: 3x3旋转矩阵
        t: 3x1平移向量
        rmse: 均方根误差，评估变换精度
    """
    # 检查输入点集是否有效
    if source_points.shape != target_points.shape:
        raise ValueError("源点集和目标点集必须具有相同的形状")
    if len(source_points) < 3:
        raise ValueError("至少需要3个点来计算变换")
    
    # 步骤1: 计算质心
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)
    
    # 步骤2: 中心化点集（移除质心）
    source_centered = source_points - centroid_source
    target_centered = target_points - centroid_target
    
    # 步骤3: 构造协方差矩阵H
    H = np.dot(source_centered.T, target_centered)
    
    # 步骤4: 对H进行奇异值分解(SVD)
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    
    # 步骤5: 计算旋转矩阵并处理镜像情况
    R = np.dot(V, U.T)
    
    # 确保旋转矩阵行列式为1（右手坐标系）
    if np.linalg.det(R) < 0:
        V[:, 2] *= -1  # 修正V矩阵
        R = np.dot(V, U.T)
    
    # 步骤6: 计算平移向量
    t = centroid_target - np.dot(R, centroid_source)
    
    # 计算均方根误差(RMSE)评估精度
    transformed_source = np.dot(source_points, R.T) + t
    errors = np.linalg.norm(transformed_source - target_points, axis=1)
    rmse = np.sqrt(np.mean(errors **2))
    
    raw_errors = np.linalg.norm(source_points - target_points, axis=1)
    raw_rmse = np.sqrt(np.mean(raw_errors **2))

    return R, t, rmse, raw_rmse

def compute_similarity_transform(source_points, target_points, get_rmse=True):
    """
    计算源点集到目标点集的相似变换（尺度s + 旋转R + 平移t）
    
    参数:
        source_points: 源点集，形状为(n, 3)的numpy数组
        target_points: 目标点集，形状为(n, 3)的numpy数组，与源点集一一对应
        
    返回:
        s: 尺度因子
        R: 3x3旋转矩阵
        t: 3x1平移向量
        rmse: 均方根误差
    """
    # 检查输入有效性
    if source_points.shape != target_points.shape:
        raise ValueError("源点集和目标点集必须形状相同")
    if len(source_points) < 3:
        raise ValueError("至少需要3个非共线点")
    
    # 步骤1: 计算质心
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)
    
    # 步骤2: 中心化点集
    source_centered = source_points - centroid_source  # 形状(n, 3)
    target_centered = target_points - centroid_target  # 形状(n, 3)
    
    # 步骤3: 计算尺度因子s
    # 分子：目标点中心化后的模长平方和
    sum_q_sq = np.sum(np.linalg.norm(target_centered, axis=1) **2)
    # 分母：源点中心化后的模长平方和
    sum_p_sq = np.sum(np.linalg.norm(source_centered, axis=1)** 2)
    if sum_p_sq == 0:
        raise ValueError("源点集不能是单点（无法计算尺度）")
    s = np.sqrt(sum_q_sq / sum_p_sq)  # 尺度因子（开平方确保s>0）
    
    # 步骤4: 归一化目标点的尺度（消除尺度影响）
    target_scaled = target_centered / s  # 形状(n, 3)
    
    # 步骤5: 用SVD计算旋转矩阵R（同刚体变换）
    H = np.dot(source_centered.T, target_scaled)  # 协方差矩阵(3,3)
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    R = np.dot(V, U.T)
    
    # 修正镜像情况（确保行列式为1）
    if np.linalg.det(R) < 0:
        V[:, 2] *= -1
        R = np.dot(V, U.T)
    
    # 步骤6: 计算平移向量t（包含尺度）
    t = centroid_target - s * np.dot(R, centroid_source)
    
    # 计算RMSE评估精度
    if get_rmse:
        transformed_source = s * np.dot(source_points, R.T) + t
        errors = np.linalg.norm(transformed_source - target_points, axis=1)
        rmse = np.sqrt(np.mean(errors **2))
        
        raw_errors = np.linalg.norm(source_points - target_points, axis=1)
        raw_rmse = np.sqrt(np.mean(raw_errors **2))
    else:
        rmse = None
        raw_rmse = None
 
    return s, R, t, rmse, raw_rmse

def ransac_pcd_registration(src_pts, dst_pts, threshold=.25, max_iterations=2000, min_inliers=8):
    """
    一个独立的 RANSAC 实现，用于点云配准，寻找最佳的旋转矩阵 R 和平移向量 t, 尺度变换 s。

    参数:
    src_pts (np.ndarray): 源点云的点集，形状为 (N, 3)。
    dst_pts (np.ndarray): 目标点云的点集，形状为 (N, 3)。
    threshold (float): 重投影误差的阈值，用于判断内点。
    max_iterations (int): RANSAC 的最大迭代次数。

    返回:
    best_R (np.ndarray): 找到的最佳旋转矩阵 (3x3)。
    best_t (np.ndarray): 找到的最佳平移向量 (3,)。
    best_s (float): 找到的最佳尺度变换因子。
    best_mask (np.ndarray): 一个布尔数组，标记哪些点是内点。
    """
    def warp_3d(pts, R, t, s):
        """
        对 3D 点云进行变换：R 旋转，t 平移，s 缩放。
        """
        return s * np.dot(pts, R.T) + t
        
    if src_pts.shape != dst_pts.shape or src_pts.shape[0] < min_inliers:
        raise ValueError(f"输入点集必须具有相同的形状, 且至少包含{min_inliers}个点。")
    
    num_points = src_pts.shape[0]
    best_inliers_count = 0
    best_R = None
    best_t = None
    best_s = None
    best_mask = np.zeros(num_points, dtype=bool)
    
    for _ in range(max_iterations):
        # 1. 随机采样：从所有点中选择4个非共线的点
        # 为了简化，我们先随机选4个，如果它们共线则跳过本次迭代
        sample_indices = random.sample(range(num_points), min_inliers)
        src_sample = src_pts[sample_indices]
        dst_sample = dst_pts[sample_indices]
        # 计算 R, t, s
        s, R, t, _, _ = compute_similarity_transform(src_sample, dst_sample, get_rmse=False)
        
        # 2. 对所有点应用变换
        transformed_src = warp_3d(src_pts, R, t, s)
        
        # 3. 计算重投影误差
        errors = np.linalg.norm(transformed_src - dst_pts, axis=1)
        inliers = errors < threshold
        
        # 4. 更新最佳模型
        inliers_count = np.sum(inliers)
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_R = R
            best_t = t
            best_s = s
            best_mask = inliers
            
        # 5. 如果内点数量足够多，提前结束
        if inliers_count >= min_inliers:
            break
    
    # 使用所有内点算一个最终的R, t, s
    if best_inliers_count >= min_inliers:
        s, R, t, rmse, raw_rmse = compute_similarity_transform(src_pts[best_mask], dst_pts[best_mask])
            
    return best_R, best_t, best_s, rmse, raw_rmse

class DataGenerator:
    def __init__(self, config_path='./occ/config.yaml', save_dir="./data_proccessed", model_dir = "./ckpt"):
        self.config_path = config_path
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Pi3.from_pretrained(model_dir).to(self.device).eval()

        self.free_label = 0
        self.pcd = None
        self.camera_intric = None # intrinsic matrix
        self.camera_pose = None # extrinsic matrix
        self.camera_trace = None
        self.norm_cam_ray = None # 归一化的相机射线方向, 默认处于相机坐标系
        
        

        with open(config_path, 'r') as stream:
            self.config = yaml.safe_load(stream)
        
        self.voxel_size = self.config['voxel_size']
        self.pc_range = self.config['pc_range']
        self.occ_size = self.config['occ_size']
        self.ray_cast_step_size = self.config['ray_cast_step_size']
        self.interval = self.config['interval']
        self.voxel_size_scale = self.config['voxel_size_scale']

    def pcd_reconstuction(self, input_path, pcd_save = False):
        imgs, traj_len = load_images_as_tensor(input_path, interval=self.interval)
        imgs = imgs.to(self.device)
        # Infer
        print("Running model inference...")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                res = self.model(imgs[None]) # Add batch dimension

        # process mask 
        masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
        non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
        masks = torch.logical_and(masks, non_edge)[0]
    
        # get points
        pcd = res['points'][0][masks]
        pcd_color = imgs.permute(0, 2, 3, 1)[masks]
        camera_pose = res['camera_poses'][0].cpu().numpy()

        # 相机位姿插值
        # camera_pose_ = camera_pose
        # camera pose interpolation
        camera_pose = interpolate_extrinsics(camera_pose, np.arange(camera_pose.shape[0]) * self.interval, np.arange(traj_len))
        # plot_camera_poses(camera_pose_, camera_pose)
        
        camera_pose_cuda = torch.from_numpy(camera_pose).to(self.device)
        # 尝试找出相机拍摄方向
        ref_cam_index = 0
        tgt_cam_index = 0
        camera_coord = camera_pose_cuda[ref_cam_index * self.interval][:3, 3].reshape(1, 1, 3)
        cam_ray = res['points'][0][ref_cam_index] - camera_coord # 用ref相机下的点减去相机光心，得到相机到该点的射线
        norm_cam_ray = cam_ray / cam_ray.norm(dim=2, keepdim=True) # h, w, 3, 归一化，此时起点未被锚定
        norm_cam_ray_cam_coords = norm_cam_ray + camera_coord.reshape(1, 3) # 起点放回相机光心
        norm_cam_ray_cam_coords = homogenize_points(norm_cam_ray_cam_coords).reshape(-1, 4)
        norm_cam_ray_cam_coords = torch.einsum('ij,bj->bi', camera_pose_cuda[ref_cam_index * self.interval].inverse(), norm_cam_ray_cam_coords)

        # 可视化 cam_ray
        # norm_cam_ray = norm_cam_ray[::20, ::20, :]# 下采样 
        # norm_cam_ray = norm_cam_ray.reshape(-1, 3)
        # ray_num = 25
        # ray_points = torch.ones(norm_cam_ray.shape[0], ray_num, 1).to(self.device)
        # ray_points = ray_points * torch.arange(ray_num, device=self.device).float().reshape(1, ray_num, 1) * 0.02
        # ray_points = ray_points * norm_cam_ray.unsqueeze(1)
        # ray_points = ray_points.reshape(-1, 3)
        # # 尝试将ray_points从ref相机下转到指定相机下
        # ray_points = ray_points + camera_coord.reshape(1, 3) # 起点放回相机光心
        # ray_points = homogenize_points(ray_points)
        # ray_points_cam = torch.einsum('ij,bj->bi', camera_pose_cuda[ref_cam_index * self.interval].inverse(), ray_points)
        # ray_points = torch.einsum('ij,bj->bi', camera_pose_cuda[tgt_cam_index], ray_points_cam)[:, :3]
        # # 造个color
        # ray_points_color = ray_points - ray_points.min()
        # ray_points_color = ray_points_color / ray_points_color.max()
        # 目前可以获得任意帧下相机的FOV锥桶点，下一步要根据该FOV区域滤除Occ，需要定义全局Occ大小和局部Occ大小，然后把全局Occ
        # 根据相机可见区域与相机位姿获得局部Occ，存储方式使用在全局Occ上的Mask还是直接存储局部Occ？



        # downsample by open3d
        pcd = pcd.cpu().numpy()
        pcd_color = pcd_color.cpu().numpy()
        pcd_ocd = o3d.geometry.PointCloud()
        pcd_ocd.points = o3d.utility.Vector3dVector(pcd)
        pcd_ocd.colors = o3d.utility.Vector3dVector(pcd_color)

        loc_range = pcd.max(0) - pcd.min(0) 
        loc_vol = np.prod(loc_range)
        pcd_num = pcd.shape[0]
        frame_num = imgs.shape[0]
        voxel_size = loc_vol / pcd_num * frame_num * self.voxel_size_scale

        pcd_ocd = pcd_ocd.voxel_down_sample(voxel_size=voxel_size) 
        pcd = np.asarray(pcd_ocd.points)
        pcd_color = np.asarray(pcd_ocd.colors)
        # pcd = torch.from_numpy(np.asarray(pcd_ocd.points)).to(imgs.device)
        # pcd_color = torch.from_numpy(np.asarray(pcd_ocd.colors)).to(imgs.device)
        print(f"downsample the pcd from {pcd_num} to {pcd.shape[0]}")
        
        ########################## visual results ########################## 
        # add the camera point to pcd according the camera pose 
        if pcd_save: 
            # ray_points = ray_points.cpu().numpy()
            # ray_points_color = ray_points_color.cpu().numpy()
            # pcd = np.concatenate([pcd, ray_points], axis=0)
            # pcd_color = np.concatenate([pcd_color, ray_points_color], axis=0)
            
            cam_trace = camera_pose[:, :3, 3]
            pcd_with_cam = np.concatenate([pcd, cam_trace], axis=0)
            cam_trace_color = np.array([[1, 0, 0]] * len(cam_trace))
            pcd_color_with_cam = np.concatenate([pcd_color, cam_trace_color], axis=0)
            # transform the pcd to camera coordinate
            pcd_with_cam = self.convert_pointcloud_world_to_camera(pcd_with_cam, camera_pose[0]) 
            save_path = input_path.split("videos/")[0] # input_path like: .../InternData-N1/vln_n1/traj_data/matterport3d_zed/gTV8FGcVJC9/trajectory_47/videos/chunk-000/observation.video.trajectory
            write_ply(pcd_with_cam, pcd_color_with_cam, os.path.join(save_path, 'pcd_with_cam_fov.ply'))

        return pcd, camera_pose, norm_cam_ray_cam_coords

    def pcd_to_occ(self, pcd):
        pcd = pcd.cpu().numpy() if isinstance(pcd, torch.Tensor) else pcd
        point_cloud_original = o3d.geometry.PointCloud()
        with_normal2 = o3d.geometry.PointCloud()
        point_cloud_original.points = o3d.utility.Vector3dVector(pcd)
        with_normal = preprocess(point_cloud_original, self.config, normals=True)
        with_normal2.points = o3d.utility.Vector3dVector(with_normal.points)
        with_normal2.normals = with_normal.normals
        mesh, _ = create_mesh_from_map(None, self.config['depth'],
                                        self.config['n_threads'],
                                        self.config['min_density'],
                                        with_normal2)
        scene_points = np.asarray(mesh.vertices, dtype=float)

        ################## remain points with a spatial range ##############
        mask = (np.abs(scene_points[:, 0]) < self.pc_range[3]) & (np.abs(scene_points[:, 1]) < self.pc_range[4]) \
            & (scene_points[:, 2] > self.pc_range[2]) & (scene_points[:, 2] < self.pc_range[5]) # 对点云区域做限制  
        scene_points = scene_points[mask]  # mesh化之后顶点需要再被过滤一遍

        ################## convert points to voxels coordinates ##############
        # fov_voxels = pcd_to_voxels(scene_points, self.voxel_size, self.pc_range, unique=True) # 这里要单独处理，不能调用转换函数
        pcd_np = scene_points.copy()
        pcd_np[:, 0] = (pcd_np[:, 0] -
                        self.pc_range[0]) / self.voxel_size  # 转到voxel下的索引下标
        pcd_np[:, 1] = (pcd_np[:, 1] - self.pc_range[1]) / self.voxel_size
        pcd_np[:, 2] = (pcd_np[:, 2] - self.pc_range[2]) / self.voxel_size

        pcd_np = np.floor(pcd_np).astype(np.int32) # 原方法 
        fov_voxels = np.unique(pcd_np, axis=0).astype(np.float64)

        ################## convert voxel coordinates to original system  ##############
        occ_pcd = voxels_to_pcd(fov_voxels, self.voxel_size, self.pc_range)
        # fov_voxels[:, :3] = (
        #     fov_voxels[:, :3] + 0.5
        # ) * self.voxel_size  # fov_voxels的xyz现在算是index，0.5是取中心
        # fov_voxels[:, 0] += self.pc_range[0]
        # fov_voxels[:, 1] += self.pc_range[1]
        # fov_voxels[:, 2] += self.pc_range[2]
        # 以上两部主要是为了把点云中点的位置挪到对应occ voxel的中心 
         
        return occ_pcd

    def occ_gen_pipeline(self, input_path, pcd_save = False):
        '''
            该方法可以根据整段episode视频得到估计的occ和相机姿态轨迹
            输入: input_path, 视频路径
            输出: self.occ, 估计的occ地图 (世界坐标系下)
                self.camera_pose, 估计的相机姿态轨迹 (世界坐标系下)
        '''
        self.camera_intric = np.array([[168.0498 ,   0.     , 240.     ],
                                        [  0.     , 192.79999, 135.     ],
                                        [  0.     ,   0.     ,   1.     ]], dtype=np.float32) # 临时的， 实际应该根据input_path读出来
        # get the pcd with camera pose
        pcd, self.camera_pose, self.norm_cam_ray = self.pcd_reconstuction(input_path, pcd_save)
        # generate the occ map
        self.occ_pcd = self.pcd_to_occ(pcd) # 目前在世界坐标系下, 且仍属于点云状态，无label
        occ_voxels_visual_0, visual_mask_0 = self.check_visual_occ(self.occ_pcd, self.camera_pose[0])
        if pcd_save: # save the occ map
            save_path = input_path.split("videos/")[0] # input_path like: .../InternData-N1/vln_n1/traj_data/matterport3d_zed/gTV8FGcVJC9/trajectory_47/videos/chunk-000/observation.video.trajectory
            occ_pcd_cam_0 = self.convert_pointcloud_world_to_camera(self.occ_pcd, self.camera_pose[0])
            write_ply(occ_pcd_cam_0[:, :3], path=os.path.join(save_path, 'occ.ply')) # 最后一维是颜色
            occ_pcd_cam_0 = np.concatenate([occ_pcd_cam_0, np.ones((occ_pcd_cam_0.shape[0], 1))], axis=1) 
            np.save(os.path.join(save_path, 'occ_pcd_cam0.npy'), occ_pcd_cam_0)
            
            occ_pcd_visual_0 = voxels_to_pcd(occ_voxels_visual_0, self.voxel_size, self.pc_range)
            write_ply(occ_pcd_visual_0[:, :3], path=os.path.join(save_path, 'occ_visual.ply')) # 最后一维是颜色
            occ_pcd_visual_0 = np.concatenate([occ_pcd_visual_0, np.ones((occ_pcd_visual_0.shape[0], 1))], axis=1) 
            np.save(os.path.join(save_path, 'occ_pcd_visual_cam0.npy'), occ_pcd_visual_0)

    def check_visual_occ(self, occ_pcd, camera_pose):
        '''
            检查occ在指定相机下是否可见
            输入: occ, 估计的occ map (世界坐标系下)
                camera_pose, 估计的相机姿态轨迹 (世界坐标系下)
            输出:  
        '''
        occ_pcd_cam = self.convert_pointcloud_world_to_camera(occ_pcd, camera_pose) # 把occ转换到相机坐标系下, shape: (-1, 4)
        occ_voxels = pcd_to_voxels(occ_pcd_cam, self.voxel_size, self.pc_range)
        occ_voxels = torch.tensor(occ_voxels, device=self.norm_cam_ray.device)  
        
        max_distance = 200
        ray_cast_step_size = 1.0
        ray_position = torch.zeros(1, 3, device=self.norm_cam_ray.device) # occ_pcd_cam已经在相机坐标系下了, 若在世界坐标系下则是camera_pose[:, :3].reshape(-1, 3)
        ray_direction_norm = self.norm_cam_ray.reshape(-1, 3)
        pc_range_tensor= torch.tensor(self.pc_range[:3], device=self.norm_cam_ray.device)
        ray_position = (ray_position - pc_range_tensor) / self.voxel_size
        # ray_direction_norm = (ray_direction_norm - pc_range_tensor) / self.voxel_size

        not_hit_ray = torch.ones(len(ray_direction_norm), device=ray_direction_norm.device).bool()
        ray_index_all = torch.arange(len(ray_direction_norm), device=ray_direction_norm.device)
        camera_visible_mask_3d = torch.zeros(self.config['occ_size'], device=ray_direction_norm.device)
        occ_voxels_3d = camera_visible_mask_3d.clone()
        occ_voxels_3d[occ_voxels[:, 0].long(), occ_voxels[:, 1].long(), occ_voxels[:, 2].long()] = 1
        occ_voxels_shape = torch.tensor(self.config['occ_size']).cuda()
        zeros_3 = torch.zeros_like(occ_voxels_shape)
        ray_traverse_start = time.time()
        for step in range(int(max_distance / ray_cast_step_size) + 1): # 在voxels坐标系下算
            if not (not_hit_ray.any() and True):
                print(f"all rays hit the occupied voxel in step {step}!")
                break

            ray_position = ray_position + ray_direction_norm * ray_cast_step_size
            voxel_coords = torch.floor(ray_position).int()

            # check if the voxel_coords is in the range
            coord_valid = (voxel_coords >= zeros_3) & (voxel_coords < occ_voxels_shape)
            position_valid = not_hit_ray & coord_valid.all(dim=1) # coord_valid[:, 0] & coord_valid[:, 1] & coord_valid[:, 2]

            # get the voxel index
            voxel_index = voxel_coords[position_valid] # the length of voxel_index is not always [?]
            ray_selected_index = ray_index_all[position_valid]

            # set the visible occ flag (include free voxel)
            voxel_index_visible = voxel_index # torch.unique(voxel_index, dim=0) # torch.unique is time wasting
            camera_visible_mask_3d[voxel_index_visible[:, 0], voxel_index_visible[:, 1], voxel_index_visible[:, 2]] = 1

            # check if the voxel is occupied
            occ_label_selected = occ_voxels_3d[voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]] # 1 dim only
            occ_not_free = occ_label_selected != self.free_label # bool tensor

            # get the ray index that hit the occupied voxel
            ray_selected_index = ray_selected_index[occ_not_free]

            # get the occupied voxel index
            occ_occupied_index = voxel_index[occ_not_free]

            # record the ray that hit the occupied voxel
            not_hit_ray[ray_selected_index] = False
        ray_traverse_end = time.time()
        print(f"ray traverse cost: {ray_traverse_end - ray_traverse_start}s")
        print("valid occ number: ", occ_voxels_3d.sum())
        occ_voxels_3d = occ_voxels_3d * camera_visible_mask_3d
        occ_voxels = voxel2points(occ_voxels_3d.cpu().numpy(), free_label=self.free_label) # occ voxel coords
        camera_visible_mask = voxel2points(camera_visible_mask_3d.cpu().numpy(), free_label=self.free_label)

        return occ_voxels, camera_visible_mask
    
    def convert_pointcloud_world_to_camera(self, points_world, T_cw):
        """
        将世界坐标系下的点云转换到相机坐标系
        :param points_world: 世界坐标系点云，shape=(N, 3)
        :param T_cw: 相机外参矩阵（相机→世界），shape=(4, 4)
        :return: 相机坐标系点云，shape=(N, 3)
        """
        # 步骤1：提取外参的旋转和平移
        R_cw = T_cw[:3, :3]
        t_cw = T_cw[:3, 3]
        
        # 步骤2：计算世界→相机的旋转和平移
        R_wc = R_cw.T  # 旋转矩阵的逆=转置
        t_wc = -R_wc @ t_cw  # 等价于 -np.dot(R_wc, t_cw)
        
        # 步骤3：对每个点进行变换 
        points_camera = (R_wc @ (points_world - t_cw).T).T
    
        return points_camera

if __name__=="__main__":

    config_path='./occ/config.yaml'
    save_dir="./data_proccessed"
    model_dir = "./ckpt"

    input_path = "/mnt/data_ssd/share/data/InternData-N1/vln_n1/traj_data/matterport3d_zed/gTV8FGcVJC9/trajectory_47/videos/chunk-000/observation.video.trajectory"
    video_name = "episode_000000.mp4"
    generator = DataGenerator(config_path, save_dir, model_dir)
    generator.occ_gen_pipeline(os.path.join(input_path, video_name), pcd_save=True)
    