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
from occ.utils import (
    create_mesh_from_map,
    preprocess,
    convert_pointcloud_world_to_camera,
    interpolate_extrinsics,
    plot_camera_poses,
    homogenize_points,
    pcd_to_voxels,
    voxels_to_pcd,
    voxel2points,
)


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
    rmse = np.sqrt(np.mean(errors**2))

    raw_errors = np.linalg.norm(source_points - target_points, axis=1)
    raw_rmse = np.sqrt(np.mean(raw_errors**2))

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
    sum_q_sq = np.sum(np.linalg.norm(target_centered, axis=1) ** 2)
    # 分母：源点中心化后的模长平方和
    sum_p_sq = np.sum(np.linalg.norm(source_centered, axis=1) ** 2)
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
        rmse = np.sqrt(np.mean(errors**2))

        raw_errors = np.linalg.norm(source_points - target_points, axis=1)
        raw_rmse = np.sqrt(np.mean(raw_errors**2))
    else:
        rmse = None
        raw_rmse = None

    return s, R, t, rmse, raw_rmse


def ransac_pcd_registration(
    src_pts, dst_pts, threshold=0.25, max_iterations=2000, min_inliers=8
):
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
        s, R, t, _, _ = compute_similarity_transform(
            src_sample, dst_sample, get_rmse=False
        )

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
        s, R, t, rmse, raw_rmse = compute_similarity_transform(
            src_pts[best_mask], dst_pts[best_mask]
        )

    return best_R, best_t, best_s, rmse, raw_rmse


class DataGenerator:
    def __init__(
        self,
        config_path="./occ/config.yaml",
        save_dir="./data_proccessed",
        model_dir="./ckpt",
    ):
        self.config_path = config_path
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Pi3.from_pretrained(model_dir).to(self.device).eval()

        self.free_label = 0
        self.pcd = None
        self.camera_intric = None  # intrinsic matrix
        self.camera_pose = None  # extrinsic matrix
        self.camera_trace = None
        self.norm_cam_ray = None  # 归一化的相机射线方向, 默认处于相机坐标系

        with open(config_path, "r") as stream:
            self.config = yaml.safe_load(stream)

        self.voxel_size = self.config["voxel_size"]
        self.pc_range = self.config["pc_range"]
        self.occ_size = self.config["occ_size"]
        self.ray_cast_step_size = self.config["ray_cast_step_size"]
        self.interval = self.config["interval"]
        self.voxel_size_scale = self.config["voxel_size_scale"]

        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(self.project_dir, "outputs")
        self.save_path = os.path.join(self.save_dir, "office_1")

    def pcd_reconstuction(self, input_path, pcd_save=False):
        imgs, traj_len = load_images_as_tensor(
            input_path, interval=self.interval
        )  # 将一段视频中的所有帧按照interval=10帧读取出来，并统一缩放到相同的大小，最后转换成一个深度学习模型可以直接使用的Tensor[N, 3, H, W]，N为视频总帧数traj_len除以interval
        imgs = imgs.to(self.device)  # [N, 3, H, W]

        # Infer 运行模型推理，得到点云以及相机姿态
        print("Running model inference...")
        dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                res = self.model(imgs[None])  # Add batch dimension $[1, N, 3, H, W]$

        # process mask 过滤噪声
        masks = torch.sigmoid(res["conf"][..., 0]) > 0.1  # 只保留模型觉得“确定”的点
        non_edge = ~depth_edge(
            res["local_points"][..., 2], rtol=0.03
        )  # 边缘过滤，去掉物体边缘那些虚无缥缈的拉丝点（深度突变处），让点云边缘更锐利
        masks = torch.logical_and(masks, non_edge)[
            0
        ]  # 只有既可信又不是边缘的点才被保留

        # get points
        pcd = res["points"][0][masks]  # 模型预测出的全局 3D 坐标 [N, H, W, 3]
        pcd_color = imgs.permute(0, 2, 3, 1)[
            masks
        ]  # 对应每个点的颜色  [N,3,H, W]--> [N, H, W, 3]
        camera_pose = res["camera_poses"][0].cpu().numpy()

        # 相机位姿插值
        # camera_pose_ = camera_pose
        # camera pose interpolation
        camera_pose = interpolate_extrinsics(
            camera_pose,
            np.arange(camera_pose.shape[0]) * self.interval,
            np.arange(traj_len),
        )  # 这里得到的camera_pose是插值为traj_len长的
        # plot_camera_poses(camera_pose_, camera_pose)
        camera_pose_cuda = torch.from_numpy(camera_pose).to(self.device)

        # 尝试找出相机拍摄方向
        ref_cam_index = 0  # 这里是由第0帧作为参考，确定相机的拍摄方向，转移到相机坐标系下，就会成为一个固定的发射锥体
        tgt_cam_index = 0
        camera_coord = camera_pose_cuda[ref_cam_index * self.interval][:3, 3].reshape(
            1, 1, 3
        )  # 该帧相机在世界坐标系下的 (x, y, z) 坐标
        cam_ray = (
            res["points"][0][ref_cam_index] - camera_coord
        )  # 用ref相机下的点减去相机光心，得到相机到该点的射线
        norm_cam_ray = cam_ray / cam_ray.norm(
            dim=2, keepdim=True
        )  # h, w, 3, 归一化，此时起点未被锚定
        norm_cam_ray_cam_coords = norm_cam_ray + camera_coord.reshape(
            1, 3
        )  # 起点放回相机光心 --norm_cam_ray_cam_coords 变成了一个具体的坐标点。这个点位于：从相机位置出发，沿着射线方向走 1 米远的地方
        norm_cam_ray_cam_coords = homogenize_points(norm_cam_ray_cam_coords).reshape(
            -1, 4
        )  # 齐次化
        norm_cam_ray_cam_coords = torch.einsum(
            "ij,bj->bi",
            camera_pose_cuda[ref_cam_index * self.interval].inverse(),
            norm_cam_ray_cam_coords,
        )  # 将norm_cam_ray_cam_coords从世界坐标系变换到相机坐标系

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
        pcd = pcd.cpu().numpy()  # 模型预测出的全局 3D 坐标 [N, H, W, 3]
        pcd_color = pcd_color.cpu().numpy()
        pcd_ocd = o3d.geometry.PointCloud()  # 创建一个 Open3D 的点云对象容器
        pcd_ocd.points = o3d.utility.Vector3dVector(pcd)
        pcd_ocd.colors = o3d.utility.Vector3dVector(pcd_color)

        loc_range = pcd.max(0) - pcd.min(
            0
        )  # 找出N帧所有点里，最东边和最西边的距离、最南和最北、最高和最低。算出了这个场景的“长、宽、高”
        loc_vol = np.prod(loc_range)  # 算出了这个场景的总体积
        pcd_num = pcd.shape[0]
        frame_num = imgs.shape[0]
        voxel_size = (
            loc_vol / pcd_num * frame_num * self.voxel_size_scale
        )  # 算出了单帧画面里，平均每个点该有的空间大小 * self.voxel_size_scale是个缩放因子，

        pcd_ocd = pcd_ocd.voxel_down_sample(
            voxel_size=voxel_size
        )  # 执行“降采样”，按照voxel_size的大小，把点云分成多个小的体素，每个体素里的点取一个平均值，作为该体素的代表点。
        pcd = np.asarray(pcd_ocd.points)  # [M, 3]， M是降采样后的点云数量
        pcd_color = np.asarray(pcd_ocd.colors)
        self.pcd_color = pcd_color
        self.pcd = pcd
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
            cam_trace_color = np.array(
                [[1, 0, 0]] * len(cam_trace)
            )  # 给相机轨迹点染上红色
            pcd_color_with_cam = np.concatenate([pcd_color, cam_trace_color], axis=0)
            # transform the pcd to camera coordinate
            pcd_with_cam = self.convert_pointcloud_world_to_camera(
                pcd_with_cam, camera_pose[0]
            )
            # save_path = input_path.split("videos/")[0] # input_path like: .../InternData-N1/vln_n1/traj_data/matterport3d_zed/gTV8FGcVJC9/trajectory_47/videos/chunk-000/observation.video.trajectory
            ply_save_path = os.path.join(
                self.save_path, "pcd_with_cam_fov.ply"
            )  # 保存相机坐标系下点云和相机轨迹的ply文件
            write_ply(pcd_with_cam, pcd_color_with_cam, ply_save_path)

        return pcd, camera_pose, norm_cam_ray_cam_coords[:, :3]

    def pcd_to_occ(self, pcd):
        pcd = pcd.cpu().numpy() if isinstance(pcd, torch.Tensor) else pcd
        point_cloud_original = o3d.geometry.PointCloud()
        with_normal2 = o3d.geometry.PointCloud()
        point_cloud_original.points = o3d.utility.Vector3dVector(pcd)
        with_normal = preprocess(point_cloud_original, self.config, normals=True)
        with_normal2.points = o3d.utility.Vector3dVector(with_normal.points)
        with_normal2.normals = with_normal.normals
        mesh, _ = create_mesh_from_map(
            None,
            self.config["depth"],
            self.config["n_threads"],
            self.config["min_density"],
            with_normal2,
        )
        scene_points = np.asarray(
            mesh.vertices, dtype=float
        )  # 提取均匀覆盖在表面上的顶点，这样的点云会比较密，比较完整地覆盖了场景的表面。

        ################## remain points with a spatial range ##############
        mask = (
            (np.abs(scene_points[:, 0]) < self.pc_range[3])
            & (np.abs(scene_points[:, 1]) < self.pc_range[4])
            & (scene_points[:, 2] > self.pc_range[2])
            & (scene_points[:, 2] < self.pc_range[5])
        )  # 对点云区域做限制   [x_min, y_min, z_min, x_max, y_max, z_max]区域内
        scene_points = scene_points[mask]  # mesh化之后顶点需要再被过滤一遍

        ################## convert points to voxels coordinates ##############
        # fov_voxels = pcd_to_voxels(scene_points, self.voxel_size, self.pc_range, unique=True) # 这里要单独处理，不能调用转换函数
        pcd_np = scene_points.copy()
        pcd_np[:, 0] = (
            pcd_np[:, 0] - self.pc_range[0]
        ) / self.voxel_size  # 转到voxels下的索引下标 原来的点云坐标是在世界坐标系下的，需要先转换到voxels坐标系下 点云的最小边界处就是voxels的原点 然后/ self.voxel_size = 每个点在voxels坐标系下的坐标
        pcd_np[:, 1] = (pcd_np[:, 1] - self.pc_range[1]) / self.voxel_size
        pcd_np[:, 2] = (pcd_np[:, 2] - self.pc_range[2]) / self.voxel_size

        pcd_np = np.floor(pcd_np).astype(
            np.int32
        )  # 原方法  [2.1, 5.8, 1.2] -> [2, 5, 1]，给voxels空间中所有的点分配格子编号
        fov_voxels = np.unique(pcd_np, axis=0).astype(
            np.float64
        )  # 去重，得到所有点云在voxels坐标系下的唯一格子索引，每个索引间隔为self.voxel_size=0.05m
        """"计算后	[2.1, 5.8, 1.2], [2.9, 5.1, 1.1]	散乱的点，带有精确坐标
            floor 后	[2, 5, 1], [2, 5, 1]	确定了它们都在同一个格子里
            unique 后	[2, 5, 1]	合并！这个格子现在被标记为“占据”"""
        ################## convert voxel coordinates to original system  ##############
        occ_pcd = voxels_to_pcd(
            fov_voxels, self.voxel_size, self.pc_range
        )  # 把voxels坐标系下的格子索引转换回世界坐标系下的坐标
        # fov_voxels[:, :3] = (
        #     fov_voxels[:, :3] + 0.5
        # ) * self.voxel_size  # fov_voxels的xyz现在算是index，0.5是取中心
        # fov_voxels[:, 0] += self.pc_range[0]
        # fov_voxels[:, 1] += self.pc_range[1]
        # fov_voxels[:, 2] += self.pc_range[2]
        # 以上两部主要是为了把点云中点的位置挪到对应occ voxel的中心

        return occ_pcd

    # 从一段视频流中还原出场景的三维空间占据情况（Occupancy）以及相机的运动轨迹
    def occ_gen_pipeline(self, input_path, pcd_save=False):
        """
        该方法可以根据整段episode视频得到估计的occ和相机姿态轨迹
        输入: input_path, 视频路径
        输出: self.occ, 估计的occ地图 (世界坐标系下)
            self.camera_pose, 估计的相机姿态轨迹 (世界坐标系下)
        """
        self.camera_intric = np.array(
            [[168.0498, 0.0, 240.0], [0.0, 192.79999, 135.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )  # 临时的， 实际应该根据input_path读出来
        # get the pcd with camera pose 三维重建，得到点云以及相机姿态轨迹
        pcd, self.camera_pose, self.norm_cam_ray = self.pcd_reconstuction(
            input_path, pcd_save
        )  # 点云pcd是世界坐标系下的点云，相机姿态轨迹self.camera_pose是世界坐标系下的相机姿态轨迹，norm_cam_ray是相机在相机坐标系下的单位射线方向
        # generate the occ map
        self.occ_pcd = self.pcd_to_occ(
            pcd
        )  # 目前在世界坐标系下, 且仍属于点云状态，只不过是进行了下采样，每个格子只有一个点，但是无label
        occ_voxels_visual_0, visual_mask_0 = self.check_visual_occ(
            self.occ_pcd, self.camera_pose[0]
        )  # 检查occ在相机0下是否可见
        if pcd_save:  # save the occ map
            save_path = input_path.split("videos/")[
                0
            ]  # input_path like: .../InternData-N1/vln_n1/traj_data/matterport3d_zed/gTV8FGcVJC9/trajectory_47/videos/chunk-000/observation.video.trajectory
            occ_pcd_cam_0 = self.convert_pointcloud_world_to_camera(
                self.occ_pcd, self.camera_pose[0]
            )  # 把世界坐标系下的occ转换到相机坐标系下
            write_ply(
                occ_pcd_cam_0[:, :3], path=os.path.join(self.save_path, "occ.ply")
            )  # 最后一维是颜色 是没有进行mask的，所有点都在相机视野内，一个全局地图
            occ_pcd_cam_0 = np.concatenate(
                [occ_pcd_cam_0, np.ones((occ_pcd_cam_0.shape[0], 1))], axis=1
            )
            np.save(os.path.join(self.save_path, "occ_pcd_cam0.npy"), occ_pcd_cam_0)

            occ_pcd_visual_0 = voxels_to_pcd(
                occ_voxels_visual_0, self.voxel_size, self.pc_range
            )  # 把“格子编号” 乘回 “格子大小”，加回 “偏移量”，变回 “米”（相机坐标系下的坐标）
            write_ply(
                occ_pcd_visual_0[:, :3],
                path=os.path.join(self.save_path, "occ_visual.ply"),
            )  # 最后一维是颜色, 是在相机0下可见的occ voxels，这是第 0 帧这一刻相机能看到的“表皮”
            occ_pcd_visual_0 = np.concatenate(
                [occ_pcd_visual_0, np.ones((occ_pcd_visual_0.shape[0], 1))], axis=1
            )
            np.save(
                os.path.join(self.save_path, "occ_pcd_visual_cam0.npy"),
                occ_pcd_visual_0,
            )

    def check_visual_occ(self, occ_pcd, camera_pose):
        """
        检查occ在指定相机下是否可见
        输入: occ, 估计的occ map (世界坐标系下)
            camera_pose, 估计的相机姿态轨迹 (世界坐标系下)
        输出:
        """
        occ_pcd_cam = self.convert_pointcloud_world_to_camera(
            occ_pcd, camera_pose
        )  # 把世界坐标系下的occ转换到相机坐标系下, shape: (-1, 3) （1.23, -0.55, 3.81）单位是米
        occ_voxels = pcd_to_voxels(
            occ_pcd_cam, self.voxel_size, self.pc_range
        )  # 把相机坐标系下的occ转换到voxels坐标系下, shape: (-1, 3)，“米”单位变成“格子编号”单位 （24， 12， 60）单位是格子序号
        occ_voxels = torch.tensor(
            occ_voxels, device=self.norm_cam_ray.device
        )  # 现在的格子地图是一个以相机为中心、但借用了世界地图尺寸刻度的“临时大盒子”
        occ_size_tensor = torch.tensor(
            self.config["occ_size"], device=self.norm_cam_ray.device
        )
        zero_size_tensor = torch.tensor([0, 0, 0], device=self.norm_cam_ray.device)
        mask_in_occ_range_max = (occ_voxels < occ_size_tensor).all(1)
        mask_in_occ_range_min = (occ_voxels >= zero_size_tensor).all(1)
        mask_in_occ_range = mask_in_occ_range_max * mask_in_occ_range_min
        occ_voxels = occ_voxels[mask_in_occ_range]
        max_distance = 200
        ray_cast_step_size = 1.0
        ray_position = torch.zeros(
            1, 3, device=self.norm_cam_ray.device
        )  # occ_pcd_cam已经在相机坐标系下了,激光发射起点就是相机光心 （0， 0， 0）单位是米
        ray_direction_norm = self.norm_cam_ray.reshape(
            -1, 3
        )  # 相机在相机坐标系下的基于相机光心的单位射线（N,3）
        pc_range_tensor = torch.tensor(
            self.pc_range[:3], device=self.norm_cam_ray.device
        )  # 世界坐标系下整个 3D 世界地图的“西南角最底端”那个原点坐标
        ray_position = (
            ray_position - pc_range_tensor
        ) / self.voxel_size  # 相机在voxels坐标系下的格子位置，把激光的“出发点”，从 0米 处，翻译成网格世界里的“第几号格子”，这是给当前帧建的local_occ地图
        # ray_direction_norm = (ray_direction_norm - pc_range_tensor) / self.voxel_size

        not_hit_ray = torch.ones(
            len(ray_direction_norm), device=ray_direction_norm.device
        ).bool()  # 这是一个超级长的开关列表（True/False）。每一根激光都有一个对应的开关，初始时都是True，代表这根激光还在“飞行”中
        ray_index_all = torch.arange(
            len(ray_direction_norm), device=ray_direction_norm.device
        )  # 这是一个超级长的索引列表，每个元素都是一根激光的索引
        camera_visible_mask_3d = torch.zeros(
            self.config["occ_size"], device=ray_direction_norm.device
        )  # 这个self.config['occ_size']就是我后面需要解决的合适的取值 是一个和地图一样大的 3D 空间，但它初始全是 0，只要激光钻过的地方，我们就在这张表上把它设为 1
        occ_voxels_3d = camera_visible_mask_3d.clone()
        occ_voxels_3d[
            occ_voxels[:, 0].long(), occ_voxels[:, 1].long(), occ_voxels[:, 2].long()
        ] = 1  # 现在要查某个格子有没有东西，只需要看 occ_voxels_3d[x, y, z] 是不是 1 即可，原来的occ_voxels是voxels坐标系下的坐标 查询麻烦
        occ_voxels_shape = torch.tensor(
            self.config["occ_size"]
        ).cuda()  # 地图的天花板（最大索引）
        zeros_3 = torch.zeros_like(occ_voxels_shape)  # 地板（全是0）
        ray_traverse_start = time.time()
        for step in range(
            int(max_distance / ray_cast_step_size) + 1
        ):  # 在voxels坐标系下算
            if not (not_hit_ray.any() and True):
                print(f"all rays hit the occupied voxel in step {step}!")
                break

            ray_position = (
                ray_position + ray_direction_norm * ray_cast_step_size
            )  # 格子坐标系下，激光沿着各自的方向前进一段距离
            voxel_coords = torch.floor(
                ray_position
            ).int()  # 把激光的“当前位置”转为体素坐标（取整）

            # check if the voxel_coords is in the range
            coord_valid = (voxel_coords >= zeros_3) & (
                voxel_coords < occ_voxels_shape
            )  # 激光算出来的格子编号必须在 [0, 0, 0] 到 [max, max, max] 之间
            position_valid = not_hit_ray & coord_valid.all(
                dim=1
            )  # 两个条件取“且”：1. 射线之前没撞过墙（not_hit_ray）；2. 射线现在没出界（coord_valid）

            # get the voxel index
            voxel_index = voxel_coords[
                position_valid
            ]  # 提取出这些有效射线的“格子坐标” --激光，现在分别踩在哪个格子上 [[10,2,3], [10,2,4], ...] 拿去查 occ_voxels_3d ，看看这些位置有没有1
            ray_selected_index = ray_index_all[
                position_valid
            ]  # 是哪束激光踩到的 布尔索引，筛选有效射线的索引 [0, 5, 12, 9999...]

            # set the visible occ flag (include free voxel)
            voxel_index_visible = voxel_index  # torch.unique(voxel_index, dim=0) # torch.unique is time wasting
            camera_visible_mask_3d[
                voxel_index_visible[:, 0],
                voxel_index_visible[:, 1],
                voxel_index_visible[:, 2],
            ] = 1  # 在 3D 掩膜矩阵里，把这些射线经过的地方全部标为 1，也就是看的见的地方

            # check if the voxel is occupied
            occ_label_selected = occ_voxels_3d[
                voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]
            ]  # 1 dim only 三维张量索引，获取有效射线当前体素的占用标签（1 = 有占用，0 = 无） 检查有效射线当前位置是否有占用 拿到的是一串 0（空）或 1
            occ_not_free = (
                occ_label_selected != self.free_label
            )  # bool tensor 判断拿到的标签是不是障碍物

            # get the ray index that hit the occupied voxel
            ray_selected_index = ray_selected_index[
                occ_not_free
            ]  # 定位！ 找出那些真正撞到障碍物的射线的 ID--- 从所有有效射线中筛选出那些撞到障碍物的射线的索引

            # get the occupied voxel index
            occ_occupied_index = voxel_index[occ_not_free]

            # record the ray that hit the occupied voxel
            not_hit_ray[ray_selected_index] = False  # 把这些撞墙射线的状态改为 False
        ray_traverse_end = time.time()
        print(f"ray traverse cost: {ray_traverse_end - ray_traverse_start}s")
        print("valid occ number: ", occ_voxels_3d.sum())
        occ_voxels_3d = (
            occ_voxels_3d * camera_visible_mask_3d
        )  # 只有既有墙、又能被看见的地方是 1
        occ_voxels = voxel2points(
            occ_voxels_3d, free_label=self.free_label
        )  # occ voxel coords (N, 3) 格子索引[[10, 20, 5], [10, 20, 6], ...]
        camera_visible_mask = voxel2points(
            camera_visible_mask_3d, free_label=self.free_label
        )  # 一个巨大的列表 (M, 3)，每个元素都是一个体素的坐标，代表地图上所有被激光“看”到的地方

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
        t_wc = -R_wc @ t_cw  # 等价于 -np.dot(R_wc, t_cw) 这一步不太懂

        # 步骤3：对每个点进行变换
        points_camera = (R_wc @ (points_world - t_cw).T).T

        return points_camera

    def convert_pointcloud_camera_to_world(self, points_camera, T_cw):
        """
        将相机坐标系下的点云转换到世界坐标系 (鲁棒版)
        :param points_camera: 相机坐标系点云，支持 Tensor 或 Numpy
        :param T_cw: 相机外参矩阵
        :return: 世界坐标系点云 (Numpy Array)
        """

        if points_camera is None or len(points_camera) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        if isinstance(points_camera, torch.Tensor):
            points_camera = points_camera.detach().cpu().numpy()

        if points_camera.ndim == 1:
            points_camera = points_camera.reshape(1, -1)

        R_cw = T_cw[:3, :3]
        t_cw = T_cw[:3, 3]

        # 公式: P_world = R * P_cam + t
        # (3,3) @ (3, N) -> (3, N).T -> (N, 3)
        points_world = (R_cw @ points_camera.T).T + t_cw

        return points_world.astype(np.float32)

    def occ_gen_all_pipeline(self, input_path, pcd_save=False):
        self.camera_intric = np.array(
            [[168.0498, 0.0, 240.0], [0.0, 192.79999, 135.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        # 1. 重建全局地图和轨迹
        pcd, self.camera_pose, self.norm_cam_ray = self.pcd_reconstuction(
            input_path, pcd_save
        )

        # 2. 生成全局点云
        self.occ_pcd = self.pcd_to_occ(pcd)

        # 3. 如果需要保存
        if pcd_save:
            print("Start processing sequence frames...")

            # --- 1. 创建混合显示 的文件夹 ---
            ply_dir = os.path.join(
                self.save_path, "ply_sequence"
            )  # 存相机坐标系下混合PLY
            if not os.path.exists(ply_dir):
                os.makedirs(ply_dir)

            npy_dir = os.path.join(
                self.save_path, "npy_sequence"
            )  # 存相机坐标系下混合NPY
            if not os.path.exists(npy_dir):
                os.makedirs(npy_dir)

            ply_world_dir = os.path.join(
                self.save_path, "ply_sequence_world"
            )  # 存世界坐标系下混合PLY
            if not os.path.exists(ply_world_dir):
                os.makedirs(ply_world_dir)

            npy_world_dir = os.path.join(
                self.save_path, "npy_sequence_world"
            )  # 存世界坐标系下混合NPY
            if not os.path.exists(npy_world_dir):
                os.makedirs(npy_world_dir)

            # --- 2. 创建【单独 Occ】的文件夹  ---
            occ_ply_dir = os.path.join(self.save_path, "occ_only_ply")  # 存纯Occ PLY
            if not os.path.exists(occ_ply_dir):
                os.makedirs(occ_ply_dir)

            occ_npy_dir = os.path.join(self.save_path, "occ_only_npy")  # 存纯Occ NPY
            if not os.path.exists(occ_npy_dir):
                os.makedirs(occ_npy_dir)

            total_frames = len(self.camera_pose)
            occ_pcd_cam_0 = self.convert_pointcloud_world_to_camera(
                self.occ_pcd, self.camera_pose[0]
            )  # 把世界坐标系下的occ转换到相机坐标系下
            write_ply(
                occ_pcd_cam_0[:, :3], path=os.path.join(self.save_path, "occ.ply")
            )  # 最后一维是颜色 是没有进行mask的，所有点都在相机视野内，一个全局地图

            for i in range(total_frames):
                current_pose = self.camera_pose[i]

                # ================= A. 计算数据 =================

                # 1. 计算当前帧可见 Occ (核心数据)
                occ_indices, _ = self.check_visual_occ(self.occ_pcd, current_pose)
                local_occ_cam = voxels_to_pcd(
                    occ_indices, self.voxel_size, self.pc_range
                )
                if isinstance(local_occ_cam, torch.Tensor):
                    local_occ_cam = local_occ_cam.detach().cpu().numpy()
                if local_occ_cam.shape[1] == 4:
                    local_occ_cam = local_occ_cam[:, :3]

                # 2. 计算当前帧相机坐标系下的可见Occ转换到世界坐标系下
                # 公式: P_world = R * P_cam + t
                local_occ_world = self.convert_pointcloud_camera_to_world(
                    local_occ_cam, current_pose
                )

                # 3. 计算背景和轨迹 但这个是在相机坐标系下的，我们不用它
                bg_cam = self.convert_pointcloud_world_to_camera(
                    self.occ_pcd, current_pose
                )
                if bg_cam.shape[1] == 4:
                    bg_cam = bg_cam[:, :3]

                traj_world = self.camera_pose[:, :3, 3]
                traj_cam = self.convert_pointcloud_world_to_camera(
                    traj_world, current_pose
                )
                if traj_cam.shape[1] == 4:
                    traj_cam = traj_cam[:, :3]

                # 我们还是使用世界坐标系下的背景和轨迹
                bg_world = self.pcd
                traj_current_world = self.camera_pose[0 : i + 1, :3, 3]
                if bg_world.shape[1] == 4:
                    bg_world = bg_world[:, :3]
                if traj_current_world.shape[1] == 4:
                    traj_current_world = traj_current_world[:, :3]

                # ================= B. 单独保存 Occ =================

                # 1. 保存单独的 Occ PLY
                if len(local_occ_cam) > 0:
                    # 纯绿色用于 PLY 显示
                    pure_occ_color = np.zeros_like(local_occ_cam)
                    pure_occ_color[:, 1] = 1.0
                    write_ply(
                        local_occ_cam,
                        pure_occ_color,
                        os.path.join(occ_ply_dir, f"occ_{i:04d}.ply"),
                    )
                else:
                    # 如果这帧没有occ，保存一个空文件或者跳过
                    # write_ply 需要至少一个点，这里简单处理，若空则跳过或存dummy
                    pass

                # 2. 保存单独的 Occ NPY
                if len(local_occ_cam) > 0:
                    # 格式: [X, Y, Z, Label=2]
                    # 加上 Label=2 是为了保持格式统一，方便后续脚本识别这是"Occ"
                    occ_npy_single = np.concatenate(
                        [local_occ_cam, np.full((local_occ_cam.shape[0], 1), 2)], axis=1
                    )
                    np.save(
                        os.path.join(occ_npy_dir, f"occ_{i:04d}.npy"),
                        occ_npy_single.astype(np.float32),
                    )
                else:
                    # 存个空数组，防止读取报错
                    np.save(
                        os.path.join(occ_npy_dir, f"occ_{i:04d}.npy"),
                        np.zeros((0, 4), dtype=np.float32),
                    )

                # ================= C. 保存相机坐标系下的混合数据  =================

                # 拼装颜色
                bg_color = np.ones_like(bg_cam) * 0.7
                traj_color = np.zeros_like(traj_cam)
                traj_color[:, 0] = 1.0

                occ_color = np.zeros_like(local_occ_cam)
                if len(occ_color) > 0:
                    occ_color[:, 1] = 1.0

                # 拼装列表
                points_list = [bg_cam, traj_cam]
                colors_list = [bg_color, traj_color]
                if len(local_occ_cam) > 0:
                    points_list.append(local_occ_cam)
                    colors_list.append(occ_color)

                final_points = np.concatenate(points_list, axis=0)
                final_colors = np.concatenate(colors_list, axis=0)

                # 保存混合 PLY
                write_ply(
                    final_points,
                    final_colors,
                    os.path.join(ply_dir, f"frame_{i:04d}.ply"),
                )

                # 保存混合 NPY (带标签)
                bg_npy = np.concatenate(
                    [bg_cam, np.zeros((bg_cam.shape[0], 1))], axis=1
                )  # Label 0
                traj_npy = np.concatenate(
                    [traj_cam, np.ones((traj_cam.shape[0], 1))], axis=1
                )  # Label 1

                if len(local_occ_cam) > 0:
                    occ_npy = np.concatenate(
                        [local_occ_cam, np.full((local_occ_cam.shape[0], 1), 2)], axis=1
                    )  # Label 2
                    final_npy_data = np.concatenate([bg_npy, traj_npy, occ_npy], axis=0)
                else:
                    final_npy_data = np.concatenate([bg_npy, traj_npy], axis=0)

                np.save(
                    os.path.join(npy_dir, f"frame_{i:04d}.npy"),
                    final_npy_data.astype(np.float32),
                )

                # ================= D. 保存世界坐标系下的混合数据 (支持真彩色) =================

                bg_world_dense = self.pcd  # (N, 3)

                # 获取背景颜色
                if hasattr(self, "pcd_color"):
                    bg_color_dense = self.pcd_color  # (N, 3) 假设是 0-1 的 float
                else:
                    bg_color_dense = np.ones_like(bg_world_dense) * 0.7

                # 维度对齐
                min_len = min(len(bg_world_dense), len(bg_color_dense))
                bg_world_dense = bg_world_dense[:min_len]
                bg_color_dense = bg_color_dense[:min_len]

                # A. 构造背景 NPY: [x, y, z, r, g, b, 0]
                bg_label = np.zeros((min_len, 1))  # Label 0
                bg_npy = np.concatenate(
                    [bg_world_dense, bg_color_dense, bg_label], axis=1
                )

                # B. 构造轨迹 NPY: [x, y, z, 0, 0, 1, 1] (蓝色占位，视频脚本会重绘，但格式要对)
                # 这里填什么颜色不重要，因为视频脚本会统一画轨迹球，但为了格式统一，填蓝色
                traj_len = len(traj_current_world)
                if traj_len > 0:
                    traj_rgb = np.tile([0.0, 0.0, 1.0], (traj_len, 1))  # 全蓝
                    traj_label = np.ones((traj_len, 1))  # Label 1
                    traj_npy = np.concatenate(
                        [traj_current_world, traj_rgb, traj_label], axis=1
                    )
                else:
                    traj_npy = np.zeros((0, 7))

                # C. 构造 OCC NPY: [x, y, z, 0.5, 0.5, 0.5, 2] (灰色占位)
                occ_len = len(local_occ_world)
                if occ_len > 0:
                    occ_rgb = np.tile([0.5, 0.5, 0.5], (occ_len, 1))  # 全灰
                    occ_label = np.full((occ_len, 1), 2)  # Label 2
                    occ_npy = np.concatenate(
                        [local_occ_world, occ_rgb, occ_label], axis=1
                    )
                else:
                    occ_npy = np.zeros((0, 7))

                final_npy_data = np.concatenate([bg_npy, traj_npy, occ_npy], axis=0)

                # 保存为 (N, 7) 的 NPY
                np.save(
                    os.path.join(npy_world_dir, f"frame_{i:04d}_world.npy"),
                    final_npy_data.astype(np.float32),
                )

                # 同时保存 PLY (用于 MeshLab 查看)
                # PLY只需要 points 和 colors，不需要 label
                # 这里的 final_npy_data 前3列是xyz，中间3列是rgb
                write_ply(
                    final_npy_data[:, :3],
                    final_npy_data[:, 3:6],
                    os.path.join(ply_world_dir, f"frame_{i:04d}_world.ply"),
                )

                if i % 10 == 0:
                    print(f"Processed frame {i}/{total_frames}")


# ================= 主函数 =================

if __name__ == "__main__":

    config_path = "./occ/config.yaml"
    save_dir = "./outputs"
    model_dir = "./ckpt"

    input_path = "/mnt/data/huangbinling/project/occgen/inputs"
    video_name = "office.mp4"

    # 初始化
    generator = DataGenerator(config_path, save_dir, model_dir)

    # 运行 Pipeline 并开启保存 (pcd_save=True)
    # 这会在 outputs/frame_sequence/ 下生成几百个 .ply 文件
    generator.occ_gen_all_pipeline(os.path.join(input_path, video_name), pcd_save=True)

"""
if __name__=="__main__":

    config_path='./occ/config.yaml'
    save_dir="./outputs"
    model_dir = "./ckpt"

    input_path = "/mnt/data/huangbinling/project/occgen/inputs"
    video_name = "1.mp4"
    generator = DataGenerator(config_path, save_dir, model_dir)
    generator.occ_gen_pipeline(os.path.join(input_path, video_name), pcd_save=True)
"""
