import torch
import open3d as o3d
import numpy as np
import os.path as osp
import os
import math
import cv2
from PIL import Image
import torch

from torchvision import transforms

import matplotlib.pyplot as plt
from copy import deepcopy

from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy.interpolate import CubicSpline






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



def estimate_intrinsics(coords):
    """
    Args:
        coords: Tensor of shape (H, W, 3), camera plane points (X, Y, Z)
    Returns:
        K: (3, 3) intrinsics estimation result from camera plane points
    """
    h, w, _ = coords.shape
    device = coords.device
 
    v, u = torch.meshgrid(torch.arange(h, device=device), 
                          torch.arange(w, device=device), 
                          indexing='ij')
    
    u = u.flatten().float()
    v = v.flatten().float()
 
    X = coords[..., 0].flatten()
    Y = coords[..., 1].flatten()
    Z = coords[..., 2].flatten()
 
    mask = Z > 0
    u, v, X, Y, Z = u[mask], v[mask], X[mask], Y[mask], Z[mask]

    x_prime = X / Z
    y_prime = Y / Z

    # A @ x = B
    # fx, cx: [x' 1] @ [fx, cx]^T = u
    ones = torch.ones_like(x_prime)
    
    # solve x part (fx, cx)
    A_u = torch.stack([x_prime, ones], dim=1)
    sol_u = torch.linalg.lstsq(A_u, u).solution
    fx, cx = sol_u[0], sol_u[1]

    # solve y part (fy, cy)
    A_v = torch.stack([y_prime, ones], dim=1)
    sol_v = torch.linalg.lstsq(A_v, v).solution
    fy, cy = sol_v[0], sol_v[1]

    # assemble intrinsics matrix K
    K = torch.tensor([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], device=device)

    return K

def voxel2points(pred_occ, mask_camera=None, free_label=0): 
    
    d, h, w = pred_occ.shape  
    x = torch.linspace(0, d - 1, d, device=pred_occ.device, dtype=pred_occ.dtype)
    y = torch.linspace(0, h - 1, h, device=pred_occ.device, dtype=pred_occ.dtype)
    z = torch.linspace(0, w - 1, w, device=pred_occ.device, dtype=pred_occ.dtype)
    X, Y, Z = torch.meshgrid(x, y, z) 
    vv = torch.stack([X, Y, Z, pred_occ], dim=-1)
 
    valid_mask = pred_occ != free_label 
    if mask_camera is not None:
        mask_camera = mask_camera.to(device=pred_occ.device, dtype=torch.bool)
        valid_mask = torch.logical_and(valid_mask, mask_camera)
 
    fov_voxels = vv[valid_mask]
 
    fov_voxels = fov_voxels.to(dtype=torch.float32)
 
    return fov_voxels


def pcd_to_voxels(pcd, voxel_size, pcd_range):
    # 要注意pcd的xyz和pcd_range的要对应
    occ_voxels = pcd.clone() if isinstance(pcd, torch.Tensor) else pcd.copy()
    occ_voxels[:, 0] = occ_voxels[:, 0] - pcd_range[0]
    occ_voxels[:, 1] = occ_voxels[:, 1] - pcd_range[1]
    occ_voxels[:, 2] = occ_voxels[:, 2] - pcd_range[2]
    occ_voxels[:, :3] = occ_voxels[:, :3] / voxel_size - 0.5
    return np.floor(occ_voxels).astype(np.int32)


def voxels_to_pcd(occ_voxels, voxel_size, pcd_range):
    pcd = occ_voxels.clone() if isinstance(occ_voxels, torch.Tensor) else occ_voxels.copy()
    pcd[:, :3] = (pcd[:, :3] + 0.5) * voxel_size
    pcd[:, 0] = pcd[:, 0] + pcd_range[0]
    pcd[:, 1] = pcd[:, 1] + pcd_range[1]
    pcd[:, 2] = pcd[:, 2] + pcd_range[2]
    return pcd


def homogenize_points(
    points,
):
    """Convert batched points (xyz) to (xyz1)."""
    if isinstance(points, torch.Tensor):
        return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    elif isinstance(points, np.ndarray):
        return np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    else:
        raise TypeError(
            f"points must be torch.Tensor or np.ndarray, but got {type(points)}"
        )


def plot_camera_poses(extrinsics_original, extrinsics_interp):
    """可视化相机位置和姿态"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 提取原始和插值后的相机位置
    pos_original = extrinsics_original[:, :3, 3]
    pos_interp = extrinsics_interp[:, :3, 3]

    # 绘制位置
    ax.scatter(
        pos_original[:, 0],
        pos_original[:, 1],
        pos_original[:, 2],
        c="red",
        s=100,
        label="original",
        zorder=5,
    )
    ax.plot(
        pos_interp[:, 0],
        pos_interp[:, 1],
        pos_interp[:, 2],
        c="blue",
        linewidth=2,
        label="interpolated",
        zorder=3,
    )

    # 绘制相机姿态（x轴方向）
    for ext in extrinsics_original[::1]:  # 每隔1个绘制原始姿态
        pos = ext[:3, 3]
        x_axis = ext[:3, 0] * 0.2  # 缩放轴长
        ax.quiver(
            pos[0],
            pos[1],
            pos[2],
            x_axis[0],
            x_axis[1],
            x_axis[2],
            color="red",
            arrow_length_ratio=0.1,
        )

    for ext in extrinsics_interp[::5]:  # 每隔5个绘制插值姿态
        pos = ext[:3, 3]
        x_axis = ext[:3, 0] * 0.2
        ax.quiver(
            pos[0],
            pos[1],
            pos[2],
            x_axis[0],
            x_axis[1],
            x_axis[2],
            color="blue",
            arrow_length_ratio=0.1,
            alpha=0.5,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.title("相机外参插值结果")
    # plt.show()
    # 保存为图像
    plt.savefig("camera_poses.png", dpi=300, bbox_inches="tight")


def interpolate_extrinsics(extrinsics, x_original, x_target):
    """
    对相机外参矩阵进行插值，同时保证一定的外推性，旋转分量不做额外外推计算，直接取最近的一个外参
    参数：
        extrinsics: 原始外参矩阵，shape=(N,4,4)
        x_original: 原始外参对应的坐标/时间，shape=(N,)
        x_target: 目标插值点的坐标/时间，shape=(M,)
    返回：
        extrinsics_interp: 插值后的外参矩阵，shape=(M,4,4)
    """
    N = len(extrinsics)
    if N < 2:
        raise ValueError("至少需要2个外参矩阵进行插值")

    # 步骤1：为平移向量构建三次样条插值器（x/y/z轴分别构建）
    # 构建3个轴的样条：trans_splines[0]→x轴，trans_splines[1]→y轴，trans_splines[2]→z轴
    trans = np.array(extrinsics[:, :3, 3])  # (N,3)
    trans_splines = [
        CubicSpline(
            x_original, trans[:, 0], bc_type="natural"
        ),  # natural：自然样条，端点二阶导为0
        CubicSpline(x_original, trans[:, 1], bc_type="natural"),
        CubicSpline(x_original, trans[:, 2], bc_type="natural"),
    ]
    # 一次性计算所有目标点的平移插值（向量化，效率更高）
    trans_interp = np.vstack(
        [
            trans_splines[0](x_target),
            trans_splines[1](x_target),
            trans_splines[2](x_target),
        ]
    ).T  # shape=(M,3)

    # 步骤2：对每个目标点的旋转分量做插值
    R_original = Rotation.from_matrix(extrinsics[:, :3, :3])
    slerper = Slerp(x_original, R_original)
    x_target_inner = x_target[x_target < x_original[-1]]  # 需要外推的单独处理，就是采样后的帧可能没有包含原有的最后一些帧，就对最后一些帧进行外推
    R_slerp = slerper(x_target_inner).as_matrix()
    # 外推：repeat最后一个区间的旋转矩阵
    R_slerp_ext = np.repeat(
        R_slerp[-1][None, ...], len(x_target) - len(x_target_inner), axis=0
    )
    R_slerp = np.concatenate([R_slerp, R_slerp_ext], axis=0)
    extrinsics_interp = np.zeros((len(x_target), 4, 4))
    extrinsics_interp[:, -1, -1] = 1
    extrinsics_interp[:, :3, :3] = R_slerp
    extrinsics_interp[:, :3, 3] = trans_interp

    return extrinsics_interp


def convert_pointcloud_world_to_camera(points_world, T_cw):
    """
    将世界坐标系下的点云转换到相机坐标系
    :param points_world: 世界坐标系点云，shape=(N, 3)
    :param T_cw: 相机外参矩阵（相机→世界），shape=(4, 4)
    :return: 相机坐标系点云，shape=(N, 3)
    """
    points_world = points_world.astype(np.float32)
    # 步骤1：提取外参的旋转和平移
    R_cw = T_cw[:3, :3]
    t_cw = T_cw[:3, 3]

    # 步骤2：计算世界→相机的旋转和平移
    R_wc = R_cw.T  # 旋转矩阵的逆=转置
    t_wc = -R_wc @ t_cw  # 等价于 -np.dot(R_wc, t_cw)

    # 步骤3：对每个点进行变换
    points_camera = (R_wc @ (points_world - t_cw).T).T

    return points_camera


def element_isin(tensor1, tensor2, invert=False):
    """
    tensor1: N x k
    tensor2: M x k
    return : N x 1, bool
    """
    eq_per_element = torch.eq(tensor1.unsqueeze(1), tensor2.unsqueeze(0))
    eq_per_tensor = eq_per_element.all(2)
    eq_per_tensor_isin = eq_per_tensor.any(1)
    if invert:
        eq_per_tensor_isin = ~eq_per_tensor_isin
    return eq_per_tensor_isin


def run_poisson(pcd, depth, n_threads, min_density=None):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=8
    )

    # Post-process the mesh
    if min_density:
        vertices_to_remove = densities < np.quantile(densities, min_density)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()

    return mesh, densities


def create_mesh_from_map(
    buffer, depth, n_threads, min_density=None, point_cloud_original=None
):

    if point_cloud_original is None:
        pcd = buffer_to_pointcloud(buffer)
    else:
        pcd = point_cloud_original

    return run_poisson(pcd, depth, n_threads, min_density)


def buffer_to_pointcloud(buffer, compute_normals=False):
    pcd = o3d.geometry.PointCloud()
    for cloud in buffer:
        pcd += cloud
    if compute_normals:
        pcd.estimate_normals()

    return pcd


def preprocess_cloud(
    pcd,
    max_nn=20,
    normals=True,
):

    cloud = deepcopy(pcd)
    if normals:
        params = o3d.geometry.KDTreeSearchParamKNN(max_nn)
        cloud.estimate_normals(params)
        cloud.orient_normals_towards_camera_location()

    return cloud


def preprocess(pcd, config, normals=False):
    return preprocess_cloud(pcd, config["max_nn"], normals=normals)


def nn_correspondance(verts1, verts2):  # unuse
    """for each vertex in verts2 find the nearest vertex in verts1

    Args:
        nx3 np.array's
    Returns:
        ([indices], [distances])

    """

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances


def point_transform_3d_batch(loc, M):
    """
    Transform a 3D point using a 4x4 matrix
    loc: Nx3 array point location
    out: Nx3 array transformed point location
    """
    hwc_input = len(loc.shape) == 3
    if hwc_input:  # h, w, c input
        h, w = loc.shape[0], loc.shape[1]
        loc = loc.reshape(-1, 3)

    if loc.device.type == 'cuda':
        ones = torch.ones((loc.shape[0], 1), device=loc.device)
        point = torch.concat((loc, ones), axis=1)
        point = point.unsqueeze(2)
        M = M.unsqueeze(0)
    else:
        point = np.concatenate((loc, np.ones((loc.shape[0], 1))), axis=1)
        point = point[:, :, np.newaxis]
        M = M[np.newaxis, :, :]

    point_transformed = M @ point
    point_transformed = point_transformed[:, :, 0]

    # normalize, 其实最后一位就是1.0
    point_transformed[:, 0] /= point_transformed[:, 3]
    point_transformed[:, 1] /= point_transformed[:, 3]
    point_transformed[:, 2] /= point_transformed[:, 3]
    point_transformed = point_transformed[:, :3]
    if hwc_input:
        point_transformed = point_transformed.reshape(h, w, 3)
    return point_transformed


def point_transform_2d_batch(loc, M):
    """
    Transform a 2D point using a 3x3 matrix
    loc: Nx2 array point location
    out: Nx2 array transformed point location
    """
    hwc_input = len(loc.shape) == 3
    if hwc_input:  # h, w, c input
        h, w = loc.shape[0], loc.shape[1]
        loc = loc.reshape(-1, 2)

    if loc.device.type == 'cuda':
        ones = torch.ones((loc.shape[0], 1), device=loc.device)
        point = torch.concat((loc, ones), axis=1)
        point = point.unsqueeze(2)
        M = M.unsqueeze(0)
    else:
        point = np.concatenate((loc, np.ones((loc.shape[0], 1))), axis=1)
        point = point[:, :, np.newaxis]
        M = M[np.newaxis, :, :]

    point_transformed = M @ point
    point_transformed = point_transformed[:, :, 0]

    # normalize, 其实最后一位就是1.0
    point_transformed[:, 0] /= point_transformed[:, 2]
    point_transformed[:, 1] /= point_transformed[:, 2]
    point_transformed = point_transformed[:, :2]
    if hwc_input:
        point_transformed = point_transformed.reshape(h, w, 2)

    return point_transformed


def quaternion_to_matrix(quat, to_wxyz=False):
    if to_wxyz:
        quat = np.roll(quat, -1)
    r = Rotation.from_quat(quat)  # 顺序为 (x, y, z, w)
    rot = r.as_matrix()
    return rot


def matrix_to_quaternion(matrix, to_wxyz=False):
    r = Rotation.from_matrix(matrix)
    quat = r.as_quat()
    if to_wxyz:
        quat = np.roll(quat, 1)
    return quat


def ray_casting(voxels, origin, direction, max_distance):
    """
    检查光线投射过程中是否有体素遮挡。

    :param voxels: 3D numpy array，表示体素网格（True表示存在体素）。
    :param origin: 光线的起点坐标（x, y, z）。
    :param direction: 光线的方向向量（已归一化）。
    :param max_distance: 最大检查距离。
    :return: 第一个被击中的体素位置或None如果没有体素被击中。
    """
    position = np.array(origin, dtype=float)
    step_size = 0.1  # 步长，可以调整以提高精度或性能

    for _ in range(int(max_distance / step_size)):
        position += direction * step_size
        voxel_coords = np.floor(position).astype(int)

        # 检查坐标是否在网格范围内
        if (
            0 <= voxel_coords[0] < voxels.shape[0]
            and 0 <= voxel_coords[1] < voxels.shape[1]
            and 0 <= voxel_coords[2] < voxels.shape[2]
        ):

            if voxels[tuple(voxel_coords)]:
                return tuple(voxel_coords)

    return None


def convert_pointcloud_camera_to_world(points_camera, T_cw):
        """
        将相机坐标系下的点云转换回世界坐标系
        :param points_camera: 相机坐标系点云 (N, 3)
        :param T_cw: 相机外参 (4, 4)，即 self.camera_pose[i]
        :return: 世界坐标系点云 (N, 3)
        """
        # T_cw 通常是 Camera-to-World 的变换矩阵 (Pose)
        # P_world = R * P_cam + t
        R = T_cw[:3, :3]
        t = T_cw[:3, 3]
        
        # 矩阵乘法：(R @ P_cam.T).T + t
        points_world = (R @ points_camera.T).T + t
        return points_world

def load_depths_as_tensor(path='data/truck', interval=1, PIXEL_LIMIT=255000):
    """
    Loads depths from a directory or video, resizes them to a uniform size,
    then converts and stacks them into a single [N, 3, H, W] PyTorch tensor.
    """
    sources = [] 
    
    # --- 1. Load depth paths or video frames ---
    if osp.isdir(path):
        print(f"Loading depths from directory: {path}")
        filenames = sorted([x for x in os.listdir(path) if x.lower().endswith(('.png', '.jpg', '.jpeg'))])
        for i in range(0, len(filenames), interval):
            img_path = osp.join(path, filenames[i])
            try:
                sources.append(Image.open(img_path))
            except Exception as e:
                print(f"Could not load depth {filenames[i]}: {e}")
    else:
        raise ValueError(f"Unsupported path. Must be a directory: {path}")

    if not sources:
        print("No depths found or loaded.")
        return torch.empty(0)

    print(f"Found {len(sources)} depths/frames. Processing...")

    # --- 2. Determine a uniform target size for all depths based on the first depth ---    
    # This is necessary to ensure all tensors have the same dimensions for stacking.
    first_img = sources[0]
    W_orig, H_orig = first_img.size
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target: k -= 1
        else: m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    # --- 3. Resize images and convert them to tensors in the [0, 1] range ---
    tensor_list = []
    # Define a transform to convert a PIL Image to a CxHxW tensor and normalize to [0,1]
    to_tensor_transform = transforms.ToTensor()
    
    for img_pil in sources:
        try:
            # Resize to the uniform target size
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            # Convert to tensor
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        print("No images were successfully processed.")
        return torch.empty(0)

    # --- 4. Stack the list of tensors into a single [N, C, H, W] batch tensor ---
    return torch.stack(tensor_list, dim=0)
    
def load_images_as_tensor(path='data/truck', interval=1, PIXEL_LIMIT=255000):
    """
    Loads images from a directory or video, resizes them to a uniform size,
    then converts and stacks them into a single [N, 3, H, W] PyTorch tensor.
    """
    sources = [] 
    all_frames = -1
    # --- 1. Load image paths or video frames ---
    if osp.isdir(path):
        print(f"Loading images from directory: {path}")
        filenames = sorted([x for x in os.listdir(path) if x.lower().endswith(('.png', '.jpg', '.jpeg'))])
        all_frames = len(filenames)
        for i in range(0, len(filenames), interval):
            img_path = osp.join(path, filenames[i])
            try:
                sources.append(Image.open(img_path).convert('RGB'))
            except Exception as e:
                print(f"Could not load image {filenames[i]}: {e}")
    elif path.lower().endswith('.mp4'):
        print(f"Loading frames from video: {path}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): raise IOError(f"Cannot open video file: {path}")
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sources.append(Image.fromarray(rgb_frame))
            frame_idx += 1
        cap.release()
        all_frames = frame_idx
    else:
        raise ValueError(f"Unsupported path. Must be a directory or a .mp4 file: {path}")

    if not sources:
        print("No images found or loaded.")
        return torch.empty(0), 0

    print(f"Found {len(sources)} images/frames. Processing...")

    # --- 2. Determine a uniform target size for all images based on the first image ---
    # This is necessary to ensure all tensors have the same dimensions for stacking.
    first_img = sources[0]
    W_orig, H_orig = first_img.size
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target: k -= 1
        else: m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    # --- 3. Resize images and convert them to tensors in the [0, 1] range ---
    tensor_list = []
    # Define a transform to convert a PIL Image to a CxHxW tensor and normalize to [0,1]
    to_tensor_transform = transforms.ToTensor()
    
    for img_pil in sources:
        try:
            # Resize to the uniform target size
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            # Convert to tensor
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        print("No images were successfully processed.")
        return torch.empty(0), 0

    # --- 4. Stack the list of tensors into a single [N, C, H, W] batch tensor ---
    return torch.stack(tensor_list, dim=0), all_frames
