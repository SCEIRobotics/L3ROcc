import math
import os
import os.path as osp
from copy import deepcopy
from itertools import islice

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from pyquaternion import Quaternion
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp
from torchvision import transforms


def compute_rigid_transform(source_points, target_points):
    """
    Computes the rigid transformation (rotation matrix R and translation vector t)
    from source points to target points.

    Args:
        source_points: Source point cloud, numpy array of shape (n, 3).
        target_points: Target point cloud, numpy array of shape (n, 3), corresponding to source points.

    Returns:
        R: 3x3 rotation matrix.
        t: 3x1 translation vector.
        rmse: Root Mean Square Error, evaluating the transformation accuracy.
    """
    # Check if input points are valid
    if source_points.shape != target_points.shape:
        raise ValueError("Source and target points must have the same shape.")
    if len(source_points) < 3:
        raise ValueError(
            "At least 3 points are required to compute the transformation."
        )

    # Step 1: Compute centroids
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)

    # Step 2: Center the point sets (remove centroids)
    source_centered = source_points - centroid_source
    target_centered = target_points - centroid_target

    # Step 3: Construct the covariance matrix H
    H = np.dot(source_centered.T, target_centered)

    # Step 4: Perform Singular Value Decomposition (SVD) on H
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Step 5: Compute rotation matrix and handle reflection
    R = np.dot(V, U.T)

    # Ensure the determinant of R is 1 (Right-handed coordinate system)
    if np.linalg.det(R) < 0:
        V[:, 2] *= -1  # Correct the V matrix
        R = np.dot(V, U.T)

    # Step 6: Compute translation vector
    t = centroid_target - np.dot(R, centroid_source)

    # Compute RMSE to evaluate accuracy
    transformed_source = np.dot(source_points, R.T) + t
    errors = np.linalg.norm(transformed_source - target_points, axis=1)
    rmse = np.sqrt(np.mean(errors**2))

    raw_errors = np.linalg.norm(source_points - target_points, axis=1)
    raw_rmse = np.sqrt(np.mean(raw_errors**2))

    return R, t, rmse, raw_rmse


def compute_similarity_transform(source_points, target_points, get_rmse=True):
    """
    Computes the similarity transformation (scale s, rotation R, translation t)
    from source points to target points.

    Args:
        source_points: Source point cloud, numpy array of shape (n, 3).
        target_points: Target point cloud, numpy array of shape (n, 3), corresponding to source points.

    Returns:
        s: Scale factor.
        R: 3x3 rotation matrix.
        t: 3x1 translation vector.
        rmse: Root Mean Square Error.
    """
    # Check input validity
    if source_points.shape != target_points.shape:
        raise ValueError("Source and target points must have the same shape.")
    if len(source_points) < 3:
        raise ValueError("At least 3 non-collinear points are required.")

    # Step 1: Compute centroids
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)

    # Step 2: Center point sets
    source_centered = source_points - centroid_source  # Shape (n, 3)
    target_centered = target_points - centroid_target  # Shape (n, 3)

    # Step 3: Compute scale factor s
    # Numerator: Sum of squared norms of centered target points
    sum_q_sq = np.sum(np.linalg.norm(target_centered, axis=1) ** 2)
    # Denominator: Sum of squared norms of centered source points
    sum_p_sq = np.sum(np.linalg.norm(source_centered, axis=1) ** 2)

    if sum_p_sq == 0:
        raise ValueError(
            "Source points cannot be a single point (scale cannot be computed)."
        )

    s = np.sqrt(sum_q_sq / sum_p_sq)  # Scale factor (sqrt ensures s > 0)

    # Step 4: Normalize target points scale (remove scale effect)
    target_scaled = target_centered / s  # Shape (n, 3)

    # Step 5: Compute rotation matrix R using SVD (same as rigid transform)
    H = np.dot(source_centered.T, target_scaled)  # Covariance matrix (3,3)
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    R = np.dot(V, U.T)

    # Handle reflection case (ensure determinant is 1)
    if np.linalg.det(R) < 0:
        V[:, 2] *= -1
        R = np.dot(V, U.T)

    # Step 6: Compute translation vector t (includes scale)
    t = centroid_target - s * np.dot(R, centroid_source)

    # Compute RMSE
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
    A standalone RANSAC implementation for point cloud registration to find the optimal
    rotation matrix R, translation vector t, and scale factor s.

    Args:
    src_pts (np.ndarray): Source point cloud, shape (N, 3).
    dst_pts (np.ndarray): Target point cloud, shape (N, 3).
    threshold (float): Reprojection error threshold for inlier determination.
    max_iterations (int): Maximum number of RANSAC iterations.
    min_inliers (int): Minimum number of inliers required.

    Returns:
    best_R (np.ndarray): Best rotation matrix (3x3).
    best_t (np.ndarray): Best translation vector (3,).
    best_s (float): Best scale factor.
    best_mask (np.ndarray): Boolean array marking inliers.
    """

    def warp_3d(pts, R, t, s):
        """
        Applies transformation to 3D point cloud: rotation R, translation t, scaling s.
        """
        return s * np.dot(pts, R.T) + t

    if src_pts.shape != dst_pts.shape or src_pts.shape[0] < min_inliers:
        raise ValueError(
            f"Input points must have the same shape and contain at least {min_inliers} points."
        )

    num_points = src_pts.shape[0]
    best_inliers_count = 0
    best_R = None
    best_t = None
    best_s = None
    best_mask = np.zeros(num_points, dtype=bool)

    for _ in range(max_iterations):
        # 1. Random sampling: Select points
        # For simplicity, select 'min_inliers' random points; assume non-collinear for now.
        import random
        sample_indices = random.sample(range(num_points), min_inliers)
        src_sample = src_pts[sample_indices]
        dst_sample = dst_pts[sample_indices]

        # Compute R, t, s
        s, R, t, _, _ = compute_similarity_transform(
            src_sample, dst_sample, get_rmse=False
        )

        # 2. Apply transformation to all points
        transformed_src = warp_3d(src_pts, R, t, s)

        # 3. Compute reprojection errors
        errors = np.linalg.norm(transformed_src - dst_pts, axis=1)
        inliers = errors < threshold

        # 4. Update best model
        inliers_count = np.sum(inliers)
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_R = R
            best_t = t
            best_s = s
            best_mask = inliers

        # 5. Early termination if sufficient inliers are found
        if inliers_count >= min_inliers:
            break

    # Re-compute final R, t, s using all best inliers
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

    v, u = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )

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
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device)

    return K


def voxel2points(pred_occ, mask_camera=None, free_label=0):
    """
    Convert voxel occupancy to 3D points.

    Args:
        pred_occ: Tensor of shape (D, H, W), voxel occupancy predictions.
        mask_camera: Optional, Tensor of shape (D, H, W), camera mask.
        free_label: Integer, label value for free space.

    Returns:
        fov_voxels: Tensor of shape (N, 4), 3D points (X, Y, Z, occ) in the camera FOV.
    """
    d, h, w = pred_occ.shape

    x = torch.linspace(0, d - 1, d, device=pred_occ.device, dtype=torch.int32)
    y = torch.linspace(0, h - 1, h, device=pred_occ.device, dtype=torch.int32)
    z = torch.linspace(0, w - 1, w, device=pred_occ.device, dtype=torch.int32)

    X, Y, Z = torch.meshgrid(x, y, z)

    vv = torch.stack([X, Y, Z, pred_occ], dim=-1)

    valid_mask = pred_occ != free_label

    if mask_camera is not None:
        mask_camera = mask_camera.to(device=pred_occ.device, dtype=torch.bool)
        valid_mask = torch.logical_and(valid_mask, mask_camera)

    fov_voxels = vv[valid_mask]

    fov_voxels = fov_voxels.to(dtype=torch.float32)

    return fov_voxels


def pcd_to_voxels(pcd, voxel_size, pc_range):
    """
    Convert point cloud to voxel occupancy.

    Args:
        pcd: Tensor of shape (N, 3), point cloud (X, Y, Z).
        voxel_size: Float, voxel size.
        pcd_range: List of 3 floats, [x_min, y_min, z_min] defining the range of the point cloud.

    Returns:
        occ_voxels: Tensor of shape (D, H, W), voxel occupancy.
    """

    # Note: Ensure correspondence between pcd xyz and pcd_range
    if isinstance(pcd, torch.Tensor):
        device = pcd.device
        dtype = pcd.dtype

        if isinstance(pc_range, torch.Tensor):
            range_min = pc_range[:3].to(device)
        else:
            range_min = torch.tensor(pc_range[:3], device=device, dtype=dtype)

        # 计算索引
        voxel_indices = torch.floor((pcd - range_min) / voxel_size).long()
        return voxel_indices

    else:
        range_np = pc_range

        if isinstance(range_np, torch.Tensor):
            range_np = range_np.detach().cpu().numpy()
        if isinstance(pcd, torch.Tensor):
            pcd_np = pcd.detach().cpu().numpy()
        else:
            pcd_np = pcd.copy()

        pcd_np[:, 0] = (pcd_np[:, 0] - range_np[0]) / voxel_size
        pcd_np[:, 1] = (pcd_np[:, 1] - range_np[1]) / voxel_size
        pcd_np[:, 2] = (pcd_np[:, 2] - range_np[2]) / voxel_size

        return np.floor(pcd_np).astype(np.int32)
 
def voxels_to_pcd(occ_voxels, voxel_size, pcd_range):
    """
    Convert voxel indices to point cloud coordinates.
    Compatible with both NumPy arrays and PyTorch Tensors (CPU/GPU).

    Args:
        occ_voxels: (N, 3) Int indices of occupied voxels.
        voxel_size: Float.
        pcd_range: List of 3 floats [x_min, y_min, z_min].

    Returns:
        pcd: (N, 3) Float coordinates. Type matches input (Tensor -> Tensor, Numpy -> Numpy).
    """

    if isinstance(occ_voxels, torch.Tensor):
        pcd = occ_voxels.clone().float()

        pcd[:, :3] = (pcd[:, :3] + 0.5) * voxel_size
        pcd[:, 0] += pcd_range[0]
        pcd[:, 1] += pcd_range[1]
        pcd[:, 2] += pcd_range[2]

        return pcd
    else:
        pcd = occ_voxels.copy().astype(np.float32)

        pcd[:, :3] = (pcd[:, :3] + 0.5) * voxel_size

        pcd[:, 0] += pcd_range[0]
        pcd[:, 1] += pcd_range[1]
        pcd[:, 2] += pcd_range[2]

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
    """Visualizes camera positions and poses."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Extract original and interpolated camera positions
    pos_original = extrinsics_original[:, :3, 3]
    pos_interp = extrinsics_interp[:, :3, 3]

    # Plot positions
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

    # Plot camera poses (x-axis direction)
    for ext in extrinsics_original[::1]:  # Plot original poses at every step
        pos = ext[:3, 3]
        x_axis = ext[:3, 0] * 0.2  # Scale axis length
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

    for ext in extrinsics_interp[::5]:  # Plot interpolated poses every 5 steps
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
    plt.title("Camera Extrinsics Interpolation")
    # plt.show()
    # Save as image
    plt.savefig("camera_poses.png", dpi=300, bbox_inches="tight")


def interpolate_extrinsics(extrinsics, x_original, x_target):
    """
    Interpolates camera extrinsic matrices while ensuring extrapolation capability.
    Rotation components are not extrapolated via calculation but use the nearest valid frame.

    Args:
        extrinsics: Original extrinsic matrices, shape=(N,4,4).
        x_original: Coordinates/times corresponding to original extrinsics, shape=(N,).
        x_target: Target coordinates/times for interpolation, shape=(M,).

    Returns:
        extrinsics_interp: Interpolated extrinsic matrices, shape=(M,4,4).
    """
    N = len(extrinsics)
    if N < 2:
        raise ValueError(
            "At least 2 extrinsic matrices are required for interpolation."
        )

    # Step 1: Construct cubic spline interpolators for translation vectors (separately for x/y/z axes)
    # Construct splines for 3 axes: trans_splines[0]->x, trans_splines[1]->y, trans_splines[2]->z
    trans = np.array(extrinsics[:, :3, 3])  # (N,3)
    trans_splines = [
        CubicSpline(
            x_original, trans[:, 0], bc_type="natural"
        ),  # natural: Natural spline, second derivative at endpoints is 0
        CubicSpline(x_original, trans[:, 1], bc_type="natural"),
        CubicSpline(x_original, trans[:, 2], bc_type="natural"),
    ]

    # Vectorized computation for all target translation points
    trans_interp = np.vstack(
        [
            trans_splines[0](x_target),
            trans_splines[1](x_target),
            trans_splines[2](x_target),
        ]
    ).T  # shape=(M,3)

    # Step 2: Interpolate rotation components for each target point
    R_original = Rotation.from_matrix(extrinsics[:, :3, :3])
    slerper = Slerp(x_original, R_original)

    # Handle extrapolation separately: if sampled frames extend beyond original frames, treat the tail separately
    x_target_inner = x_target[x_target < x_original[-1]]
    R_slerp = slerper(x_target_inner).as_matrix()

    # Extrapolation: Repeat the rotation matrix of the last interval
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
    Converts point cloud from world coordinate system to camera coordinate system.

    :param points_world: World coordinate point cloud, shape=(N, 3).
    :param T_cw: Camera extrinsics matrix (Camera -> World), shape=(4, 4).
    :return: Camera coordinate point cloud, shape=(N, 3).
    """
    points_world = points_world.astype(np.float32)

    # Step 1: Extract rotation and translation from extrinsics
    R_cw = T_cw[:3, :3]
    t_cw = T_cw[:3, 3]

    # Step 2: Compute World -> Camera rotation and translation
    R_wc = R_cw.T  # Inverse of rotation matrix = Transpose
    t_wc = -R_wc @ t_cw  # Equivalent to -np.dot(R_wc, t_cw)

    # Step 3: Transform each point
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
    if loc.device.type == "cuda":
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

    # Normalize (the last component is 1.0)
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
    if loc.device.type == "cuda":
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

    # Normalize (the last component is 1.0)
    point_transformed[:, 0] /= point_transformed[:, 2]
    point_transformed[:, 1] /= point_transformed[:, 2]
    point_transformed = point_transformed[:, :2]

    if hwc_input:
        point_transformed = point_transformed.reshape(h, w, 2)
    return point_transformed


def quaternion_to_matrix(quat, to_wxyz=False):
    if to_wxyz:
        quat = np.roll(quat, -1)
    r = Rotation.from_quat(quat)  # Order is (x, y, z, w)
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
    Checks for voxel occlusion during ray casting.

    :param voxels: 3D numpy array representing voxel grid (True indicates presence).
    :param origin: Origin coordinates of the ray (x, y, z).
    :param direction: Normalized direction vector of the ray.
    :param max_distance: Maximum checking distance.
    :return: Coordinates of the first hit voxel, or None if no voxel is hit.
    """
    position = np.array(origin, dtype=float)
    step_size = 0.1  # Step size, can be adjusted for precision or performance
    for _ in range(int(max_distance / step_size)):
        position += direction * step_size
        voxel_coords = np.floor(position).astype(int)

        # Check if coordinates are within grid bounds
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
    Converts point cloud from camera coordinate system back to world coordinate system.

    :param points_camera: Camera coordinate point cloud (N, 3).
    :param T_cw: Camera extrinsics (4, 4), i.e., self.camera_pose[i].
    :return: World coordinate point cloud (N, 3).
    """
    # T_cw is typically Camera-to-World transformation matrix (Pose)
    # P_world = R * P_cam + t
    R = T_cw[:3, :3]
    t = T_cw[:3, 3]

    # Matrix multiplication: (R @ P_cam.T).T + t
    points_world = (R @ points_camera.T).T + t
    return points_world


def load_depths_as_tensor(path="data/truck", interval=1, PIXEL_LIMIT=255000):
    """
    Loads depths from a directory or video, resizes them to a uniform size,
    then converts and stacks them into a single [N, 3, H, W] PyTorch tensor.
    """
    sources = []

    # --- 1. Load depth paths or video frames ---
    if osp.isdir(path):
        print(f"Loading depths from directory: {path}")
        filenames = sorted(
            [
                x
                for x in os.listdir(path)
                if x.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
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
        if k / m > W_target / H_target:
            k -= 1
        else:
            m -= 1
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


def load_images_as_tensor(path="data/truck", interval=1, PIXEL_LIMIT=255000,
                          condit_depth_path="videos/observation.images.depth", intrinsics_np=None, device='cpu'):
    """
    Loads images from a directory or video, resizes them to a uniform size,
    then converts and stacks them into a single [N, 3, H, W] PyTorch tensor.

    Optionally loads conditional depth maps and camera intrinsics (used by Pi3X
    multimodal conditioning). Returns an additional dict of conditions.

    Returns:
        images_tensor (torch.Tensor): RGB frames, shape [N, 3, H, W], values in [0, 1].
        all_frames (int): Total number of frames in the source (before interval subsampling).
        conditions (dict): {'poses', 'depths', 'intrinsics'} ready to splat into Pi3X.
    """
    sources = []
    all_frames = -1

    # --- 1. Load image paths or video frames ---
    if osp.isdir(path):
        print(f"Loading images from directory: {path}")
        filenames = sorted(
            [
                x
                for x in os.listdir(path)
                if x.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        all_frames = len(filenames)
        for i in range(0, len(filenames), interval):
            img_path = osp.join(path, filenames[i])
            try:
                sources.append(Image.open(img_path).convert("RGB"))
            except Exception as e:
                print(f"Could not load image {filenames[i]}: {e}")
    elif path.lower().endswith(".mp4"):
        print(f"Loading frames from video: {path}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {path}")
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sources.append(Image.fromarray(rgb_frame))
            frame_idx += 1
        cap.release()
        all_frames = frame_idx
        print(f"Total frames loaded from video: {all_frames}")
    else:
        raise ValueError(
            f"Unsupported path. Must be a directory or a .mp4 file: {path}"
        )

    if not sources:
        print("No images found or loaded.")
        return torch.empty(0), 0, {
            'poses': None,            # (1, N, 4, 4)
            'depths': None,           # (1, N, H, W)
            'intrinsics': None,       # (1, N, 3, 3)
            'K_rescaled': None,
        }

    print(f"Found {len(sources)} images/frames. Processing...")

    # --- 2. Determine a uniform target size for all images based on the first image ---
    # This is necessary to ensure all tensors have the same dimensions for stacking.
    first_img = sources[0]
    W_orig, H_orig = first_img.size
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target:
            k -= 1
        else:
            m -= 1
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
        return torch.empty(0), 0, {
            'poses': None,
            'depths': None,
            'intrinsics': None,
            'K_rescaled': None,
        }

    # --- 4. Stack the list of tensors into a single [N, C, H, W] batch tensor ---
    images_tensor = torch.stack(tensor_list, dim=0).to(device)

    N_out = images_tensor.shape[0]  # The actual number of successfully loaded RGB frames

    out_poses = None
    out_depths = None
    out_intrinsics = None
    out_K_rescaled = None  # numpy (3,3) at TARGET_W x TARGET_H resolution; metadata, not a model kwarg

    # Calculate resize ratios for geometry alignment.
    # (Must be calculated here because TARGET_W/H might have been adjusted by the while loop above.)
    scale_x = TARGET_W / W_orig
    scale_y = TARGET_H / H_orig

    # --- 5. Build intrinsic condition (rescaled to the model input resolution) ---
    if intrinsics_np is not None and isinstance(intrinsics_np, np.ndarray) and intrinsics_np.shape == (3, 3):
        # Work on a copy so the caller's array is not mutated in place.
        intr = intrinsics_np.astype(np.float32).copy()
        intr[0, 0] *= scale_x  # fx
        intr[0, 2] *= scale_x  # cx
        intr[1, 1] *= scale_y  # fy
        intr[1, 2] *= scale_y  # cy

        camera_tensor = torch.from_numpy(intr).float()  # float32
        out_intrinsics = camera_tensor[None].repeat(N_out, 1, 1)[None].to(device)  # (1, N, 3, 3)
        out_K_rescaled = intr  # keep numpy copy so downstream can save it as ground-truth K

    # --- 6. Build depth condition ---
    # 三种合法的"公制深度源":
    #   A. 16-bit PNG 目录 (InternData-N1: observation.images.depth/<idx>.png)
    #   B. gray16le mp4/mkv (Unitree gray16le 编码)
    # 其它(尤其是 8-bit yuv420p mp4)会被静默升位成假 16-bit、再除以 1000,
    # 输出 ~12-15 量级的"非米噪声",上一版 cv2 8-bit fallback 同样错(0-255÷255 也不是米)。
    # 因此本版**拒绝任何非 16-bit 的视频源**,把噪声从源头掐掉。
    out_depth_source = "none"
    if condit_depth_path is not None and os.path.exists(condit_depth_path):
        ext = condit_depth_path.lower()
        resized_depths_list = []

        # 6.0 自动切源: 当上层把 N1 的 observation.video.depth/<ep>.mp4 (8-bit) 当 condit
        # 传进来,但同 trajectory 下其实有 observation.images.depth/ (16-bit PNG 序列),
        # 优先切到 PNG 目录 —— 这是 N1 真公制深度的官方原件。
        resolved_path = condit_depth_path
        if os.path.isfile(condit_depth_path) and (ext.endswith(".mp4") or ext.endswith(".mkv")):
            png_dir = os.path.dirname(condit_depth_path).replace(
                "observation.video.depth", "observation.images.depth"
            )
            if (
                png_dir != os.path.dirname(condit_depth_path)
                and os.path.isdir(png_dir)
                and any(f.lower().endswith(".png") for f in os.listdir(png_dir))
            ):
                print(f"[depth] auto-switch from mp4 to 16-bit PNG dir: {png_dir}")
                resolved_path = png_dir

        if os.path.isdir(resolved_path):
            # 6A. PNG 序列 (InternData-N1 真公制 16-bit 编码)
            #
            # 编码约定 (实测推导, 见 README §11.3): N1 PNG 是 uint16, **0.1 mm/LSB**:
            #   - 0  = 缺失/未填充 (Pi3X 已经把 0 当无效处理)
            #   - 65535 = 饱和 / 远裁面 / 天空 (必须当作无效, 否则 Pi3X 把它当 6.55 m 的真深度)
            #   - 其它 = 真深度毫米十分位 (满量程 65535 -> 6.5535 m, 典型室内远裁面)
            #
            # 这与 Unitree gray16le mp4 (1 mm/LSB) 不是同一个约定; 不能混用除数。
            png_files = [f for f in os.listdir(resolved_path) if f.lower().endswith(".png")]
            # 数字 stem 排序: 兼容 0.png/100.png 这种非零填充的 N1 命名
            def _stem_key(name):
                stem = os.path.splitext(name)[0]
                return int(stem) if stem.isdigit() else stem
            png_files = sorted(png_files, key=_stem_key)
            median_check_sample = None
            for i in range(0, len(png_files), interval):
                d_map = cv2.imread(
                    os.path.join(resolved_path, png_files[i]), cv2.IMREAD_UNCHANGED
                )
                if d_map is None:
                    continue
                # 远裁面/天空 sentinel 65535 -> 0 (Pi3X 视为无效, 不参与 conditioning)
                d_map = d_map.copy()
                d_map[d_map == 65535] = 0
                d_resized = cv2.resize(d_map, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)
                d_resized = d_resized.astype(np.float32) / 10000.0  # 0.1 mm -> m
                d_resized[~np.logical_and(d_resized > 0, np.isfinite(d_resized))] = 0
                resized_depths_list.append(torch.from_numpy(d_resized))
                # 取第一帧做合理性检查
                if median_check_sample is None:
                    pos = d_resized[d_resized > 0]
                    if pos.size > 0:
                        median_check_sample = float(np.median(pos))
            # 合理性警示: 若 ÷10000 后的中位深度仍 > 8 m, 多半接到了非 N1 编码的 PNG 集
            if median_check_sample is not None and median_check_sample > 8.0:
                print(f"[Warning] PNG depth median {median_check_sample:.2f} m > 8 m. "
                      f"N1 约定 0.1 mm/LSB 通常给出 0.5-3 m 中位。该数据集可能用了不同约定, "
                      f"核对编码后再用 model_dc 结果。")
            out_depth_source = "png_dir"
        elif ext.endswith(".mkv") or ext.endswith(".mp4"):
            # 6B. mp4/mkv: 先验证是 16-bit pix_fmt 再走 gray16le; 8-bit 直接拒
            import av
            try:
                container = av.open(str(resolved_path))
            except Exception as e:
                print(f"[Warning] failed to open depth video {resolved_path}: {e}. "
                      "Skipping depth conditioning.")
                container = None
            if container is not None:
                video_stream = container.streams.video[0]
                pix_fmt = (video_stream.codec_context.pix_fmt or "").lower()
                # gray16le, yuv420p10le, gbrp16le 等都含 "16"; 8-bit yuv420p / yuvj420p / gray 不含
                is_16bit = ("16" in pix_fmt) or ("gray16" in pix_fmt)
                if not is_16bit:
                    print(
                        f"[Warning] depth video pix_fmt={pix_fmt!r} is not 16-bit. "
                        f"PyAV gray16le decode would left-shift 8-bit luma into fake 16-bit -> "
                        f"non-metric noise. Skipping depth conditioning. "
                        f"For metric depth, switch to a 16-bit PNG sequence under "
                        f"observation.images.depth/ alongside the mp4."
                    )
                    container.close()
                    out_depth_source = "none_skipped_8bit_mp4"
                else:
                    # 现有 gray16le 路径 (Unitree), 逐字节保留
                    for av_frame in islice(container.decode(video_stream), 0, None, interval):
                        d_map = av_frame.to_ndarray(format='gray16le')
                        d_resized = cv2.resize(d_map, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)
                        d_resized = d_resized.astype(np.float32) / 1000.0  # mm -> m
                        d_resized[~np.logical_and(d_resized > 0, np.isfinite(d_resized))] = 0
                        resized_depths_list.append(torch.from_numpy(d_resized))
                    container.close()
                    out_depth_source = "mp4_gray16le"
        else:
            print(f"[Warning] Unsupported depth path: {condit_depth_path}. "
                  "Expected a .mkv/.mp4 file or a directory of 16-bit PNGs. "
                  "Skipping depth conditioning.")

        n_depth = len(resized_depths_list)
        if n_depth > 0:
            if n_depth != N_out:
                if n_depth < N_out:
                    # Pad missing frames with zero maps -- Pi3X treats 0-depth as invalid (no conditioning).
                    zero_map = torch.zeros(TARGET_H, TARGET_W, dtype=torch.float32)
                    resized_depths_list.extend([zero_map] * (N_out - n_depth))
                    print(f"[Warning] Depth frames ({n_depth}) < RGB frames ({N_out}). "
                          f"Padding last {N_out - n_depth} frames with zeros (no depth conditioning).")
                else:
                    resized_depths_list = resized_depths_list[:N_out]
                    print(f"[Warning] Depth frames ({n_depth}) > RGB frames ({N_out}). "
                          f"Truncating to {N_out}.")
            out_depths = torch.stack(resized_depths_list, dim=0)[None].to(device)  # (1, N, H, W)
    elif condit_depth_path is not None:
        print(f"[Warning] Depth path does not exist: {condit_depth_path}. Skipping depth conditioning.")

    return images_tensor, all_frames, {
        'poses': out_poses,            # (1, N, 4, 4)
        'depths': out_depths,          # (1, N, H, W)
        'intrinsics': out_intrinsics,  # (1, N, 3, 3)
        # Metadata — NOT a Pi3X kwarg. Callers that do ``model(**conditions)`` must pop this first.
        # Set only when an external real intrinsic was provided; downstream uses it as the
        # authoritative K (at resized resolution) instead of the model's back-calculated estimate.
        'K_rescaled': out_K_rescaled,  # numpy (3, 3) or None
        # Metadata: 实际投喂给 Pi3X 的深度源, 用于 metrics.json 事后对账。
        # 取值: 'png_dir' / 'mp4_gray16le' / 'none' / 'none_skipped_8bit_mp4'
        'depth_source': out_depth_source,
    }
