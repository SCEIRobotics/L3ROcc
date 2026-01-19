import torch
import os
import random
import yaml
import time
import glob
import numpy as np
import open3d as o3d
from collections import deque
from scipy import sparse
import pandas as pd

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
    point_transform_2d_batch,
    estimate_intrinsics
)

class DataGenerator:
    def __init__(
        self,
        config_path="./occ/config.yaml",
        save_dir="./outputs",
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
        self.camera_intric = np.array(
            [[168.0498, 0.0, 240.0], [0.0, 192.79999, 135.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )  # intrinsic matrix
        self.camera_intric_rs = None 
        self.camera_pose = None  # extrinsic matrix
        self.camera_trace = None
        self.norm_cam_ray = None  # 归一化的相机射线方向, 默认处于相机坐标系

        with open(config_path, "r") as stream:
            self.config = yaml.safe_load(stream)

        self.fps = self.config["fps"]
        self.voxel_size = self.config["voxel_size"]
        self.pc_range = self.config["pc_range"]
        self.occ_size = self.config["occ_size"]
        self.ray_cast_step_size = self.config["ray_cast_step_size"]
        self.interval = self.config["interval"]
        self.voxel_size_scale = self.config["voxel_size_scale"]
        self.history_len = self.config["history_len"]
        self.history_step = self.config["history_step"]
        self.occ_history_buffer = deque(maxlen=self.history_len) # 创建一个定长队列，自动挤出旧数据
        self.save_path = self.save_dir

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
        ref_cam_index = 0
        tgt_cam_index = 0  # vis_index # len(camera_pose) // 3 * 2
        camera_pose_cuda = torch.from_numpy(camera_pose).to(self.device)
 
        # version 2
        norm_cam_ray_cam_coords = res["local_points"][0][ref_cam_index] / res[
            "local_points"
        ][0][ref_cam_index].norm(
            dim=2, keepdim=True
        )  # shape: (378, 672, 3) 
        # compute the camera intrinsic matrix from solve DLT
        self.camera_intric_rs = estimate_intrinsics(res["local_points"][0][ref_cam_index])


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
            #write_ply(pcd_with_cam, pcd_color_with_cam, ply_save_path)

        return pcd, camera_pose, norm_cam_ray_cam_coords.reshape(-1, 3)

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

    def get_temporal_occ(self, new_occ_world, current_pose_matrix, save_to_history=False):
        """
        滑动窗口累积 OCC，并转换到当前相机坐标系
        
        Args:
            new_occ_world: 当前帧新检测到的 OCC (N, 3), 世界坐标系
            current_pose_matrix: 当前相机的位姿 (4, 4), 世界坐标系
            save_to_history: 是否将当前帧数据加入滑动队列
            
        Returns:
            merged_occ_cam: 累积并转换后的 OCC (M, 3), 当前相机坐标系
            merged_occ_world: 累积后的 OCC (M, 3), 世界坐标系 (用于存 NPY)
        """
        #将当前帧的数据加入滑动队列
        if save_to_history and len(new_occ_world)>0:
            self.occ_history_buffer.append(new_occ_world)

        # 首先取出所有历史点
        candidates = list(self.occ_history_buffer)

        # 如果当前帧没有被存入历史（save_to_history=False），
        # 我们依然需要在画面中看到它，所以要手动把它加到临时的融合列表中。
        if not save_to_history and len(new_occ_world) > 0:
            candidates.append(new_occ_world)

        if len(candidates) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

        #合并缓冲区内的所有点 此时所有点都在世界坐标系下 直接合并不需要坐标变换
        merged_occ_world = np.concatenate(candidates, axis=0)

        #体素下采样 
        # 因为多帧累积会有大量重叠点，必须降采样去重，否则点数会爆炸
        if len(merged_occ_world) > 0:
            pcd_tmp = o3d.geometry.PointCloud()
            pcd_tmp.points = o3d.utility.Vector3dVector(merged_occ_world)
            pcd_tmp = pcd_tmp.voxel_down_sample(voxel_size=self.voxel_size) 
            merged_occ_world = np.asarray(pcd_tmp.points, dtype=np.float32)
        
        if len(merged_occ_world) == 0:
             return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
        
        # 转换到当前相机坐标系
        merged_occ_cam = self.convert_pointcloud_world_to_camera(merged_occ_world, current_pose_matrix)
        
        return merged_occ_cam, merged_occ_world
    
    def _get_io_paths(self, input_path):
        """生成并创建输出所需的所有路径字典"""
        
        # 1. 构建目录
        data_chunk_dir = os.path.join(self.save_path, "data", "chunk-000")
        video_chunk_dir = os.path.join(self.save_path, "videos", "chunk-000")
        occ_view_dir = os.path.join(video_chunk_dir, "observation.occ.view")
        occ_mask_dir = os.path.join(video_chunk_dir, "observation.occ.mask")

        for d in [data_chunk_dir, video_chunk_dir, occ_view_dir, occ_mask_dir]:
            if not os.path.exists(d): os.makedirs(d)

        # 2. 定义文件路径
        paths = {
            'ply': os.path.join(data_chunk_dir, "origin_pcd.ply"),
            'global_occ': os.path.join(data_chunk_dir, "all_occ.npz"),
            'parquet': os.path.join(data_chunk_dir, "episode_000000.parquet"),
            'occ_seq': os.path.join(occ_view_dir, "occ_sequence.npz"),
            'mask_seq': os.path.join(occ_mask_dir, "mask_sequence.npz")
        }
        return paths

    def _save_global_data(self, paths):
        """保存全局点云和OCC"""
        import shutil
        for p in [paths['ply'], paths['global_occ']]:
            if os.path.isdir(p):
                shutil.rmtree(p)
        
        write_ply(self.pcd, self.pcd_color, paths['ply'])
        np.savez_compressed(paths['global_occ'], data=self.occ_pcd.astype(np.float32))

    def _compute_sequence_data(self):
        """遍历帧，计算 OCC Grid 和收集 Poses"""
        total_frames = len(self.camera_pose)
        grid_dims = self.config['occ_size'] 
        
        all_frames_voxels = [] 
        all_frames_cam_mask_voxels = [] 
        all_camera_poses = [] 

        for i in range(total_frames):
            current_pose = self.camera_pose[i] 
            
            # 收集外参: 格式 [Array(row1), Array(row2)...]
            pose_rows = [row.astype(np.float32) for row in current_pose]
            all_camera_poses.append(pose_rows)

            # 计算单帧可见性
            occ_indices, cam_visible_mask = self.check_visual_occ(self.occ_pcd, current_pose)
            
            # 格式转换
            if isinstance(occ_indices, torch.Tensor):
                occ_voxel_indices = occ_indices.detach().cpu().numpy()
            else:
                occ_voxel_indices = occ_indices

            if isinstance(cam_visible_mask, torch.Tensor):
                cam_mask_voxel_indices = cam_visible_mask.detach().cpu().numpy()
            else:
                cam_mask_voxel_indices = cam_visible_mask

            # --- 1. OCC Grid ---
            valid_mask_occ = (
                (occ_voxel_indices[:, 0] >= 0) & (occ_voxel_indices[:, 0] < grid_dims[0]) &
                (occ_voxel_indices[:, 1] >= 0) & (occ_voxel_indices[:, 1] < grid_dims[1]) &
                (occ_voxel_indices[:, 2] >= 0) & (occ_voxel_indices[:, 2] < grid_dims[2])
            )
            valid_voxels_occ = occ_voxel_indices[valid_mask_occ].astype(np.int64) 
            
            frame_grid_occ = np.zeros(grid_dims, dtype=np.uint8)
            frame_grid_occ[valid_voxels_occ[:, 0], valid_voxels_occ[:, 1], valid_voxels_occ[:, 2]] = 1
            all_frames_voxels.append(frame_grid_occ)

            # --- 2. Mask Grid ---
            valid_mask_cam = (
                (cam_mask_voxel_indices[:, 0] >= 0) & (cam_mask_voxel_indices[:, 0] < grid_dims[0]) &
                (cam_mask_voxel_indices[:, 1] >= 0) & (cam_mask_voxel_indices[:, 1] < grid_dims[1]) &
                (cam_mask_voxel_indices[:, 2] >= 0) & (cam_mask_voxel_indices[:, 2] < grid_dims[2])
            )
            valid_voxels_cam = cam_mask_voxel_indices[valid_mask_cam].astype(np.int64)
            
            frame_grid_mask = np.zeros(grid_dims, dtype=np.uint8)
            frame_grid_mask[valid_voxels_cam[:, 0], valid_voxels_cam[:, 1], valid_voxels_cam[:, 2]] = 1
            all_frames_cam_mask_voxels.append(frame_grid_mask)

        return np.stack(all_frames_voxels, axis=0), np.stack(all_frames_cam_mask_voxels, axis=0), all_camera_poses

    def _update_parquet_metadata(self, parquet_save_path, all_camera_poses, input_path):
        """更新 Parquet 文件"""
        print(f"Updating Parquet metadata at {parquet_save_path}...")
        try:
            # 回溯 4 层目录找到 trajectory 根目录 (基于 input_path 为 mp4 文件)
            traj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(input_path))))
            origin_parquet_path = os.path.join(traj_root, 'data', 'chunk-000', 'episode_000000.parquet')
            
            if os.path.exists(origin_parquet_path):
                df_orig = pd.read_parquet(origin_parquet_path, engine='pyarrow')
            else:
                print(f"Error: Original parquet NOT found at {origin_parquet_path}")
                return 
        except Exception as e:
            print(f"Error finding/reading parquet: {e}")
            return

        # 准备数据
        current_len = len(df_orig)
        new_col_data = all_camera_poses[:current_len]
        
        if len(new_col_data) < current_len:
            print(f"Warning: Padding {current_len - len(new_col_data)} frames.")
            new_col_data.extend([None] * (current_len - len(new_col_data)))

        # 添加新列
        df_orig['observation.camera_extrinsic_occ'] = new_col_data

        # 保存
        try:
            df_orig.to_parquet(parquet_save_path, engine='pyarrow')
            print(f"Successfully updated parquet: {parquet_save_path}")
        except Exception as e:
            print(f"Error saving parquet: {e}")

    def _update_json_metadata(self, input_path):
        """更新 info.json 文件"""
        import json
        print(f"Updating info.json metadata...")
        try:
            traj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(input_path))))
            info_json_path = os.path.join(traj_root, "meta", "info.json")
            
            if os.path.exists(info_json_path):
                with open(info_json_path, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                
                new_occ_feature = {
                    "dtype": "float32",
                    "shape": [4, 4],
                    "names": [
                        "extrinsic_0_0", "extrinsic_0_1", "extrinsic_0_2", "extrinsic_0_3",
                        "extrinsic_1_0", "extrinsic_1_1", "extrinsic_1_2", "extrinsic_1_3",
                        "extrinsic_2_0", "extrinsic_2_1", "extrinsic_2_2", "extrinsic_2_3",
                        "extrinsic_3_0", "extrinsic_3_1", "extrinsic_3_2", "extrinsic_3_3"
                    ]
                }
                
                if "features" in meta_data:
                    meta_data["features"]["observation.camera_extrinsic_occ"] = new_occ_feature
                    with open(info_json_path, 'w', encoding='utf-8') as f:
                        json.dump(meta_data, f, indent=4)
                    print(f"Successfully updated info.json")
                else:
                    print(f"Warning: 'features' key not found in {info_json_path}")
            else:
                print(f"Warning: info.json not found at {info_json_path}")
                
        except Exception as e:
            print(f"Error updating info.json: {e}")


    def occ_gen_compressed_pipeline(self, input_path, pcd_save=False):
        """整个 OCC 生成、保存和元数据更新的流程"""
        
        #  注入默认内参 
        if self.camera_intric is None:
            self.camera_intric = np.array(
                [[168.0498, 0.0, 240.0], [0.0, 192.79999, 135.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )

        # 三维重建 & 全局 OCC 生成
        pcd, self.camera_pose, self.norm_cam_ray = self.pcd_reconstuction(input_path, pcd_save)
        self.occ_pcd = self.pcd_to_occ(pcd)

        if pcd_save:
            print("Start processing sequence frames...")
            
            # --- 1: 生成并创建所有需要的路径 ---
            paths = self._get_io_paths(input_path)
            
            # --- 2: 保存全局 PCD 和 OCC ---
            self._save_global_data(paths)

            # --- 3: 核心计算 (生成每一帧的 4D OCC Grid 和 Pose List) ---
            arr_4d_occ, arr_4d_mask, all_camera_poses = self._compute_sequence_data()

            # --- 4: 保存 4D 压缩序列 ---
            print("Saving 4D Sequence Arrays...")
            np.savez_compressed(paths['occ_seq'], data=arr_4d_occ)
            np.savez_compressed(paths['mask_seq'], data=arr_4d_mask)
            print(f"Saved 4D Sequence shape: {arr_4d_occ.shape}")

            # --- 5: 更新元数据 (Parquet 和 JSON) ---
            # 这里的 input_path 是原始视频路径，用于回溯根目录
            self._update_parquet_metadata(paths['parquet'], all_camera_poses, input_path)
            self._update_json_metadata(input_path)

# ================= 主函数 =================

if __name__ == "__main__":

    config_path = "./occ/config.yaml"
    save_dir = "./outputs"
    model_dir = "./ckpt"

    input_path = "/mnt/data/huangbinling/project/occgen/inputs"
    video_name = "office.mp4"
    generator = DataGenerator(config_path, save_dir, model_dir)
    generator.occ_gen_compressed_pipeline(os.path.join(input_path, video_name), pcd_save=True)


