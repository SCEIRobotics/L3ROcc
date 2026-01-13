import sys, os
import yaml
import numpy as np
import mayavi.mlab as mlab
import cv2
import glob
import re
from argparse import ArgumentParser

# 这个代码用于将世界坐标系下含有初始点云、OCC、相机轨迹的npy文件转换成视频


def numerical_sort(value):
    numbers = re.compile(r"(\d+)")
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


if __name__ == "__main__":
    mlab.options.offscreen = True

    parse = ArgumentParser()
    # 指向 npy_sequence_world (里面现在应该是 N x 7 的数据)
    parse.add_argument(
        "--input_dir",
        type=str,
        default="/Users/huangbinling/Documents/trae_projects/occgen/occgen/outputs/office_1/npy_sequence_world",
    )
    parse.add_argument(
        "--output_video",
        type=str,
        default="/Users/huangbinling/Documents/trae_projects/occgen/occgen/outputs/office_1/real_color_world.mp4",
    )  # 改个名防止覆盖
    parse.add_argument(
        "--config",
        type=str,
        default="/Users/huangbinling/Documents/trae_projects/occgen/occgen/occ/config.yaml",
    )

    args = parse.parse_args()
    input_dir = args.input_dir
    output_video = args.output_video
    config_path = args.config

    voxel_size = 0.05
    if os.path.exists(config_path):
        with open(config_path, "r") as stream:
            config = yaml.safe_load(stream)
        voxel_size = config.get("voxel_size", 0.05)

    files = sorted(glob.glob(os.path.join(input_dir, "*.npy")), key=numerical_sort)
    if not files:
        print(f"Error: No .npy files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(files)} frames. Generating Real-Color video...")

    width, height = 800, 800
    figure = mlab.figure(size=(width, height), bgcolor=(1, 1, 1))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, 30.0, (width, height))

    for i, file_path in enumerate(files):
        mlab.clf()

        try:
            data = np.load(file_path)  # Shape: (N, 7) -> [x, y, z, r, g, b, label]
        except:
            continue

        if data.shape[1] < 7:
            print(
                f"Error: Frame {i} has shape {data.shape}, expected (N, 7). Did you run the new Section D code?"
            )
            continue

        # 1. 拆分数据
        # Label 在第 6 列 (index 6)
        mask_bg = data[:, 6] == 0
        mask_traj = data[:, 6] == 1
        mask_occ = data[:, 6] == 2

        # 背景数据: XYZ 和 RGB
        bg_xyz = data[mask_bg, :3]
        bg_rgb = data[mask_bg, 3:6]  # R, G, B in [0, 1]

        pts_traj = data[mask_traj, :3]
        pts_occ = data[mask_occ, :3]

        # 2. 绘制背景 (支持真彩色!)
        if len(bg_xyz) > 0:
            N = len(bg_xyz)
            # 创建 0 到 N-1 的索引作为 scalar
            scalars = np.arange(N)

            # 绘制点，模式为 2dvertex
            pts = mlab.points3d(
                bg_xyz[:, 0],
                bg_xyz[:, 1],
                bg_xyz[:, 2],
                scalars,  # 传入索引
                mode="2dvertex",
                scale_factor=0.03,  # 仅占位
            )
            # 必须是 (N, 4) 的 uint8，包含 Alpha 通道
            lut = np.zeros((N, 4), dtype=np.uint8)
            lut[:, :3] = (bg_rgb * 255).astype(np.uint8)  # 填入 RGB
            lut[:, 3] = 255  # Alpha = 255 (不透明)

            # 将这个 LUT 强制赋值给 Mayavi 对象
            pts.module_manager.scalar_lut_manager.lut.number_of_colors = N
            pts.module_manager.scalar_lut_manager.lut.table = lut

            # 关闭自动缩放，防止颜色错位
            pts.glyph.scale_mode = "scale_by_vector"

        # 3. 绘制 OCC (Label 2) -> 深灰色方块
        if len(pts_occ) > 0:
            occ_plot = mlab.points3d(
                pts_occ[:, 0],
                pts_occ[:, 1],
                pts_occ[:, 2],
                mode="cube",
                color=(0.4, 0.4, 0.4),  # 深灰
                scale_factor=voxel_size - 0.005,
                opacity=1.0,
            )
            occ_plot.glyph.scale_mode = "data_scaling_off"

        # 4. 绘制轨迹 (Label 1) -> 蓝线 + 红头 (已修改大小)
        if len(pts_traj) > 0:
            # 历史轨迹
            if len(pts_traj) > 1:
                hist_traj = pts_traj[:-1]
                mlab.points3d(
                    hist_traj[:, 0],
                    hist_traj[:, 1],
                    hist_traj[:, 2],
                    mode="sphere",
                    color=(0.0, 0.0, 1.0),  # 纯蓝
                    # 【修改点】从 0.15 改为 0.05，缩小3倍
                    scale_factor=0.025,
                )

            # 当前车头
            curr_pos = pts_traj[-1]
            mlab.points3d(
                curr_pos[0],
                curr_pos[1],
                curr_pos[2],
                mode="sphere",
                color=(1.0, 0.0, 0.0),  # 纯红
                # 【修改点】从 0.3 改为 0.1，缩小3倍
                scale_factor=0.05,
            )

        # 5. 设置相机 通过visual.py获取想要的视角参数
        # -------------------------------------------------
        cam = figure.scene.camera
        cam.position = [-5.682662716850507, -2.3581944446045062, 1.1414235902534458]
        cam.focal_point = [0.3545984131007137, 0.24273155461184365, -0.1329467104813029]
        cam.view_angle = 30.0
        cam.view_up = [0.24965504712039718, -0.826617329042394, -0.5043571638969063]
        cam.clipping_range = [1.851463337952073, 12.81058369015543]
        cam.compute_view_plane_normal()

        figure.scene.render()

        img_array = mlab.screenshot(figure=figure, mode="rgb", antialiased=True)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        writer.write(img_bgr)

        if i % 10 == 0:
            print(f"Processed {i}/{len(files)}")

    writer.release()
    mlab.close(all=True)
    print(f"✅ Real-Color Fused Video saved to: {output_video}")
