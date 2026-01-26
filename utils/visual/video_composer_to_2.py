import cv2
import numpy as np
import os
from tqdm import tqdm

# 这个代码用于第一版本拼接两个视频+左上角图片


def combine_videos(
    path_left_video,
    path_right_video,
    path_pip_image,
    output_path,
    pip_scale=0.3,  # 画中画缩放比例
    margin=20,  # 画中画距离左上角的边距
):
    # 1. 检查文件是否存在
    for p in [path_left_video, path_right_video, path_pip_image]:
        if not os.path.exists(p):
            print(f"❌ 错误：文件不存在 -> {p}")
            return

    # 2. 打开视频流
    cap_left = cv2.VideoCapture(path_left_video)
    cap_right = cv2.VideoCapture(path_right_video)

    # 获取视频基本信息 左侧为准
    fps = cap_left.get(cv2.CAP_PROP_FPS)
    w_left = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_left = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))

    # 3. 读取并处理画中画图片 (点云图)
    pip_img = cv2.imread(path_pip_image)
    if pip_img is None:
        print("❌ 错误：无法读取点云图片")
        return

    # 计算画中画的目标大小 (保持原图比例)
    pip_h_orig, pip_w_orig = pip_img.shape[:2]
    target_pip_w = int(w_left * pip_scale)  # 宽度占左边视频的 30%
    target_pip_h = int(target_pip_w * (pip_h_orig / pip_w_orig))

    # 缩放点云图
    pip_resized = cv2.resize(pip_img, (target_pip_w, target_pip_h))

    # 给画中画加个白色边框，看起来更明显 (可选)
    pip_resized = cv2.copyMakeBorder(
        pip_resized, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    pip_h, pip_w = pip_resized.shape[:2]

    # 4. 准备输出视频流
    # 我们需要先读取一帧右侧视频来确定最终画布的宽度
    ret, frame_right_sample = cap_right.read()
    if not ret:
        return
    # 将右侧视频的高度强行缩放到与左侧一致，保持对齐
    h_right_orig, w_right_orig = frame_right_sample.shape[:2]
    scale_factor = h_left / h_right_orig
    w_right_new = int(
        w_right_orig * scale_factor * 1.78
    )  # 看看这里需不需要按照这个缩放

    # 重置右侧视频指针到开头
    cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 最终画布尺寸
    canvas_w = w_left + w_right_new
    canvas_h = h_left

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (canvas_w, canvas_h))

    print(f"🎬 开始处理...")
    print(f"左视频尺寸: {w_left}x{h_left}")
    print(f"右视频缩放后: {w_right_new}x{h_left}")
    print(f"画布总尺寸: {canvas_w}x{canvas_h}")

    # 5. 循环处理每一帧
    # 使用 tqdm 显示进度条
    pbar = tqdm(total=total_frames, unit="frame")

    while True:
        ret1, frame_left = cap_left.read()
        ret2, frame_right = cap_right.read()

        # 如果任意一个视频读完了，就结束
        if not ret1 or not ret2:
            break

        # A. 处理右侧视频：缩放高度以匹配左侧
        frame_right = cv2.resize(frame_right, (w_right_new, h_left))

        # B. 拼接 (水平拼接)
        # axis=1 表示横向，axis=0 表示纵向
        canvas = np.concatenate((frame_left, frame_right), axis=1)

        # C. 插入画中画 (覆盖左上角)
        # 坐标范围: [y_start : y_end, x_start : x_end]
        y1, y2 = margin, margin + pip_h
        x1, x2 = margin, margin + pip_w

        # 确保不越界
        if y2 < h_left and x2 < w_left:
            # 直接像素覆盖
            canvas[y1:y2, x1:x2] = pip_resized

        # D. 写入文件
        writer.write(canvas)
        pbar.update(1)

    # 6. 释放资源
    cap_left.release()
    cap_right.release()
    writer.release()
    pbar.close()
    print(f"\n✅ 合成完成！视频已保存至: {output_path}")


# ==========================================
# 在这里修改你的路径
# ==========================================
if __name__ == "__main__":
    # 原视频 (左边)
    input_video_path = (
        "/Users/huangbinling/Documents/trae_projects/occgen/occgen/inputs/e1.mp4"
    )

    # OCC 生成的视频 (右边)
    occ_video_path = "/Users/huangbinling/Documents/trae_projects/occgen/occgen/outputs/e1_02/occ_video_e04.mp4"

    # 初始点云截图 (左上角画中画)
    pcd_image_path = (
        "/Users/huangbinling/Documents/trae_projects/occgen/occgen/snapshot01.png"
    )

    # 输出路径
    output_video_path = "/Users/huangbinling/Documents/trae_projects/occgen/occgen/outputs/e1_02/final_demo.mp4"

    combine_videos(
        input_video_path,
        occ_video_path,
        pcd_image_path,
        output_video_path,
        pip_scale=0.3,  # 画中画占左侧宽度的 35%
        margin=0,  # 边距 像素
    )
