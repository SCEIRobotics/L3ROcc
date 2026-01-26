import cv2
import numpy as np
import os
from tqdm import tqdm

# 这个代码用于拼接RGB视频+world视频+occ视频---》最终用这个！


def combine_three_videos_crop_middle(
    path_input_video,  # 左：原视频 (基准)
    path_world_video,  # 中：世界坐标系融合视频 (将被裁剪)
    path_occ_video,  # 右：OCC 视频
    output_path,
    crop_ratio=0.2,  # 中间视频裁剪比例
):
    # 1. 检查文件是否存在
    inputs = [path_input_video, path_world_video, path_occ_video]
    names = ["原视频", "World视频", "OCC视频"]
    for p, n in zip(inputs, names):
        if not os.path.exists(p):
            print(f"❌ 错误：{n} 不存在 -> {p}")
            return

    # 2. 打开三个视频流
    cap_1 = cv2.VideoCapture(path_input_video)  # 左
    cap_2 = cv2.VideoCapture(path_world_video)  # 中
    cap_3 = cv2.VideoCapture(path_occ_video)  # 右

    # 获取基准信息 (以左侧原视频为准)
    fps = cap_1.get(cv2.CAP_PROP_FPS)
    h1 = int(cap_1.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 基准高度
    w1 = int(cap_1.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap_1.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- 处理中间视频 (World) - 需要裁剪 ---
    ret2, frame2_sample = cap_2.read()
    if not ret2:
        return
    h2_raw, w2_raw = frame2_sample.shape[:2]

    # 计算裁剪边距
    # crop_ratio = 0.2 表示上下左右各去掉 20%
    margin_h = int(h2_raw * crop_ratio)
    margin_w = int(w2_raw * crop_ratio)

    # 计算裁剪后的原始尺寸 (用于计算缩放比例)
    h2_cropped_orig = h2_raw - 2 * margin_h
    w2_cropped_orig = w2_raw - 2 * margin_w

    # 计算缩放比例 (让裁剪后的高度对齐到 h1)
    scale2 = h1 / h2_cropped_orig
    w2_new = int(w2_cropped_orig * scale2)  # 最终在画布上的宽度

    cap_2.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置指针

    # --- 处理右侧视频 (OCC) - 不需要裁剪 ---
    ret3, frame3_sample = cap_3.read()
    if not ret3:
        return
    h3_orig, w3_orig = frame3_sample.shape[:2]
    scale3 = h1 / h3_orig
    w3_new = int(w3_orig * scale3)
    cap_3.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置指针

    # 计算总宽度
    canvas_w = w1 + w2_new + w3_new
    canvas_h = h1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (canvas_w, canvas_h))

    print(f"🎬 开始处理三屏合并 (中间视频裁剪 {crop_ratio*100:.0f}%)...")
    print(f"视窗1 (Input) : {w1}x{h1}")
    print(
        f"视窗2 (World) : 原始 {w2_raw}x{h2_raw} -> 裁剪后 {w2_cropped_orig}x{h2_cropped_orig} -> 最终缩放 {w2_new}x{h1}"
    )
    print(f"视窗3 (OCC)   : {w3_new}x{h1}")
    print(f"最终画布      : {canvas_w}x{canvas_h}")

    # 5. 循环处理
    pbar = tqdm(total=total_frames, unit="frame")

    while True:
        ret1, frame1 = cap_1.read()
        ret2, frame2 = cap_2.read()
        ret3, frame3 = cap_3.read()

        # 只要有一个视频播完，就结束
        if not ret1 or not ret2 or not ret3:
            break

        # A. 处理中间视频：先裁剪，再缩放
        # 裁剪语法: image[y_start:y_end, x_start:x_end]
        # 使用负索引 -margin_h 等价于 h2_raw - margin_h
        frame2_cropped = frame2[margin_h:-margin_h, margin_w:-margin_w]
        frame2 = cv2.resize(frame2_cropped, (w2_new, h1))

        # B. 处理右侧视频：直接缩放
        frame3 = cv2.resize(frame3, (w3_new, h1))

        # C. 三屏拼接 [ Left | Middle | Right ]
        canvas = np.concatenate((frame1, frame2, frame3), axis=1)

        # (已移除 PiP 插入代码)

        writer.write(canvas)
        pbar.update(1)

    cap_1.release()
    cap_2.release()
    cap_3.release()
    writer.release()
    pbar.close()
    print(f"\n✅ 三屏(带裁剪)视频已保存至: {output_path}")


if __name__ == "__main__":
    base_dir = "/Users/huangbinling/Documents/trae_projects/occgen/occgen"

    # 1. 最左边: 原视频
    path_input = os.path.join(base_dir, "inputs/office.mp4")

    # 2. 中间: World 坐标系融合视频 (将被裁剪放大)
    # 使用你刚才生成的那个真彩色的视频
    path_world = os.path.join(base_dir, "outputs/office_1/real_color_world.mp4")

    # 3. 最右边: 纯 OCC 视频
    path_occ = os.path.join(base_dir, "outputs/office_1/occ_only.mp4")

    # 4. 输出路径
    path_output = os.path.join(base_dir, "outputs/office_1/final_3screen_crop_demo.mp4")

    # 运行
    combine_three_videos_crop_middle(
        path_input,
        path_world,
        path_occ,
        path_output,
        crop_ratio=0.15,  # 上下左右各裁掉 15%
    )
