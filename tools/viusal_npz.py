import sys, os
import yaml
import numpy as np
import mayavi.mlab as mlab 
import argparse

def get_points_from_data(raw_data, frame_index=0):
    points = None
    data_source = None
    
    # 尝试获取核心数据
    if 'data' in raw_data:
        data_source = raw_data['data']
    elif isinstance(raw_data, np.ndarray):
        data_source = raw_data
    else:
        # 兼容旧格式 keys
        if 'mask_camera' in raw_data and 'semantics' in raw_data:
            print("Detected old format (mask + semantics).")
            grid = raw_data["semantics"] * raw_data["mask_camera"]
            return np.argwhere(grid != 0)
        else:
            raise ValueError(f"Unknown file format. Keys: {list(raw_data.keys())}")

    print(f"Data Shape: {data_source.shape}")

    #  根据维度判断处理逻辑
    # 4D 序列 (N, H, W, D) -> occ_sequence.npz
    if data_source.ndim == 4:
        total_frames = data_source.shape[0]
        if frame_index >= total_frames:
            print(f"Frame index {frame_index} out of bounds (Max {total_frames-1}). Using last frame.")
            frame_index = -1
        
        print(f"Mode: 4D Sequence | Extracting Frame {frame_index}/{total_frames}")
        grid_frame = data_source[frame_index] # 取出一帧 (H, W, D)
        
        points = np.argwhere(grid_frame > 0)

    # 3D Grid (H, W, D) -> 单帧 grid
    elif data_source.ndim == 3:
        print(" Mode: 3D Grid")
        points = np.argwhere(data_source > 0)

    # 点云列表 (N, 3) -> all_occ.npz
    elif data_source.ndim == 2 and data_source.shape[1] == 3:
        print(" Mode: Point List (Global OCC)")
        points = data_source
        
        # 针对 Global OCC 的特殊处理：如果是浮点数(米)，转成整数索引
        if np.issubdtype(points.dtype, np.floating):
            print("   -> Detected Meters. Quantizing to Indices...")
            # 这里的 voxel_size 只是为了量化，取 0.05 即可
            points = points - points.min(axis=0) # 归一化到原点
            points = np.round(points / 0.05).astype(int)
    
    else:
        raise ValueError(f"Unsupported data shape: {data_source.shape}")

    return points

if __name__=="__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--path', type=str, default="/Users/huangbinling/Documents/trae_projects/occgen/occgen/inputs/occ_sequence.npz")  
    parser.add_argument('--frame', type=int, default=30, help="View specific frame for sequence data")
    parser.add_argument('--bg_black', action='store_true', default=True, help="Use black background")
    
    args = parser.parse_args()
    
    print(f"Loading: {args.path}")
    try:
        raw_data = np.load(args.path)
        points = get_points_from_data(raw_data, args.frame)
    except Exception as e:
        print(f"Error loading file: {e}")
        exit()

    if points is None or len(points) == 0:
        print("Error: Result is empty (0 voxels occupied). Check frame index or data.")
        exit()

    print(f"   Plotting {len(points)} voxels...")
    print(f"   X range: {points[:,0].min()} - {points[:,0].max()}")
    print(f"   Y range: {points[:,1].min()} - {points[:,1].max()}")
    print(f"   Z range: {points[:,2].min()} - {points[:,2].max()}")

    # ================= 绘图逻辑 =================
    bg_color = (0, 0, 0) if args.bg_black else (1, 1, 1)
    fg_color = (1, 1, 1) if args.bg_black else (0, 0, 0)
    
    mlab.figure(size=(1000, 800), bgcolor=bg_color, fgcolor=fg_color)
    
    mlab.points3d(
        points[:, 0], 
        points[:, 1], 
        points[:, 2],
        mode="cube",
        color=(1, 1, 1), 
        scale_factor=1.0, 
        scale_mode='none',
        opacity=1.0
    )
    
    mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
    mlab.orientation_axes()
    print("Window Opened. Interact with the scene to verify structure.")
    mlab.show()