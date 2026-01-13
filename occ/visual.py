import sys, os
sys.path.append('.')
sys.path.append('..')
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # add file dir
import yaml
import numpy as np

import mayavi.mlab as mlab 
from globals import get_global
#这个代码用于可视化.npy文件！关于可视化的都要在本地安装python=3.8，再跑!我用这个代码是用于获取第一帧最想要的参数视角

LABEL_COLORS = get_global('LABEL_COLORS') * 255
alpha = np.ones((LABEL_COLORS.shape[0], 1)) * 255
LABEL_COLORS = np.concatenate((LABEL_COLORS, alpha), axis=1)
LABEL_COLORS = LABEL_COLORS.astype(np.uint8)
FREE_LABEL = len(LABEL_COLORS)


def voxel2points(pred_occ, mask_camera = None, free_label = 0):
 
    x = np.linspace(0, pred_occ.shape[0] - 1, pred_occ.shape[0])
    y = np.linspace(0, pred_occ.shape[1] - 1, pred_occ.shape[1])
    z = np.linspace(0, pred_occ.shape[2] - 1, pred_occ.shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    vv = np.stack([X, Y, Z, pred_occ], axis=-1)
    valid_mask = pred_occ != free_label
    if mask_camera is not None:
        valid_mask = np.logical_and(valid_mask, mask_camera)
    fov_voxels = vv[valid_mask].astype(np.float32)
 
    return fov_voxels

if __name__=="__main__":
    offscreen = False
    mlab.options.offscreen = offscreen
    from argparse import ArgumentParser
    parse = ArgumentParser() 
    parse.add_argument('--visual_path', type=str, default="/mnt/data/huangbinling/project/occgen/outputs/occ_pcd_visual_cam0.npy")  
    parse.add_argument('--visual_save_dir', type=str, default="/mnt/data/huangbinling/project/occgen/outputs/visual")
    args = parse.parse_args()
    visual_path = args.visual_path
    visual_save_dir = args.visual_save_dir
    config_path = './occ/config.yaml'
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
        
    voxel_size = config['voxel_size']
    pc_range = config['pc_range']
    occ_size = config['occ_size'] # occ_range
 
    if visual_path.split('.')[-1]=="npz":
        mask_camera = np.load(visual_path)["mask_camera"] # mask_camera or semantics
        semantics = np.load(visual_path)["semantics"] 
        fov_voxels = voxel2points(semantics)
    else:
        fov_voxels = np.load(visual_path).astype(np.float32)
    if len(fov_voxels.shape) == 3:
        fov_voxels = voxel2points(fov_voxels)
    if fov_voxels.shape[1] == 4:
        fov_voxels = fov_voxels[fov_voxels[..., 3] >= 0]
        
     
    # fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
    # fov_voxels[:, 0] += pc_range[0]
    # fov_voxels[:, 1] += pc_range[1]
    # fov_voxels[:, 2] += pc_range[2]


    figure = mlab.figure(size=(600, 600), bgcolor=(1, 1, 1))
    
    fov_voxels[fov_voxels[:, 3]>=len(LABEL_COLORS),3] = len(LABEL_COLORS)-1
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05*voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=len(LABEL_COLORS) - 1,
    )


    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = LABEL_COLORS

    if offscreen:
        mlab.savefig(os.path.join(visual_save_dir, 'occ_visual.png'))
    else:
        mlab.show()
