# 修改 visual.py，添加离线渲染配置
import os
# 禁用 GUI 显示，使用 offscreen 渲染
os.environ['ETS_TOOLKIT'] = 'null'
os.environ['MLAB_OFFSCREEN'] = '1'
os.environ['TRAITSUI_TOOLKIT'] = 'qt5'  
import mayavi.mlab as mlab

# 示例：渲染 3D 图形并保存为图片
x, y, z = [1,2,3], [4,5,6], [7,8,9]
mlab.plot3d(x, y, z)
mlab.savefig('output.png')  # 保存图片而非显示窗口
mlab.close()