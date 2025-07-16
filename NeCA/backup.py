import odl
import numpy as np
import os
import nibabel as nib
import yaml

# 加载体积数据（.nii.gz）
def load_volume(path):
    img = nib.load(path)
    data = img.get_fdata()
    return data.astype(np.float32)

# 保存投影为 .npy 文件
def save_projection(projs, save_path):
    np.save(save_path, projs)

# 模拟投影（cone-beam）
def generate_projections(volume, angles, geometry_config):
    vol_shape = volume.shape

    # 定义体积空间
    space = odl.uniform_discr(
        min_pt=[-51.2, -51.2, -51.2],
        max_pt=[51.2, 51.2, 51.2],
        shape=vol_shape,
        dtype='float32'
    )

    # 角度分区
    angle_partition = odl.uniform_partition(0, 2 * np.pi, len(angles))

    # ✅ 二维探测器分区（修复了你遇到的问题）
    detector_partition = odl.uniform_partition(
        [-128, -128], [128, 128], [512, 512]
    )

    # Cone-beam 几何
    geometry = odl.tomo.ConeBeamGeometry(
        angle_partition,
        detector_partition,
        src_radius=geometry_config['src_radius'],
        det_radius=geometry_config['det_radius']
    )

    # 射线变换
    ray_trafo = odl.tomo.RayTransform(space, geometry)

    # 创建 ODL 元素
    f = space.element(volume)

    # 生成投影图像
    projections = ray_trafo(f)
    return np.expand_dims(projections.asarray(), axis=0)  # shape: (1, num_proj, H, W)

# 示例运行
if __name__ == '__main__':

    for i in range(91,100):

        # 输入体积路径
        vol_path = f'./data/CCTA_raw/1-200/{i}.label.nii.gz'
    
        # 输出模拟投影路径
        save_path = f'./data/CCTA_test/{i}_proj.npy'

        # 模拟投影几何参数
        geometry_config = {
            'src_radius': 750.0,
            'det_radius': 250.0
        }

        # 设置投影角度（3 个均匀角度）
        angles = np.linspace(0, 2 * np.pi, 3, endpoint=False)

        # 加载体积并生成投影
        volume = load_volume(vol_path)
        proj = generate_projections(volume, angles, geometry_config)
    
        # 保存投影
        save_projection(proj, save_path)