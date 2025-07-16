import numpy as np
import os
import matplotlib.pyplot as plt


for k in range(91,100):
    # === 参数定义 ===
    npy_path = f'/home/yilin/syn_tree/NeCA/data/CCTA_test/{k}_proj.npy'   # 输入 .npy 文件路径
    save_dir = '/home/yilin/syn_tree/NeCA/data/CCTA_test_visual_proj'                  # 输出图像保存文件夹
    prefix = 'proj'                             # 图像文件名前缀

    # === 加载投影数据 ===
    projections = np.load(npy_path)  # Shape: (1, 30, 512, 512)
    projections = projections[0]     # Shape: (30, 512, 512)

    # === 创建输出文件夹 ===
    os.makedirs(save_dir, exist_ok=True)

    # === 保存每个角度的投影图 ===
    for i, proj_img in enumerate(projections):
        plt.figure(figsize=(6, 6))
        plt.imshow(proj_img, cmap='gray')
        plt.axis('off')
        filename = os.path.join(save_dir, f'{prefix}_{k}_{i:02d}.png')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

    print(f"✅ successfully saved to：{save_dir}")