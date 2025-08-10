import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# 设置文件夹路径
# rendered_folder = '/home/lzy/workingspace/Scaffold-GS-cpp/results/replica_rgbd_0/office0/30000_shutdown/image'
# gt_folder = '/home/lzy/workingspace/Scaffold-GS-cpp/results/replica_rgbd_0/office0/30000_shutdown/image_gt'
rendered_folder = '/home/lzy/workingspace/Scaffold-GS-cpp/results/tum_rgbd_/rgbd_dataset_freiburg1_desk/30000_shutdown/image'
gt_folder = '/home/lzy/workingspace/Scaffold-GS-cpp/results/tum_rgbd_/rgbd_dataset_freiburg1_desk/30000_shutdown/image_gt'

output_folder_ssim = '/home/lzy/workingspace/Scaffold-GS-cpp/results/tum_rgbd_/rgbd_dataset_freiburg1_desk/30000_shutdown/ssim'
output_folder_heatmap = '/home/lzy/workingspace/Scaffold-GS-cpp/results/tum_rgbd_/rgbd_dataset_freiburg1_desk/30000_shutdown/heatmap'
output_folder_ssimHeatmap = '/home/lzy/workingspace/Scaffold-GS-cpp/results/tum_rgbd_/rgbd_dataset_freiburg1_desk/30000_shutdown/ssim_heatmap'

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder_ssim, exist_ok=True)
os.makedirs(output_folder_heatmap, exist_ok=True)
os.makedirs(output_folder_ssimHeatmap, exist_ok=True)

# 获取文件夹中的所有图像文件
rendered_files = sorted(os.listdir(rendered_folder))
gt_files = sorted(os.listdir(gt_folder))

# 逐一处理图像
for rendered_file, gt_file in zip(rendered_files, gt_files):
    # 读取渲染图像和GT图像
    rendered_path = os.path.join(rendered_folder, rendered_file)
    gt_path = os.path.join(gt_folder, gt_file)
    rendered_image = cv2.imread(rendered_path)
    gt_image = cv2.imread(gt_path)

    # 确保图像尺寸和通道数一致
    if rendered_image.shape != gt_image.shape:
        print(f"Skipping {rendered_file} and {gt_file} due to size mismatch.")
        continue

    # 计算 SSIM
    ssim_index, ssim_image = ssim(rendered_image, gt_image, full=True, multichannel=True)
    ssim_image = (ssim_image * 255).astype("uint8")
    ssim_output_path = os.path.join(output_folder_ssim, f"ssim_{rendered_file}")
    cv2.imwrite(ssim_output_path, ssim_image)

    # 计算 SSIM热图
    #     蓝色 通常表示 相似性较低 的区域（差异较大）。
    # 红色 或 黄色 通常表示 相似性较高 的区域（差异较小）
    ssimHeatmap_output_path = os.path.join(output_folder_ssimHeatmap, f"ssim_heatmap_{rendered_file}")
    ssim_image_colored = cv2.applyColorMap(ssim_image, cv2.COLORMAP_JET)
    cv2.imwrite(ssimHeatmap_output_path, ssim_image_colored)

    # 计算绝对差异图像
    difference = cv2.absdiff(rendered_image, gt_image)
    difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    normalized_diff = cv2.normalize(difference_gray, None, 0, 255, cv2.NORM_MINMAX)

    # 生成热图
    heatmap = cv2.applyColorMap(normalized_diff, cv2.COLORMAP_JET)
    heatmap_output_path = os.path.join(output_folder_heatmap, f"heatmap_{rendered_file}")
    cv2.imwrite(heatmap_output_path, heatmap)

    print(f"Processed {rendered_file} and {gt_file}")

print("Processing complete.")
