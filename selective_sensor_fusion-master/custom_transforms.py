from __future__ import division
import torch
import numpy as np
import math
# Compose类：这个类用于将多个数据转换操作组合在一起。在初始化时，它接受一个转换操作的列表。
# 在调用时，它会依次对输入的图像进行这些转换操作，并返回转换后的图像。
class Compose_imgs(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, images):
        for t in self.transforms:
            images = t(images)
        return images
    
class Compose_points(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, points):
        for t in self.transforms:
            points = t(points)
        return points   
    

# Normalize类：这个类用于对图像进行标准化处理。在初始化时，它接受均值（mean）和标准差（std）作为参数。
# 在调用时，它会对输入的图像进行标准化处理，即将每个像素值减去均值并除以标准差。
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images
    
def project_to_cylindrical(point_cloud, height = 256, width = 512):
    x = point_cloud[:, :, 0]  # 提取点云的 x 坐标
    y = point_cloud[:, :, 1]  # 提取点云的 y 坐标
    z = point_cloud[:, :, 2]  # 提取点云的 z 坐标
    alpha = torch.atan2(y, x)  # 计算 alpha
    beta = torch.asin(z / (torch.sqrt(x**2 + y**2 + z**2) + 1e-6))  # 计算 beta

    # 将 alpha 和 beta 映射到柱面图像的坐标范围内
    alpha_mapped = (alpha / (2 * math.pi) + 0.5) * width  # 映射到 [0, width] 范围内
    beta_mapped = (beta + 0.5) * height  # 映射到 [0, height] 范围内

    # 创建一个空的柱面图像张量
    cylindrical_image = torch.zeros((point_cloud.shape[0], height, width))

    # 将点云投影到柱面图像中
    for i in range(point_cloud.shape[0]):
        for j in range(point_cloud.shape[1]):
            alpha_idx = int(alpha_mapped[i, j])
            beta_idx = int(beta_mapped[i, j])
            if alpha_idx >= 0 and alpha_idx < width and beta_idx >= 0 and beta_idx < height:
                if cylindrical_image[i, beta_idx, alpha_idx] == 0 or cylindrical_image[i, beta_idx, alpha_idx] > torch.sqrt(x[i, j]**2 + y[i, j]**2 + z[i, j]**2):
                    cylindrical_image[i, beta_idx, alpha_idx] = torch.sqrt(x[i, j]**2 + y[i, j]**2 + z[i, j]**2)
    return cylindrical_image

# ArrayToTensor类：这个类用于将numpy数组转换为PyTorch张量。
# 在调用时，它会将输入的图像从HWC（高度、宽度、通道）格式转换为CHW（通道、高度、宽度）格式，并将其转换为PyTorch的FloatTensor类型。
class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of
    shape (C x H x W) with a intrinsics tensor."""
    def __call__(self, images):
        tensors = []
        for im in images:
            # put it from HWC to CHW format
            im = np.transpose(im, (2, 0, 1))
            # handle numpy array
            tensors.append(torch.from_numpy(im).float())
        return tensors
