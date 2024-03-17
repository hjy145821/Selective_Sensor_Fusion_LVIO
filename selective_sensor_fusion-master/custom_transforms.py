from __future__ import division
import torch
import numpy as np

# Compose类：这个类用于将多个数据转换操作组合在一起。在初始化时，它接受一个转换操作的列表。
# 在调用时，它会依次对输入的图像进行这些转换操作，并返回转换后的图像。
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, images):
        for t in self.transforms:
            images = t(images)
        return images
    
# # PointCloudToTensor类，用于将点云数据转换为PyTorch张量的格式。
class PointCloudToTensor(object):
    def __call__(self, pointclouds):
        tensors = []
        for pc in pointclouds:
            # handle numpy array
            tensors.append(torch.from_numpy(pc.T).float())
        return tensors

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
