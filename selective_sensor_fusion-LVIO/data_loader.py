import torch.utils.data as data
import numpy as np
from pathlib import Path
import cv2
import torch

class Args:
    def __init__(self):
        self.data = 'D:\\SLAM\\Dataset\\KITTI_RAW_Synced'
        self.fusion_mode = 6
        self.sequence_length = 3
        self.rotation_mode = 'euler'
        self.workers = 4
        self.epochs = 1
        self.epoch_size = 0
        self.batch_size = 128
        self.lr = 1e-4
        self.momentum = 0.9
        self.beta = 0.999
        self.weight_decay = 0
        self.print_freq = 50
        self.log_summary = 'progress_log_summary.csv'

class KITTI_Loader(data.Dataset):

    def __init__(self, root, train=0, fusion_mode=0, sequence_length=3, transform=None):
        self.root = Path(root)
        if train == 0:
            scene_list_path = self.root/'train_test.txt'
        if train == 1:
            scene_list_path = self.root/'val.txt'
        if train == 2:
            scene_list_path = self.root/'test.txt'

        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.fusion_mode = fusion_mode
        self.sequence_length = sequence_length

        if (train == 0) or (train == 1):
            self.crawl_folders(sequence_length)
        else:
            self.crawl_test_folders()

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length + 1))
        for scene in self.scenes:
            print(scene)

            imgs_l = sorted(scene.glob('*.jpg'))
            imus_l = sorted(scene.glob('*.txt'))
            poses_l = np.genfromtxt(scene / 'poses.txt').astype(np.float64).reshape(-1, 3, 4)
            if self.fusion_mode in [4, 5, 6]:
                clouds_l = sorted(scene.glob('*.bin'))

            if len(imgs_l) < sequence_length:
                continue

            for i in range(demi_length, len(imgs_l) - demi_length):
                sample = {'imgs': [], 'poses': [], 'imus': [],'clouds': []}
                for j in shifts:
                    sample['imgs'].append(imgs_l[i + j])
                    sample['poses'].append(poses_l[i + j, :, :])
                    sample['imus'].append(np.genfromtxt(imus_l[i + j]).astype(np.float32).reshape(-1, 6))

                    if self.fusion_mode in [4, 5, 6]:
                        sample['clouds'].append(clouds_l[i + j])

                sequence_set.append(sample)

        self.samples = sequence_set

    def crawl_test_folders(self):

        sequence_set = []
        for scene in self.scenes:
            print(scene)
            # load data from left camera
            imgs_l = sorted(scene.glob('*.jpg'))
            imus_l = sorted(scene.glob('*.txt'))
            poses_l = np.genfromtxt(scene / 'poses.txt').astype(np.float64).reshape(-1, 3, 4)
            if self.fusion_mode in [4, 5, 6]:
                clouds_l = sorted(scene.glob('*.bin'))
            sample = {'imgs': [], 'poses': [], 'imus': [], 'clouds': []}

            for i in range(len(imgs_l)):
                sample['imgs'].append(imgs_l[i])
                sample['poses'].append(poses_l[i, :, :])
                sample['imus'].append(np.genfromtxt(imus_l[i]).astype(np.float32).reshape(-1, 6))

                if self.fusion_mode in [4, 5, 6]:
                    sample['clouds'].append(clouds_l[i])

            sequence_set.append(sample)

        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]

        imgs = [cv2.imread(str(img)).astype(np.float32) for img in sample['imgs']]
        poses = [pose for pose in sample['poses']]
        imus = [imu for imu in sample['imus']]

        if self.fusion_mode in [4, 5, 6]:
            clouds = [load_point_cloud(str(cloud)) for cloud in sample['clouds']]  # Load point clouds from file paths
            pc_imgs = [cylindrical_projection(pc) for pc in clouds]  # Apply cylindrical projection to each point cloud
        else:
            clouds = []
            pc_imgs = []

        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, imus, poses, pc_imgs

    def __len__(self):
        return len(self.samples)
    
def load_point_cloud(bin_file):
    point_cloud = np.fromfile(bin_file, dtype=np.float32)
    point_cloud = point_cloud.reshape(-1, 4)
    return point_cloud

def cylindrical_projection(point_cloud, H=256, W=512, C=3):
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]
    intensity = point_cloud[:, 3]

    r = np.sqrt(x**2 + y**2 + z**2)
    r[r == 0] = 1e-3
    alpha = np.arctan2(y, x)
    beta = np.arcsin(z / r)

    delta_alpha = 2 * np.pi / W
    delta_beta = np.pi / H

    alpha_idx = (alpha / delta_alpha).astype(np.int64)
    beta_idx = (beta / delta_beta).astype(np.int64)

    # Initialize cylindrical image and depth map
    cylindrical_img = np.zeros((C, H, W))
    depth_map = np.full((H, W), 1e10)

    # Fill the cylindrical image
    for i in range(point_cloud.shape[0]):
        if r[i] < depth_map[beta_idx[i], alpha_idx[i]]:  # If the current point is closer
            depth_map[beta_idx[i], alpha_idx[i]] = r[i]  # Update depth map
            cylindrical_img[0, beta_idx[i], alpha_idx[i]] = r[i]  # Update r
            cylindrical_img[1, beta_idx[i], alpha_idx[i]] = intensity[i]  # Update intensity
            cylindrical_img[2, beta_idx[i], alpha_idx[i]] = 0

    # Convert to PyTorch tensor
    cylindrical_img = torch.from_numpy(cylindrical_img).float()

    return cylindrical_img