import torch.utils.data as data
import numpy as np
# from scipy.misc import imread
from PIL import Image
from pathlib import Path
import random
import cv2
import numpy as np
import struct

class Args:
    def __init__(self):
        self.data = 'E:\\SLAM\\Dataset\\KITTI_RAW_Synced'
        self.sequence_length = 3
        self.rotation_mode = 'euler'
        self.workers = 4
        self.epochs = 1
        self.epoch_size = 0
        self.batch_size = 8
        self.lr = 1e-4
        self.momentum = 0.9
        self.beta = 0.999
        self.weight_decay = 0
        self.print_freq = 20
        self.seed = 0
        self.log_summary = 'progress_log_summary.csv'

class KITTI_Loader(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """
    def __init__(self, root, seed=None, train=2, sequence_length=3, transform_imgs=None, transform_points=None, data_degradation=0, data_random=True):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        print("train parameter value:", train)
        if train == 0:
            scene_list_path = self.root / 'train.txt'
        if train == 1:
            scene_list_path = self.root / 'val.txt'
        if train == 2:
            scene_list_path = self.root / 'test.txt'
            
        print(scene_list_path)
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform_imgs = transform_imgs
        self.transform_points = transform_points
        # degradation mode
        # 0: normal data 1: occlusion 2: blur 3: image missing 4: imu noise and bias 5: imu missing
        # 6: spatial misalignment 7: temporal misalignment 8: vision degradation 9: all degradation
        self.data_degradation = data_degradation
        self.sequence_length = sequence_length
        self.random = data_random

        if (train == 0) or (train == 1):
            self.crawl_folders(sequence_length)
        else:
            self.crawl_test_folders()

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length + 1))

        for scene in self.scenes:

            # load data from left camera
            imgs_l = sorted(scene.glob('*.png'))  # 获取匹配模式的文件列表
            imus_l = sorted(scene.glob('*.txt'))
            poses_l = np.genfromtxt(scene / 'poses.txt').astype(np.float64).reshape(-1, 3, 4)
            point_clouds_1 = sorted(scene.glob('*.bin'))

            for i in range(demi_length, len(imgs_l) - demi_length):
                sample = {'imgs': [], 'poses': [], 'imus': [], 'point_clouds': [], 'data_degradation': []}
                flag = True  # 初始化标志变量为True
                for j in shifts:
                    sample['imgs'].append(imgs_l[i + j])
                    sample['poses'].append(poses_l[i + j, :, :])
                    sample['imus'].append(np.genfromtxt(imus_l[i + j]).astype(np.float32).reshape(-1, 6))
                    # 将point_cloud的定义移动到内部循环中
                    with open(point_clouds_1[i + j], 'rb') as file:
                        raw_data = file.read()
                        num_points = len(raw_data) // 16
                        point_cloud = struct.unpack('f' * (num_points * 4), raw_data)
                        point_cloud = np.array(point_cloud).reshape(-1, 4)
                        sample['point_clouds'].append(point_cloud)

                sample['data_degradation'] = np.zeros(sequence_length).tolist()

                previous_img = -1
                lost_n = 0
                for n_img, img in enumerate(sample['imgs']):
                    current_img = int(str(img)[-14:-4])
                    if previous_img != -1:
                        if current_img - previous_img != 1:
                            flag = False  # 发现丢失的图像，将标志变量设置为False
                            lost_n = current_img - previous_img
                    previous_img = current_img

                if flag:
                    generate_degrade(sample, sequence_length, self.data_degradation)
                    degrade_imu_data(sample, sequence_length)
                    sequence_set.append(sample)
                else:
                    print(lost_n)

            if self.random:
                random.shuffle(sequence_set)

            self.samples = sequence_set

    def crawl_test_folders(self):
        sequence_set = []

        for scene in self.scenes:
            # Load data from left camera
            imgs_l = sorted(scene.glob('*.png'))
            imus_l = sorted(scene.glob('*.txt'))
            print("test")
            print(scene)
            poses_l = np.genfromtxt(scene / 'poses.txt').astype(np.float64).reshape(-1, 3, 4)

            sample = {'imgs': [], 'poses': [], 'imus': [], 'point_clouds': [], 'data_degradation': []}

            # Load point cloud data
            point_clouds_1 = sorted(scene.glob('*.bin'))
            for pc_file in point_clouds_1:
                with open(pc_file, 'rb') as file:
                    raw_data = file.read()
                    num_points = len(raw_data) // 16  # Assuming each point is represented by 4 floats (16 bytes)
                    point_cloud = struct.unpack('f' * (num_points * 4), raw_data)
                    point_cloud = np.array(point_cloud).reshape(-1, 4)
                    sample['point_clouds'].append(point_cloud)
            # load imu、imgs、poses
            for i in range(len(imgs_l)):
                sample['imgs'].append(imgs_l[i])
                sample['poses'].append(poses_l[i, :, :])
                sample['imus'].append(np.genfromtxt(imus_l[i]).astype(np.float32).reshape(-1, 6))

            sample['data_degradation'] = np.zeros(len(imgs_l)).tolist()

            generate_degrade(sample, len(imgs_l), self.data_degradation)
            degrade_imu_data(sample, len(imgs_l))
            sequence_set.append(sample)

        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]

        data_degradation = np.copy(sample['data_degradation'])

        imgs = [load_as_float(img, data_degradation[n_img]) for n_img, img in enumerate(sample['imgs'])]
        poses = [pose for pose in sample['poses']]
        imus = [imu for imu in sample['imus']]
        point_clouds = [point_cloud for point_cloud in sample['point_clouds']]

        if self.transform_imgs is not None:
            imgs = self.transform_imgs(imgs)
        if self.transform_points is not None:
            point_clouds = self.transform_points(point_clouds)

        return imgs, imus, poses, point_clouds

    def __len__(self):
        return len(self.samples)

def generate_degrade(sample, sequence_length, mode):

    # degradation mode
    # 0: normal data 1: occlusion 2: blur 3: image missing 4: imu noise and bias 5: imu missing
    # 6: spatial misalignment 7: temporal misalignment 8: vision degradation 9: all degradation

    for i in range(sequence_length):

        rand_label = np.random.rand(1)

        if mode == 0:
            return

        if (mode > 0) and (mode < 8):

            if rand_label < 0.3:
                sample['data_degradation'][i] = mode

        if mode == 8:

            if rand_label < 0.10:
                sample['data_degradation'][i] = 1

            if (rand_label > 0.10) and (rand_label < 0.20):
                sample['data_degradation'][i] = 2

            if (rand_label > 0.20) and (rand_label < 0.30):
                sample['data_degradation'][i] = 3

        if mode == 9:

            if rand_label < 0.05:
                sample['data_degradation'][i] = 1

            if (rand_label > 0.05) and (rand_label < 0.10):
                sample['data_degradation'][i] = 2

            if (rand_label > 0.10) and (rand_label < 0.15):
                sample['data_degradation'][i] = 3

            if (rand_label > 0.15) and (rand_label < 0.20):
                sample['data_degradation'][i] = 4

            if (rand_label > 0.20) and (rand_label < 0.25):
                sample['data_degradation'][i] = 5

            if (rand_label > 0.25) and (rand_label < 0.30):
                sample['data_degradation'][i] = 6

            if (rand_label > 0.30) and (rand_label < 0.35):
                sample['data_degradation'][i] = 7

    return

def degrade_imu_data(sample, sequence_length):

    for i in range(sequence_length):

        if sample['data_degradation'][i] == 4:

            imu_seq = sample['imus'][i]

            for imu_n, imu in enumerate(imu_seq):

                imu_new = np.copy(imu)
                for k in range(3):
                    imu_new[k] = imu[k] + np.random.rand(1) * 0.1 + 0.1
                    imu_new[k + 3] = imu[k + 3] + np.random.rand(1) * 0.001 + 0.001

                imu_seq[imu_n] = imu_new

            sample['imus'][i] = imu_seq

        if sample['data_degradation'][i] == 5:
            sample['imus'][i] = np.zeros((10, 6)).astype(np.float32)

        if sample['data_degradation'][i] == 6:

            theta = int(np.random.rand(1) * 5) + 5
            rot_theta = np.array([[np.cos(theta), -np.sin(theta), 0],
                                 [np.sin(theta), np.cos(theta), 0],
                                 [0, 0, 1]])

            imu_seq = sample['imus'][i]

            for imu_n, imu in enumerate(imu_seq):
                imu_new = np.copy(imu)

                imu_new[:3] = imu[:3] @ rot_theta
                imu_new[3:] = imu[3:] @ rot_theta

                imu_seq[imu_n] = imu_new

            sample['imus'][i] = imu_seq

        if sample['data_degradation'][i] == 7:

            imu_seq = sample['imus'][i]

            for imu_n, imu in enumerate(imu_seq):

                if np.random.rand(1) < 0.5:
                    imu_seq[imu_n] = np.zeros(6).astype(np.float32)

            sample['imus'][i] = imu_seq

    return

def load_as_float(path, label):
    img = np.array(Image.open(path).convert('RGB'), dtype=np.float32)
    # img = Image.open(path).convert('RGB').astype(np.float32)
    # img = imread(path).astype(np.float32)

    if label == 1:
        height_start = int(np.random.rand(1)*128)

        width_start = int(np.random.rand(1)*384)

        for ind_h in range(height_start, height_start+128):
            for ind_w in range(width_start, width_start+128):
                for ind_c in range(0, 3):
                    img[ind_h, ind_w, ind_c] = 0

    if label == 3:
        img = np.zeros((256, 512, 3)).astype(np.float32)

    if label == 2:
        kernel = np.ones((15, 15), np.float32) / 225
        img = cv2.filter2D(img, -1, kernel)

        row, col, ch = img.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(img)
        # Salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in img.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in img.shape]
        out[coords] = 0

        img = out

    return img
