import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取poses数据
poses = np.loadtxt('D:\\SLAM\\Dataset\\KITTI_RAW_Synced\\00\\poses.txt').reshape(-1, 3, 4)

# Compute translation only (simple subtraction)
def compute_trans_pose_simple(ref_pose, tgt_pose):
    trans_pose = np.copy(tgt_pose)
    trans_pose[:, -1] -= ref_pose[:, -1]
    return trans_pose

# Compute translation with rotation consideration
def compute_trans_pose_with_rotation(ref_pose, tgt_pose):
    tmp_pose = np.copy(tgt_pose)
    tmp_pose[:, -1] -= ref_pose[:, -1]
    trans_pose = np.linalg.inv(ref_pose[:, :3]) @ tmp_pose
    return trans_pose

def compute_trajectory(poses, compute_trans_pose_func):
    positions = []
    current_position = np.array([0.0, 0.0, 0.0])
    current_rotation = np.eye(3)

    for i in range(len(poses) - 1):
        ref_pose = poses[i]
        tgt_pose = poses[i + 1]
        
        trans_pose = compute_trans_pose_func(ref_pose, tgt_pose)
        translation = trans_pose[:, -1]
        rotation_matrix = trans_pose[:, :3]
        
        current_position += current_rotation @ translation
        current_rotation = current_rotation @ rotation_matrix
        
        positions.append(current_position.copy())

    return np.array(positions)

# Compute trajectories using both methods
trajectory_simple = compute_trajectory(poses, compute_trans_pose_simple)
trajectory_with_rotation = compute_trajectory(poses, compute_trans_pose_with_rotation)

# Plotting the trajectories
plt.figure()
plt.plot(trajectory_simple[:, 0], trajectory_simple[:, 2], label='Simple Translation', linestyle='--')
plt.plot(trajectory_with_rotation[:, 0], trajectory_with_rotation[:, 2], label='With Rotation Consideration', linestyle='-')
plt.xlabel('X')
plt.ylabel('Z')
plt.legend()
plt.title('Trajectory Comparison')
plt.show()
