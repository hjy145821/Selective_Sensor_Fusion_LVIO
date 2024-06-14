import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_

gt_pose_file = 'D:\\SLAM\\results\\lvio\\hard-100\\truth_pose_seq10_10.csv'
gt_pose_data = np.loadtxt(gt_pose_file, delimiter=',')
gt_euler_file = 'D:\\SLAM\\results\\lvio\\hard-100\\truth_euler_seq10_10.csv'
gt_euler_data = np.loadtxt(gt_euler_file, delimiter=',')


# gt
positions_gt = []
current_position_gt = np.array([0.0, 0.0, 0.0])
current_rotation_gt = np.eye(3)
for i in range(len(gt_pose_data)):
    translation = gt_pose_data[i, :]
    euler_angles = gt_euler_data[i, :]
    
    rotation_matrix = R_.from_euler('xyz', euler_angles).as_matrix()
    
    current_position_gt += current_rotation_gt @ translation
    current_rotation_gt = current_rotation_gt @ rotation_matrix
    
    positions_gt.append(current_position_gt.copy())

positions_gt = np.array(positions_gt)

# # LVIO
# for i in range(10,19):  # 从0到88
#     lvio_result_file = f'D:\\SLAM\\results\\lvio\\hard-100\\result_seq10_{i}.csv'
#     lvio_result_data = np.loadtxt(lvio_result_file, delimiter=',')

#     positions_lvio_result = []
#     current_position_lvio_result = np.array([0.0, 0.0, 0.0])
#     current_rotation_lvio_result = np.eye(3)
#     for j in range(len(lvio_result_data)):
#         translation = lvio_result_data[j, :3]
#         euler_angles = lvio_result_data[j, 3:]

#         rotation_matrix = R_.from_euler('xyz', euler_angles).as_matrix()

#         current_position_lvio_result += current_rotation_lvio_result @ translation
#         current_rotation_lvio_result = current_rotation_lvio_result @ rotation_matrix

#         positions_lvio_result.append(current_position_lvio_result.copy())

#     positions_lvio_result = np.array(positions_lvio_result)

#     plt.plot(positions_lvio_result[:, 0], positions_lvio_result[:, 2], label=f'LVIO_{i}')

# LVIO
lvio_result_file = f'D:\\SLAM\\results\\lvio\\hard-100\\result_seq10_16.csv'
lvio_result_data = np.loadtxt(lvio_result_file, delimiter=',')

positions_lvio_result = []
current_position_lvio_result = np.array([0.0, 0.0, 0.0])
current_rotation_lvio_result = np.eye(3)
for j in range(len(lvio_result_data)):
    translation = lvio_result_data[j, :3]
    euler_angles = lvio_result_data[j, 3:]

    rotation_matrix = R_.from_euler('xyz', euler_angles).as_matrix()

    current_position_lvio_result += current_rotation_lvio_result @ translation
    current_rotation_lvio_result = current_rotation_lvio_result @ rotation_matrix

    positions_lvio_result.append(current_position_lvio_result.copy())

positions_lvio_result = np.array(positions_lvio_result)
plt.plot(positions_lvio_result[:, 0], positions_lvio_result[:, 2], label=f'LVIO_16')

plt.plot(positions_gt[:, 0], positions_gt[:, 2], color='red', label='Ground Truth')
plt.xlabel("X")
plt.ylabel("Z")
plt.title("Trajectory_seq10")
plt.grid(True)
plt.legend()
plt.show()