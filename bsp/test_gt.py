import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# 文件路径
pose_file = 'D:\\SLAM\\results\\lvio\\direct-50\\truth_pose_seq7_0.csv'
euler_file = 'D:\\SLAM\\results\\lvio\\direct-50\\truth_euler_seq7_0.csv'
lvio_result_file = 'D:\\SLAM\\results\\lvio\\direct-50\\result_seq7_24.csv'
vio_result_file = 'D:\\SLAM\\results\\vio\\hard\\result_seq7_27.csv'

# 读取数据
truth_pose = np.loadtxt(pose_file, delimiter=',')
truth_euler = np.loadtxt(euler_file, delimiter=',')
lvio_result_data = np.loadtxt(lvio_result_file, delimiter=',')
vio_result_data = np.loadtxt(vio_result_file, delimiter=',')

# 初始化轨迹和旋转矩阵
positions_truth = []
current_position_truth = np.array([0.0, 0.0, 0.0])
current_rotation_truth = np.eye(3)

# 计算真实轨迹
for i in range(len(truth_pose)):
    translation = truth_pose[i]
    euler_angles = truth_euler[i]
    
    rotation_matrix = R.from_euler('xyz', euler_angles).as_matrix()
    
    current_position_truth += current_rotation_truth @ translation
    current_rotation_truth = current_rotation_truth @ rotation_matrix
    
    positions_truth.append(current_position_truth.copy())

positions_truth = np.array(positions_truth)

# 初始化LVIO结果轨迹
positions_lvio_result = []
current_position_lvio_result = np.array([0.0, 0.0, 0.0])
current_rotation_lvio_result = np.eye(3)

# 计算LVIO结果轨迹
for i in range(len(lvio_result_data)):
    translation = lvio_result_data[i, :3]
    euler_angles = lvio_result_data[i, 3:]
    
    rotation_matrix = R.from_euler('xyz', euler_angles).as_matrix()
    
    current_position_lvio_result += current_rotation_lvio_result @ translation
    current_rotation_lvio_result = current_rotation_lvio_result @ rotation_matrix
    
    positions_lvio_result.append(current_position_lvio_result.copy())

positions_lvio_result = np.array(positions_lvio_result)

# 初始化VIO结果轨迹
positions_vio_result = []
current_position_vio_result = np.array([0.0, 0.0, 0.0])
current_rotation_vio_result = np.eye(3)

# 计算VIO结果轨迹
for i in range(len(vio_result_data)):
    translation = vio_result_data[i, :3]
    euler_angles = vio_result_data[i, 3:]
    
    rotation_matrix = R.from_euler('xyz', euler_angles).as_matrix()
    
    current_position_vio_result += current_rotation_vio_result @ translation
    current_rotation_vio_result = current_rotation_vio_result @ rotation_matrix
    
    positions_vio_result.append(current_position_vio_result.copy())

positions_vio_result = np.array(positions_vio_result)

# 绘制轨迹
plt.figure()
plt.plot(positions_truth[:, 0], positions_truth[:, 2], label='Truth Trajectory', linestyle='-')
plt.plot(positions_lvio_result[:, 0], positions_lvio_result[:, 2], label='LVIO Result Trajectory', linestyle='--')
plt.plot(positions_vio_result[:, 0], positions_vio_result[:, 2], label='VIO Result Trajectory', linestyle='-.')
plt.xlabel('X')
plt.ylabel('Z')
plt.legend()
plt.title('Comparison of LVIO and VIO Truth and Result Trajectories')
plt.show()
