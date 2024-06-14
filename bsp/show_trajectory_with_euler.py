import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_

# 读取位移数据
gt = pd.read_csv(r'D:\SZU\SLAM\results\lvio\direct\truth_pose_seq7_0.csv', delimiter=",")

gt_x = gt.iloc[:, 0].values
gt_z = gt.iloc[:, 2].values

# 计算累积位移
cumulative_x = np.cumsum(gt_x)
cumulative_z = np.cumsum(gt_z)

# 创建旋转矩阵
theta = np.radians(90)  # 转换为弧度
c = np.cos(theta)
s = np.sin(theta)
R = np.array(((c, -s), (s, c)))

# LVIO
for i in range(10,19):  # 从0到88
    lvio_result_file = f'D:\\SZU\\SLAM\\results\\lvio\\hard-100\\result_seq7_{i}.csv'
    lvio_result_file = f'D:\\SZU\\SLAM\\results\\vo-100\\result_seq7_{i}.csv'
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

    plt.plot(positions_lvio_result[:, 0], positions_lvio_result[:, 2], label=f'LVIO_{i}')

plt.plot(cumulative_x, cumulative_z, color='red', label='Ground Truth')
plt.xlabel("X")
plt.ylabel("Z")
plt.title("Trajectory")
plt.grid(True)
plt.legend()
plt.show()