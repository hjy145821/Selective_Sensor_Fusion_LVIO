import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_

# 读取位移数据
gt = pd.read_csv(r'D:\SLAM\results\vio\hard\truth_pose_seq7_0.csv', delimiter=",")

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

# VIO
for i in range(20,31):  # 从0到88
    lvio_result_file = f'D:\\SLAM\\results\\vio\\hard\\result_seq7_{i}.csv'
    lvio = pd.read_csv(lvio_result_file, delimiter=",")

    lvio_x = lvio.iloc[:, 0].values
    lvio_z = lvio.iloc[:, 2].values

    # 计算累积位移
    lvio_x = np.cumsum(lvio_x)
    lvio_z = np.cumsum(lvio_z)

    # 旋转
    rotated_result_lvio = np.dot(R, np.array([lvio_x, lvio_z]))

    plt.plot(rotated_result_lvio[0], rotated_result_lvio[1], label=f'VIO_{i}')

plt.plot(cumulative_x, cumulative_z, color='red', label='Ground Truth')
plt.xlabel("X")
plt.ylabel("Z")
plt.title("Trajectory")
plt.grid(True)
plt.legend()
plt.show()