import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_

# 读取位移数据
gt = pd.read_csv(r'D:\SZU\SLAM\results\lvio\direct\truth_pose_seq7_0.csv', delimiter=",")
vo = pd.read_csv(r'D:\SZU\SLAM\results\vo-100\result_seq7_57.csv', delimiter=",")
vio = pd.read_csv(r'D:\SZU\SLAM\results\vio\hard\result_seq7_27.csv', delimiter=",")
lvio = pd.read_csv(r'D:\SZU\SLAM\results\lvio\hard-100\result_seq7_30.csv', delimiter=",")

lvio_result_file = 'D:\\SZU\\SLAM\\results\\lvio\\hard-100\\result_seq7_26.csv'
lvio_result_data = np.loadtxt(lvio_result_file, delimiter=',')
vo_result_file = 'D:\\SZU\\SLAM\\results\\vo-100\\result_seq7_26.csv'
vo_result_data = np.loadtxt(vo_result_file, delimiter=',')

gt_x = gt.iloc[:, 0].values
gt_z = gt.iloc[:, 2].values
vo_x = vo.iloc[:, 0].values
vo_z = vo.iloc[:, 2].values
vio_x = vio.iloc[:, 0].values
vio_z = vio.iloc[:, 2].values
lvio_x = lvio.iloc[:, 0].values
lvio_z = lvio.iloc[:, 2].values

# 计算累积位移
cumulative_x = np.cumsum(gt_x)
cumulative_z = np.cumsum(gt_z)
vo_x = np.cumsum(vo_x)
vo_z = np.cumsum(vo_z)
vio_x = np.cumsum(vio_x)
vio_z = np.cumsum(vio_z)
lvio_x = np.cumsum(lvio_x)
lvio_z = np.cumsum(lvio_z)

# 创建旋转矩阵
theta = np.radians(90)  # 转换为弧度
c = np.cos(theta)
s = np.sin(theta)
R = np.array(((c, -s), (s, c)))
rotated_result_vo = np.dot(R, np.array([vo_x, vo_z]))
rotated_result_vio = np.dot(R, np.array([vio_x, vio_z]))
rotated_result_lvio = np.dot(R, np.array([lvio_x, lvio_z]))

# # LVIO
# positions_lvio_result = []
# current_position_lvio_result = np.array([0.0, 0.0, 0.0])
# current_rotation_lvio_result = np.eye(3)
# for i in range(len(lvio_result_data)):
#     translation = lvio_result_data[i, :3]
#     euler_angles = lvio_result_data[i, 3:]
    
#     rotation_matrix = R_.from_euler('xyz', euler_angles).as_matrix()
    
#     current_position_lvio_result += current_rotation_lvio_result @ translation
#     current_rotation_lvio_result = current_rotation_lvio_result @ rotation_matrix
    
#     positions_lvio_result.append(current_position_lvio_result.copy())

# positions_lvio_result = np.array(positions_lvio_result)

plt.plot(cumulative_x, cumulative_z, color='red', label='Ground Truth')
plt.plot(rotated_result_vo[0], rotated_result_vo[1], color='blue', label='VO')
plt.plot(rotated_result_vio[0], rotated_result_vio[1], color='green', label='VIO')
# plt.plot(positions_vio_result[:, 0], positions_vio_result[:, 2], label='VIO')
plt.plot(rotated_result_lvio[0], rotated_result_lvio[1], color='m', label='LVIO')
# plt.plot(positions_lvio_result[:, 0], positions_lvio_result[:, 2], label='LVIO')
plt.xlabel("X")
plt.ylabel("Z")
plt.title("Trajectory")
plt.grid(True)
plt.legend()
plt.show()