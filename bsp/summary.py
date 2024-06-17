import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件，指定制表符为分隔符
data = pd.read_csv("D:/SLAM/path/checkpoints/KITTI_RAW_Synced/progress_log_summary.csv", sep='\t')

# 创建一个新的图形，包含3个子图
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# 绘制训练损失和验证损失
axs[0].plot(data['train_loss'], label='Train Loss')
axs[0].plot(data['validation_loss'], label='Validation Loss')
axs[0].set_title('Loss over epochs')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()

# 绘制训练姿态和验证姿态
axs[1].plot(data['train_pose'], label='Train Pose')
axs[1].plot(data['val_pose'], label='Validation Pose')
axs[1].set_title('Pose over epochs')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Pose')
axs[1].legend()

# 绘制训练欧拉角和验证欧拉角
axs[2].plot(data['train_euler'], label='Train Euler')
axs[2].plot(data['val_euler'], label='Validation Euler')
axs[2].set_title('Euler over epochs')
axs[2].set_xlabel('Epochs')
axs[2].set_ylabel('Euler')
axs[2].legend()

# 调整子图间距
plt.tight_layout()

# 显示图形
plt.show()