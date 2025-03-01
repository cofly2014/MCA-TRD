import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.ticker as ticker # 导入ticker模块

# 定义原始的a1、a2参数值（作为标签使用）
a1_labels = np.array([0.005, 0.05, 0.5, 5])
a2_labels = np.array([0.005, 0.05, 0.5, 5])

# 生成等间距的坐标数据（这里假设有4个等间距的坐标点，你可以根据实际需求修改点数）
num_points = 4
a1_coords = np.linspace(0, 1, num_points)
a2_coords = np.linspace(0, 1, num_points)

# 生成数据（示例数据，你可以替换为真实数据）
#hmdb
acc = np.array([
    [51.14, 53.20, 52.40, 53.22],
    [50.18, 54.20, 53.30, 51.96],
    [50.32, 51.70, 50.36, 50.00],
    [22.30, 22.72, 28.54, 23.30]
])
dataset_name = "HMDB51"

#Diving48

acc = np.array([
    [42.46, 42.90, 42.20, 42.08],
    [41.14, 43.8, 42.08, 42.04],
    [41.08, 41.66, 41.24, 40.88],
    [30.20, 24.24, 26.00, 28.44]
])
dataset_name = "Diving48"

acc = acc/100

# 创建图形对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制柱状图，使用渐变色
xpos, ypos = np.meshgrid(a1_coords, a2_coords)
#xpos, ypos = np.meshgrid(a2_coords, a1_coords)
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)
dx = dy = 0.15 * np.ones_like(zpos)
dz = acc.flatten()

# 手动设置颜色映射的区间，这里夸大了颜色变化范围
norm = plt.Normalize(dz.min(), dz.max())
colors = cm.viridis(norm(dz))

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)

# 设置自定义的轴标签（使用原始提供的标签值）
ax.set_xlabel('a2', color='blue')
ax.set_xticks(a1_coords)
ax.set_xticklabels(a1_labels)

ax.set_ylabel('a1', color='blue')
ax.set_yticks(a2_coords)
ax.set_yticklabels(a2_labels)

ax.set_zlabel('ACC', color='blue')

# 创建一个新的Axes来放置颜色条
cax = fig.add_axes([0.89, 0.1, 0.03, 0.8])
mappable = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
mappable.set_array(dz)
colorbar = fig.colorbar(mappable, cax=cax)
# 设置颜色条刻度格式为两位小数
#colorbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
fig.suptitle("The effect of hyper-parameters in distillation ({})".format(dataset_name))
plt.tight_layout()
# 保存图形为PNG文件，你可以修改文件名和路径
fig.savefig('hyper-parameter-distillation-{}.jpg'.format(dataset_name))