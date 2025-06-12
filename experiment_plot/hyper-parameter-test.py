import matplotlib
matplotlib.use('Agg')  # 在导入 pyplot 之前设置
import matplotlib.pyplot as plt
import numpy as np
# 设置参数范围和假设的精度值
alpha_values = np.arange(0.1, 1.0, 0.1)

# 假设的精度数据 - α=0.3时达到峰值
# HMDB51数据集
hmdb51_acc = [0.552, 0.556, 0.555, 0.565, 0.567, 0.553, 0.562, 0.549, 0.54]
# Diving48数据集
diving48_acc = [0.426, 0.432, 0.432, 0.443, 0.425, 0.438, 0.430, 0.437, 0.430]
#diving48_acc = [0.426, 0.432, 0.443, 0.432, 0.425, 0.438, 0.430, 0.437, 0.430]
hmdb51_acc_1shot = [0.372, 0.378, 0.382, 0.377, 0.4003, 0.3828, 0.3988, 0.379, 0.3746]
diving48_acc_1shot = [0.3228, 0.3196, 0.328, 0.3246, 0.3156, 0.332, 0.3226, 0.33, 0.318]

# 创建图形
plt.figure(figsize=(8, 5))

# 绘制两条曲线
plt.plot(alpha_values, hmdb51_acc, 'b-s', label='HMDB51-5shot', linewidth=2, markersize=8)
plt.plot(alpha_values, diving48_acc, 'r-^', label='Diving48-5shot', linewidth=2, markersize=8)

plt.plot(alpha_values, hmdb51_acc_1shot, 'b--s', label='HMDB51-1shot', linewidth=2, markersize=8)
plt.plot(alpha_values, diving48_acc_1shot, 'r--^', label='Diving48-1shot', linewidth=2, markersize=8)

# 标记最高点
max_hmdb = np.argmax(hmdb51_acc)
max_diving = np.argmax(diving48_acc)

max_hmdb_1shot = np.argmax(hmdb51_acc_1shot)
max_diving_1shot = np.argmax(diving48_acc_1shot)

plt.scatter(alpha_values[max_hmdb], hmdb51_acc[max_hmdb], color='blue', s=100, zorder=5,  marker='o')
plt.scatter(alpha_values[max_diving], diving48_acc[max_diving], color='red', s=100, zorder=5, marker='o')
plt.scatter(alpha_values[max_hmdb_1shot], hmdb51_acc_1shot[max_hmdb_1shot], color='blue', s=100, zorder=5,  marker='o')
plt.scatter(alpha_values[max_diving_1shot], diving48_acc_1shot[max_diving_1shot], color='red', s=100, zorder=5, marker='o')

# 添加图例和标签
plt.title('The Accuracy According to α', fontsize=14)
plt.xlabel('α', fontsize=12)
plt.ylabel('Recognition Accuracy', fontsize=12)
plt.legend(fontsize=12)

# 设置坐标轴范围
plt.xlim(0.05, 0.95)
plt.ylim(0.30, 0.60)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.6)

# 将图例移动到左下角
#plt.legend(fontsize=12, loc='lower left')

# 或者使用 bbox_to_anchor 精确指定位置，例如移动到图形外的右侧中间
plt.legend(fontsize=10, bbox_to_anchor=(0.99, 0.68), loc='center right', borderaxespad=0.)

# 保存为PDF
plt.savefig('accuracy_according_to_a.pdf', format='pdf', bbox_inches='tight')

# 显示图形
plt.show()