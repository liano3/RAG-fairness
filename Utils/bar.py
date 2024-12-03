import pandas as pd
import numpy as np
import scienceplots
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

fig, axs = plt.subplots(1, 4, figsize=(24, 6))  # 创建1x4的子图

font1 = {'size': 30}
labelsize = 20
labelsize_x = 20

# 数据
x = np.arange(3)
bar_width = 0.2
tick_label = ["K=3", "K=5", "K=10"]

alpha = 0.6

# 第一个子图
ax = axs[0]
Base = [0.25, 0.348, 0.47]
Filter = [0.042, 0.049, 0.061]
Train = [0.208, 0.257, 0.411]
ax.bar(x - bar_width, Base, bar_width, linewidth=2, edgecolor='black', alpha=alpha, align="center", label='Base', color='#2100D3')
ax.bar(x, Train, bar_width, linewidth=2, edgecolor='black', alpha=alpha, align="center", label='Train', color='#2662FE')
ax.bar(x + bar_width, Filter, bar_width, linewidth=2, edgecolor='black', alpha=alpha, align="center", label='Filter', color='#18ADF9')

ax.set_xlabel("(a) Fairness on Web", font1)
ax.set_ylabel("Fairness (agreement-ratio)", font1)
ax.set_xticks(x)
ax.set_xticklabels(tick_label)
ax.set_yticks(np.arange(0, 0.51, 0.1))
ax.tick_params(axis='y', labelsize=labelsize)
ax.tick_params(axis='x', labelsize=labelsize_x)
ax.set_xlim((-0.5, 2.5))
ax.set_ylim([0, 0.5])
ax.legend(loc='upper left', fontsize=20)  # 图例放右上角
ax.grid()

# 第二个子图
ax = axs[1]
Base = [0.163, 0.279, 0.379]
Filter = [0.044, 0.051, 0.054]
Train = [0.151, 0.242, 0.369]
ax.bar(x - bar_width, Base, bar_width, linewidth=2, edgecolor='black', alpha=alpha, align="center", label='Base', color='#2100D3')
ax.bar(x, Train, bar_width, linewidth=2, edgecolor='black', alpha=alpha, align="center", label='Train', color='#2662FE')
ax.bar(x + bar_width, Filter, bar_width, linewidth=2, edgecolor='black', alpha=alpha, align="center", label='Filter', color='#18ADF9')

ax.set_xlabel("(b) Fairness on Wiki", font1)
ax.set_ylabel("Fairness (agreement-ratio)", font1)
ax.set_xticks(x)
ax.set_xticklabels(tick_label)
ax.set_yticks(np.arange(0, 0.51, 0.1))
ax.tick_params(axis='y', labelsize=labelsize)
ax.tick_params(axis='x', labelsize=labelsize_x)
ax.set_xlim((-0.5, 2.5))
ax.set_ylim([0, 0.5])
ax.grid()

# 第三个子图
ax = axs[2]
Base = [0.214, 0.215, 0.221]
Filter = [0.21, 0.229, 0.244]
Train = [0.221, 0.222, 0.224]
ax.bar(x - bar_width, Base, bar_width, linewidth=2, edgecolor='black', alpha=alpha, align="center", label='Base', color='#2100D3')
ax.bar(x, Train, bar_width, linewidth=2, edgecolor='black', alpha=alpha, align="center", label='Train', color='#2662FE')
ax.bar(x + bar_width, Filter, bar_width, linewidth=2, edgecolor='black', alpha=alpha, align="center", label='Filter', color='#18ADF9')

ax.set_xlabel("(c) Utility on Web", font1)
ax.set_ylabel("Utility (accuracy)", font1)
ax.set_xticks(x)
ax.set_xticklabels(tick_label)
ax.set_yticks(np.arange(0, 0.31, 0.1))
ax.tick_params(axis='y', labelsize=labelsize)
ax.tick_params(axis='x', labelsize=labelsize_x)
ax.set_xlim((-0.5, 2.5))
ax.set_ylim([0, 0.3])
ax.grid()

# 第四个子图
ax = axs[3]
Base = [0.179, 0.186, 0.191]
Filter = [0.185, 0.196, 0.198]
Train = [0.183, 0.187, 0.2]
ax.bar(x - bar_width, Base, bar_width, linewidth=2, edgecolor='black', alpha=alpha, align="center", label='Base', color='#2100D3')
ax.bar(x, Train, bar_width, linewidth=2, edgecolor='black', alpha=alpha, align="center", label='Train', color='#2662FE')
ax.bar(x + bar_width, Filter, bar_width, linewidth=2, edgecolor='black', alpha=alpha, align="center", label='Filter', color='#18ADF9')

ax.set_xlabel("(d) Utility on Wiki", font1)
ax.set_ylabel("Utility (accuracy)", font1)
ax.set_xticks(x)
ax.set_xticklabels(tick_label)
ax.set_yticks(np.arange(0, 0.31, 0.1))
ax.tick_params(axis='y', labelsize=labelsize)
ax.tick_params(axis='x', labelsize=labelsize_x)
ax.set_xlim((-0.5, 2.5))
ax.set_ylim([0, 0.3])
ax.grid()

plt.tight_layout()
plt.savefig('./bar.pdf', bbox_inches='tight')
plt.show()
