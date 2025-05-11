import matplotlib.pyplot as plt
import numpy as np
export_type = "png"
title_size = 30
label_aixs_size = 20
# 数据
labels = ['HC+CoT', 'TSD+HC', 'TSD+CoT', 'TSD+HC+CoT']
roadbank = [95.28, 97.01, 94.8, 98.78]
bridge = [77.23, 63.55, 60.07, 80.17]

x = np.arange(len(labels))  # 标签位置
width = 0.25  # 柱状图宽度
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
fig, ax = plt.subplots(figsize=(10,6))

# 绘制柱状图
rects1 = ax.bar(x - width/2, roadbank, width, label='Transition')
rects2 = ax.bar(x + width/2, bridge, width, label='Bridge')

# 添加一些文本标签
ax.set_ylabel('ACC (%)',fontsize=title_size)
ax.set_xticks(x)
ax.set_xticklabels(labels,fontsize=label_aixs_size)
# ax.legend(ncol=1, bbox_to_anchor=(0.40, 0.8),fontsize=label_aixs_size)
ax.legend(ncol=2, bbox_to_anchor=(0.5, 1.15), loc='center', fontsize=label_aixs_size)

# 添加数值标签
def autolabel(rects):
    """为每个柱形图添加数值标签"""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# autolabel(rects1)
# autolabel(rects2)

fig.tight_layout()

plt.show()
plt.savefig(f'ablation1.{export_type}',dpi=600,format=export_type)