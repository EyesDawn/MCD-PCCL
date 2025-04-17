import matplotlib.pyplot as plt
marker_dict = {
    "HAR":'o',
    "EPI":'*',
    "ISRUC":'v'
}
title_size = 36
label_aixs_size = 28

export_type = "svg"


################################### batch size ##########################################################
# plt.savefig(f'batch_size_mf1.svg',dpi=600,format='svg')
plt.figure(figsize=(20, 20))
plt.subplot(2, 2, 1) 
x = [32, 64, 128, 256, 512]

x_axis = range(len(x))
data = {
    "HAR": [94.98925461, 94.59337179, 94.49157335, 94.94401086, 94.79696867],
    "EPI": [98.01449275, 97.85507246, 98.23188406, 98.33333333, 98.28985507],
    "ISRUC": [82.18859139, 81.42297245, 82.05828483, 82.12495149, 83.12495149]
}



for key, value in data.items():
    plt.plot(x_axis, value, marker=marker_dict[key], label=key,linewidth=5,markersize=15)

plt.xlabel('Batch size',fontsize=title_size)
plt.ylabel('ACC (%)',fontsize=title_size)
plt.legend(ncol=3, bbox_to_anchor=(1.85, 1.22),fontsize=title_size)

plt.xticks(x_axis, x,fontsize=label_aixs_size)  # 设置自定义横坐标标签
plt.yticks(fontsize=label_aixs_size)  # 设置自定义横坐标标签
plt.grid(True, which='both', linestyle='--', axis="y")  # 绘制横向的虚线网格线

################################### Representation Dimension ##########################################################

plt.subplot(2, 2, 2) 
data = {
    "HAR": [89.60525,	90.91732,	93.95996,	95.62267,	95.19285,	95.92806],
    "EPI": [94.59420,	96.47826,	97.31884,	97.39130,	97.71014,	98.17391],
    "ISRUC": [77.63097,	79.15483,	81.66473,	81.53492,	82.09856,	82.60652]
}



x = [32,		64,		128,		256,		512,		1024,	]

x_axis = range(len(x))

for key, value in data.items():
    plt.plot(x_axis, value, marker=marker_dict[key], label=key,linewidth=5,markersize=15)

plt.xlabel('Representation Dimension',fontsize=title_size)
plt.ylabel('ACC (%)',fontsize=title_size)
#plt.legend(ncol=3, bbox_to_anchor=(0.85, 1.12),fontsize="18")

plt.xticks(x_axis, x,fontsize=label_aixs_size)  # 设置自定义横坐标标签
plt.yticks(fontsize=label_aixs_size)  # 设置自定义横坐标标签
plt.grid(True, which='both', linestyle='--', axis="y")  # 绘制横向的虚线网格线

plt.show()



################################### lambda for loss ##########################################################

plt.subplot(2, 2, 3) 
data = {
    "HAR": [95.69053,	95.33989,	95.00057,	94.88746,	94.49157,	93.82423],
    "EPI": [98.05797,	98.05797,	97.95652,	98.11594,	98.08696,	98.08696],
    "ISRUC": [82.12844,	83.00504,	81.92394,	82.48118,	82.83042,	82.69461]
}




x = [0.1,		0.2,		0.4,		0.6,		0.8	,	1	]

x_axis = range(len(x))

for key, value in data.items():
    plt.plot(x_axis, value, marker=marker_dict[key], label=key,linewidth=5,markersize=15)

plt.xlabel(r'$\lambda$',fontsize=title_size)
plt.ylabel('ACC (%)',fontsize=title_size)
#plt.legend(ncol=3, bbox_to_anchor=(0.85, 1.12),fontsize="18")

plt.xticks(x_axis, x,fontsize=label_aixs_size)  # 设置自定义横坐标标签
plt.yticks(fontsize=label_aixs_size)  # 设置自定义横坐标标签
plt.grid(True, which='both', linestyle='--', axis="y")  # 绘制横向的虚线网格线

plt.show()

################################### temperature ##########################################################

plt.subplot(2, 2, 4) 
data = {
    "HAR": [86.93587,	90.22735,	94.16356,	95.35120,	93.34917,	90.49881],
    "EPI": [97.26087,	97.95652,	97.60870,	98.00000,	97.86957,	97.82609],
    "ISRUC": [76.21420,	78.11409,	81.09313,	82.49010,	78.54249,	76.21420]
}

x = [0.001,		0.01,		0.05,		0.1,		0.5	,	1	]

x_axis = range(len(x))

for key, value in data.items():
    plt.plot(x_axis, value, marker=marker_dict[key], label=key,linewidth=5,markersize=15)

plt.xlabel(r'$t$',fontsize=title_size)
plt.ylabel('ACC (%)',fontsize=title_size)
#plt.legend(ncol=3, bbox_to_anchor=(0.85, 1.12),fontsize="18")

plt.xticks(x_axis, x,fontsize=label_aixs_size)  # 设置自定义横坐标标签
plt.yticks(fontsize=label_aixs_size)  # 设置自定义横坐标标签
plt.grid(True, which='both', linestyle='--', axis="y")  # 绘制横向的虚线网格线

plt.show()


#plt.savefig(f'batch_size_mf1.svg',dpi=600,format='svg')
plt.savefig(f'sensitive.{export_type}',dpi=600,format=export_type)

