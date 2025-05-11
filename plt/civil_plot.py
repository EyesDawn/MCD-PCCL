import matplotlib.pyplot as plt
marker_dict = {
    "Bridge":'d',
    "Transition":'*',
}
title_size = 30
label_aixs_size = 20

export_type = "png"


################################### batch size ##########################################################
# plt.savefig(f'batch_size_mf1.svg',dpi=600,format='svg')
plt.figure(figsize=(20, 16))
plt.subplot(2, 2, 1) 
x = [32, 64, 128, 256, 512]

x_axis = range(len(x))
data = {
    "Transition": [0.984792055, 0.983860956, 0.984792055, 0.987895717, 0.986964618],
    "Bridge": [0.823206897, 0.893103448, 0.925137931, 0.891324138, 0.856206897],
}


for key, value in data.items():
    plt.plot(x_axis, value, marker=marker_dict[key], label=key, linewidth=5, markersize=15)
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.xlabel(u'Batch Size',fontsize=title_size)
plt.ylabel(u'ACC (%)',fontsize=title_size)
plt.legend(ncol=3, bbox_to_anchor=(0.50, 1),fontsize=title_size)

plt.xticks(x_axis, x,fontsize=label_aixs_size)  # 设置自定义横坐标标签
plt.yticks(fontsize=label_aixs_size)  # 设置自定义横坐标标签
plt.grid(True, which='both', linestyle='--', axis="y")  # 绘制横向的虚线网格线
# plt.xlabel('Batch size',fontsize=title_size)
# plt.ylabel('ACC (%)',fontsize=title_size)
# plt.legend(ncol=3, bbox_to_anchor=(1.85, 1.22),fontsize=title_size)

# plt.xticks(x_axis, x,fontsize=label_aixs_size)  # 设置自定义横坐标标签
# plt.yticks(fontsize=label_aixs_size)  # 设置自定义横坐标标签
# plt.grid(True, which='both', linestyle='--', axis="y")  # 绘制横向的虚线网格线
################################### Representation Dimension ##########################################################

plt.subplot(2, 2, 2) 
data = {
    "Transition": [0.961824953, 0.969739292, 0.983240223, 0.984171322, 0.987895717, 0.980757294],
    "Bridge": [0.67816092, 0.672413793, 0.738505747, 0.775862069, 0.801724138, 0.790229885]
}

x = [32,		64,		128,		256,		320,		512,	]

x_axis = range(len(x))

for key, value in data.items():
    plt.plot(x_axis, value, marker=marker_dict[key], label=key, linewidth=5, markersize=15)

plt.xlabel(u'Representation Dimension',fontsize=title_size)
# plt.ylabel('ACC (%)',fontsize=title_size)
#plt.legend(ncol=3, bbox_to_anchor=(0.85, 1.12),fontsize="18")

plt.xticks(x_axis, x,fontsize=label_aixs_size)  # 设置自定义横坐标标签
plt.yticks(fontsize=label_aixs_size)  # 设置自定义横坐标标签
plt.grid(True, which='both', linestyle='--', axis="y")  # 绘制横向的虚线网格线

plt.show()

################################### lambda for loss ##########################################################

plt.subplot(2, 2, 3) 
data = {
    "Transition":[0.985102421, 0.984636872, 0.981378026, 0.985412787, 0.981998759, 0.985102421, 0.985723153],
    "Bridge": [0.784482759, 0.781609195, 0.798850575, 0.787356322, 0.791494253, 0.798505747, 0.807471264]
}

x = [0.01,0.1,		0.2,		0.4,		0.6,		0.8	,	1	]

x_axis = range(len(x))

for key, value in data.items():
    plt.plot(x_axis, value, marker=marker_dict[key], label=key, linewidth=5, markersize=15)

plt.xlabel(r'$\alpha$',fontsize=title_size)
plt.ylabel('ACC (%)',fontsize=title_size)
#plt.legend(ncol=3, bbox_to_anchor=(0.85, 1.12),fontsize="18")

plt.xticks(x_axis, x,fontsize=label_aixs_size)  # 设置自定义横坐标标签
plt.yticks(fontsize=label_aixs_size)  # 设置自定义横坐标标签
plt.grid(True, which='both', linestyle='--', axis="y")  # 绘制横向的虚线网格线

plt.show()

################################### gamma ##########################################################

plt.subplot(2, 2, 4) 
data = {
    "Transition": [0.986343886, 0.983240223, 0.982309125, 0.982929857, 0.980757294],
    "Bridge": [0.792643678, 0.794252874, 0.795517241, 0.795977011, 0.79137931]
}

x = [0.001,		0.01,		0.05,		0.1,	1	]

x_axis = range(len(x))

for key, value in data.items():
    plt.plot(x_axis, value, marker=marker_dict[key], label=key, linewidth=5, markersize=15)

plt.xlabel(r'$\gamma$',fontsize=title_size)
# plt.ylabel('ACC (%)',fontsize=title_size)
#plt.legend(ncol=3, bbox_to_anchor=(0.85, 1.12),fontsize="18")

plt.xticks(x_axis, x,fontsize=label_aixs_size)  # 设置自定义横坐标标签
plt.yticks(fontsize=label_aixs_size)  # 设置自定义横坐标标签
plt.grid(True, which='both', linestyle='--', axis="y")  # 绘制横向的虚线网格线

plt.show()


plt.savefig(f'civil1.{export_type}',dpi=600,format=export_type)