import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# 指定大区间
start_original = 100
end_original = 250


# 延迟触发的场景
import numpy as np


def crop(original_data, start_original=100, end_original=250):
    """
    对原始数据进行裁剪和补充处理。

    :param original_data: np.array, 原始数据数组
    :param start_original: int, 起始区间的开始位置
    :param end_original: int, 起始区间的结束位置
    :return: np.array, 处理后的数据
    """
    # 在100+20到250-30区间内随机选择一个分割点
    split_point = np.random.randint(start_original + 20, end_original - 30)

    # 随机选择保留前面或后面的数据
    keep_front = np.random.choice([True, False])

    # 指定从哪里开始截取补充数据
    start_supplement = 300

    if keep_front:
        # 保留从开始到分割点的数据
        retained_data = original_data[:split_point]
    else:
        # 保留从分割点到末尾的数据
        retained_data = original_data[split_point:]

    # 计算需要补充的数据长度
    length_to_add = original_data.shape[0] - retained_data.shape[0]
    # 从指定位置开始截取补充数据
    additional_data = original_data[start_supplement : start_supplement + length_to_add]

    # 如果补充数据长度不够，循环补充直至达到所需长度
    while additional_data.shape[0] < length_to_add:
        additional_data = np.append(
            additional_data,
            original_data[
                start_supplement : start_supplement
                + (length_to_add - additional_data.shape[0])
            ],
        )

    # 最终的数据
    if keep_front:
        final_data = np.concatenate([additional_data, retained_data])
    else:
        final_data = np.concatenate([retained_data, additional_data])

    # 添加很小的扰动，假设扰动的标准差为0.01
    # noise = np.random.normal(0, 0.001, final_data.shape)
    # final_data += noise

    # 显示结果
    # print("Original Data Shape:", original_data.shape,end=" | ")
    # print("Final Data Shape:", final_data.shape,end=" | ")
    # print("Split Point:", split_point,end=" | ")
    # print("Keep Front Data:", keep_front)
    return final_data


# 类似于故障标签
def drop_seg(original_data):

    # 指定大区间
    start_original = 100
    end_original = 250

    # 在这个大区间内随机决定一个子区间的起始点和长度
    sub_interval_start = np.random.randint(start_original, end_original)
    sub_len = end_original - start_original + 1
    sub_interval_length = np.random.randint(int(sub_len * 0.1), int(sub_len * 0.5))
    sub_interval_end = min(sub_interval_start + sub_interval_length, end_original)

    # 删除指定的子区间
    part1 = original_data[:sub_interval_start]
    part2 = original_data[sub_interval_end + 1 :]

    # 计算被删除的数据长度
    length_to_remove = sub_interval_end - sub_interval_start + 1

    # 从尾部截取相同长度的数据补充
    replacement_data = original_data[-length_to_remove:]

    # 构建新的数据数组，确保长度仍为500
    new_data = np.concatenate([part1, part2, replacement_data])
    # 假定扰动的标准差为原数据标准差的1%
    noise_std = 0.01 * np.std(new_data)
    noise = np.random.normal(0, noise_std, size=new_data.shape)
    new_data += noise
    # 显示结果
    # print("Original Data Shape:", original_data.shape,end=" | ")
    # print("New Data Shape:", new_data.shape,end=" | ")
    # print("Deleted Interval: [{}-{}]".format(sub_interval_start, sub_interval_end))
    return new_data


# 类似于测试列车标签
def scratch(original_data):

    # 指定区间
    start_original = 100
    end_original = 250
    start_stretch = 100
    end_stretch = 400

    # 提取要拉伸的数据
    data_to_stretch = original_data[start_original : end_original + 1]

    # 计算拉伸后的新长度
    stretch_length = end_stretch - start_stretch + 1

    # 线性插值以拉伸数据
    x_original = np.linspace(0, 1, end_original - start_original + 1)
    x_stretched = np.linspace(0, 1, stretch_length)
    stretched_data = np.interp(x_stretched, x_original, data_to_stretch)
    # 假定扰动的标准差为原数据标准差的1%
    noise_std = 0.01 * np.std(stretched_data)
    noise = np.random.normal(0, noise_std, size=stretched_data.shape)
    stretched_data += noise
    # 构建新的数组
    new_data = np.empty(500 + stretch_length - (end_original - start_original + 1))
    new_data[:start_stretch] = original_data[:start_stretch]
    new_data[start_stretch : start_stretch + stretch_length] = stretched_data
    new_data[start_stretch + stretch_length :] = original_data[end_original + 1 :]
    # print("new_data shape ",new_data.shape)
    return new_data[:500]


def plot_true(name="C23240003_02"):

    # os.makedirs(os.path.join('plots', name), exist_ok=True)
    with open(
        f"./data/C23240003_test/{name}.txt",
        "r",
        encoding="utf-8",
    ) as f:
        data = []
        fns = []
        dim_max = float("-inf")
        dim_min = float("inf")
        pdf = PdfPages(f"plots/{name}_.pdf")
        idx = 0

        for line in f:
            idx += 1

            if idx == 232:
                line = line.strip().split(":")
                data_list = eval(line[1])
                data_list = np.array(data_list)
                data_list = data_list[:1500]
                data_list = data_list[::3]
                # data_list = crop(data_list)
                x = range(0, data_list.shape[0])
                plt.figure(figsize=(10, 5))
                plt.title("normal1")
                plt.plot(x, data_list, linewidth=0.4, label="origin", color='#0072B2')
                pdf.savefig()

                new_list = scratch(data_list)
                plt.figure(figsize=(10, 5))
                plt.title("normal2")
                plt.plot(
                    range(0, len(new_list)),
                    new_list,
                    linewidth=0.4,
                    label="strech",
                    color='#0072B2',
                )
                pdf.savefig()
                print(idx)

            if idx == 233:
                line = line.strip().split(":")
                data_list = eval(line[1])
                data_list = np.array(data_list)
                data_list = data_list[:1500]
                data_list = data_list[::3]
                # data_list = drop_seg(data_list)
                x = range(0, data_list.shape[0])
                plt.figure(figsize=(10, 5))
                plt.title("anormal2")
                plt.plot(x, data_list, linewidth=0.4, label="origin", color='#0072B2')
                pdf.savefig()
                print(idx)

            if idx == 237:
                line = line.strip().split(":")
                data_list = eval(line[1])
                data_list = np.array(data_list)
                data_list = data_list[:1500]
                data_list = data_list[::3]
                # data_list = crop(data_list)
                x = range(0, data_list.shape[0])
                plt.figure(figsize=(10, 5))
                plt.title("anormal1")
                plt.plot(x, data_list, linewidth=0.4, label="origin", color='#0072B2')
                pdf.savefig()
                print(idx)

            if idx > 237:
                break

    pdf.close()


plot_true("C23240003_08")


label_2 = [147, 148, 305, 304, 811, 812, 1051, 1115, 1245]
label_3 = [243, 714, 932, 1391]
label_4 = [232, 237, 238, 240]
skip = [165, 181, 336, 441, 1196, 1269]
map_dict = {
    "C24170018_01": "C23240003_01",
    "C24170018_02": "C23240003_02",
    "C24170018_03": "C23240003_03",
    "C24170018_04": "C23240003_04",
    "C24170018_05": "C23240003_05",
    "C24170018_06": "C23240003_06",
    "C24170018_07": "C23240003_07",
    "C24170018_08": "C23240003_08",
    "C24170018_09": "C23240003_09",
    "C24170018_10": "C23240003_10",
    "C24170018_11": "C23240003_11",
    "C24170018_12": "C23240003_12",
    "C24170018_13": "C23240003_13",
    "C24170018_14": "C23240003_14",
    "C24170018_15": "C23240003_15",
    "C24170018_16": "C23240003_16",
}

uni_name_map_dict = {
    "C24170018_01": "gdbhxjsd",
    "C24170018_02": "gdbzxjsd",
    "C24170018_03": "gdbsxjsd",
    "C24170018_04": "ltsxzdjsd",
    "C24170018_05": "lthxzdjsd",
    "C24170018_06": "ltsxzf",
    "C24170018_07": "lthxzf",
    "C24170018_08": "gdbsxwy",
    "C24170018_09": "gdbhxwy",
    "C24170018_10": "zzzxwy",
    "C24170018_11": "zzsxwy",
    "C24170018_12": "dthxzf",
    "C24170018_13": "ldqj",
    "C24170018_14": "ldwy",
    "C24170018_15": "ltkzrd",
    "C24170018_16": "ltkzyb",
}


def load_elder(name="C23240003_08"):
    skip_edler = [59, 64, 65, 194, 195, 324, 325, 326, 455]
    with open(
        f"./data1/{name}.txt", "r", encoding="utf-8"
    ) as f:
        data = []
        fns = []
        dim_max = float("-inf")
        dim_min = float("inf")
        pdf = PdfPages(f"plots/{name}.pdf")
        idx = 0

        for line in f:
            line = line.strip().split(":")
            data_list = eval(line[1])
            data_list = np.array(data_list)
            data_list = data_list[:1500]
            data_list = data_list[::3]
            data.append(data_list)
        data = np.array(data)
        # print(len(data))
        data = np.delete(data, skip_edler, axis=0)
        # print(len(data))
        return data


def load_bridge(name="C23240003_08"):
    if name in map_dict:
        name = map_dict[name]
    with open(
        f"./data/C23240003_test/{name}.txt",
        "r",
        encoding="utf-8",
    ) as f:
        data = []
        fns = []
        dim_max = float("-inf")
        dim_min = float("inf")
        pdf = PdfPages(f"plots/{name}.pdf")
        idx = 0

        for line in f:
            idx += 1
            line = line.strip().split(":")
            data_list = eval(line[1])
            data_list = np.array(data_list)
            data_list = data_list[:1500]
            data_list = data_list[::3]
            data.append(data_list)

    label_2_np = np.array(label_2)
    label_3_np = np.array(label_3)
    label_4_np = np.array(label_4)
    skip_np = np.array(skip)

    # 合并所有标签到一个数组
    all_labels = np.concatenate((label_2_np, label_3_np, label_4_np, skip_np))
    data = np.array(data)
    noise_data = np.delete(data, all_labels, axis=0)[:923]
    return noise_data


def gen_other_pattern(nomal_datum, is_print=True):
    scratchs = []  # 标签3
    drop_segs = []  # 标签4
    crops = []  # 标签2
    for data in nomal_datum:
        scratchs.append(scratch(data))
        drop_segs.append(drop_seg(data))
        crops.append(crop(data))

    scratchs = np.array(scratchs)[: len(scratchs) - 11]
    drop_segs = np.array(drop_segs)[: len(scratchs) - 8]
    crops = np.array(crops)[: len(scratchs) - 17]
    if is_print:
        print(
            f"scratchs shape : {scratchs.shape} | drop_segs shape : {drop_segs.shape} | crops shape : {crops.shape}"
        )
    return scratchs, drop_segs, crops


def gen_bridge1(name="C23240003_08"):
    bin_label = np.loadtxt("./data/C23240003_test/labels.txt")
    # print(bin_label)
    noise_label = np.where(bin_label == 1)[0]
    true_labels = np.where(bin_label == 0)[0]

    label_2_np = np.array(label_2)
    label_3_np = np.array(label_3)
    label_4_np = np.array(label_4)
    skip_np = np.array(skip)

    # 合并所有标签到一个数组
    all_labels = np.concatenate((label_2_np, label_3_np, label_4_np, skip_np))

    # 使用 np.isin() 来找出 true_labels 中存在于 all_labels 的元素
    mask = np.isin(true_labels, all_labels)
    print(true_labels)
    print(mask)
    # 使用逻辑非操作 (~) 来选择不存在于 all_labels 的元素
    filtered_labels = true_labels[~mask]
    # print(filtered_labels)

    # os.makedirs(os.path.join('plots', name), exist_ok=True)
    with open(
        f"./data/C23240003_test/{name}.txt",
        "r",
        encoding="utf-8",
    ) as f:
        data = []
        fns = []
        dim_max = float("-inf")
        dim_min = float("inf")
        pdf = PdfPages(f"plots/{name}.pdf")
        idx = 0

        for line in f:
            idx += 1
            line = line.strip().split(":")
            data_list = eval(line[1])
            data_list = np.array(data_list)
            data_list = data_list[:1500]
            data_list = data_list[::3]
            data.append(data_list)
            # print(idx)

    print(len(data))
    data = np.array(data)
    nomal_data = data[filtered_labels]
    noise_data = data[noise_label][:932]
    elder_data = load_elder(name)
    new_data = np.concatenate((nomal_data, elder_data))
    scratchs, drop_segs, crops = gen_other_pattern(new_data)
    print(
        f"nomal_data shape : {nomal_data.shape} | noise_data shape : {noise_data.shape} "
    )
    return new_data, noise_data, scratchs, drop_segs, crops  # 0,1,3,4,2


def gen_roadbank(name="C23240002_08"):
    # bin_label = np.loadtxt("./data1/labels.txt")
    # # print(bin_label)
    # noise_label = np.where(bin_label==1)[0]
    # true_labels = np.where(bin_label==0)[0]

    skip = [
        0,
        1,
        2,
        35,
        36,
        37,
        38,
        39,
        43,
        66,
        67,
        107,
        131,
        171,
        295,
        259,
        297,
        300,
        324,
        364,
        365,
        388,
        413,
        453,
        492,
        493,
        517,
        557,
        558,
        560,
        582,
        546,
        710,
        774,
        838,
        902,
        966,
        1157,
        1158,
        1159,
        1160,
        1161,
    ]

    # 合并所有标签到一个数组
    all_labels = np.array(skip)

    # 使用 np.isin() 来找出 true_labels 中存在于 all_labels 的元素
    # mask = np.isin(true_labels, all_labels)

    # 使用逻辑非操作 (~) 来选择不存在于 all_labels 的元素
    # filtered_labels = true_labels[~mask]
    # print(filtered_labels)

    # os.makedirs(os.path.join('plots', name), exist_ok=True)
    with open(
        f"./data/{name}.txt", "r", encoding="utf-8"
    ) as f:
        data = []
        fns = []
        dim_max = float("-inf")
        dim_min = float("inf")
        pdf = PdfPages(f"plots/{name}.pdf")
        idx = 0

        for line in f:
            idx += 1
            line = line.strip().split(":")
            data_list = eval(line[1])
            data_list = np.array(data_list)
            data_list = data_list[:1500]
            data_list = data_list[::3]
            data.append(data_list)
            if idx > 1200:
                break
            # print(idx)

    # print(len(data))
    data = np.array(data)
    nomal_data = np.delete(data, all_labels, axis=0)

    # noise_data = data[noise_label][:932]
    elder_data = load_elder_road_bank(name)
    # new_data = np.concatenate((nomal_data,elder_data))
    scratchs, drop_segs, crops = gen_other_pattern(nomal_data)
    print(
        f"nomal_data shape : {nomal_data.shape} | noise_data shape : {elder_data.shape} "
    )
    return nomal_data, elder_data, scratchs, drop_segs, crops
    # return nomal_data[:np.random.randint(319,380)],elder_data[:np.random.randint(319,380)]\
    #     ,scratchs[:np.random.randint(319,380)],drop_segs[:np.random.randint(319,380)],crops[:np.random.randint(319,380)] # 0,1,3,4,2
    # print(len(noise_label[0]))


def gen_bridge(name="C23240003_08"):
    # bin_label = np.loadtxt("./data1/labels.txt")
    # # print(bin_label)
    # noise_label = np.where(bin_label==1)[0]
    # true_labels = np.where(bin_label==0)[0]

    skip = [
        0,
        3,
        5,
        9,
        10,
        99,
        158,
        222,
        286,
        350,
        414,
        478,
        543,
        607,
        671,
        735,
        799,
        864,
        928,
        992,
    ]

    # 合并所有标签到一个数组
    all_labels = np.array(skip)

    # 使用 np.isin() 来找出 true_labels 中存在于 all_labels 的元素
    # mask = np.isin(true_labels, all_labels)

    # 使用逻辑非操作 (~) 来选择不存在于 all_labels 的元素
    # filtered_labels = true_labels[~mask]
    # print(filtered_labels)

    # os.makedirs(os.path.join('plots', name), exist_ok=True)
    with open(
        f"./data/C03_new/{name}.txt", "r", encoding="utf-8"
    ) as f:
        data = []
        fns = []
        dim_max = float("-inf")
        dim_min = float("inf")
        pdf = PdfPages(f"plots/{name}.pdf")
        idx = 0

        for line in f:
            idx += 1
            line = line.strip().split(":")
            data_list = eval(line[1])
            data_list = np.array(data_list)
            # data_list = data_list[:-4]
            data_list = data_list[:: len(data_list) // 500]
            data_list = data_list[:500]
            data.append(data_list)
            if idx % 200 == 0:
                print(idx)
            if idx > 1000:
                break
            # print(idx)
    # data = data[11:1000]
    print(len(data), len(data[0]))
    data = np.array(data)
    nomal_data = np.delete(data, all_labels, axis=0)

    # noise_data = data[noise_label][:932]
    noise_data = load_bridge(name)
    # new_data = np.concatenate((nomal_data,elder_data))
    scratchs, drop_segs, crops = gen_other_pattern(nomal_data)
    print(
        f"nomal_data shape : {nomal_data.shape} | noise_data shape : {noise_data.shape} "
    )
    return nomal_data, noise_data, scratchs, drop_segs, crops  # 0,1,3,4,2
    # return nomal_data[:np.random.randint(319,380)],noise_data[:np.random.randint(319,380)]\
    #     ,scratchs[:np.random.randint(319,380)],drop_segs[:np.random.randint(319,380)],crops[:np.random.randint(319,380)] # 0,1,3,4,2
    # print(len(noise_label[0]))


def load_elder_road_bank(name="C23240003_08"):
    # skip_edler = [59,64,65,194,195,324,325,326,455]
    bin_label = np.loadtxt("./data1/labels.txt")
    # print(bin_label)
    # noise_label = np.where(bin_label==1)[0]
    true_labels = np.where(bin_label == 0)[0]
    with open(
        f"./data1/{name}.txt", "r", encoding="utf-8"
    ) as f:
        data = []
        fns = []
        dim_max = float("-inf")
        dim_min = float("inf")
        pdf = PdfPages(f"plots/{name}.pdf")
        idx = 0

        for line in f:
            line = line.strip().split(":")
            data_list = eval(line[1])
            data_list = np.array(data_list)
            data_list = data_list[:1500]
            data_list = data_list[::3]
            data.append(data_list)
        data = np.array(data)
        # print(len(data))
        data = np.delete(data, true_labels, axis=0)
        # print(len(data))
        return data

def gen_all_data(data_name="Bridge"):
    task_name_list = [
        "C23240002_01",
        "C23240002_02",
        "C23240002_05",
        "C23240002_06",
        "C23240002_07",
    ]

    nomal_data_list, noise_data_list, scratch_list, drop_seg_list, crop_list = (
        [],
        [],
        [],
        [],
        [],
    )
    for name_item in task_name_list:
        print("=" * 50)
        print(data_name, "=========", name_item)

        nomal_data, noise_data, scratchs, drop_segs, crops = gen_roadbank(name_item)
        print("nomal_data shape ", nomal_data.shape)
        nomal_data_list.append(nomal_data)
        noise_data_list.append(noise_data)
        scratch_list.append(scratchs)
        drop_seg_list.append(drop_segs)
        crop_list.append(crops)
        print("=" * 50)
        data_name = uni_name_map_dict[name_item]

    label = [0] * len(nomal_data_list[0])
    label.extend([1] * len(noise_data_list[0]))
    # label.extend([3]* len(scratch_list[0]))
    # label.extend([4]* len(drop_seg_list[0]))
    # label.extend([2]* len(crop_list[0]))
    label.extend([0] * len(scratch_list[0]))
    label.extend([1] * len(drop_seg_list[0]))
    label.extend([1] * len(crop_list[0]))
    # print(label,len(label))
    # print(nomal_data_list)
    print(np.array(nomal_data_list).shape)
    nomal_data_list = np.array(nomal_data_list).transpose(1, 0, 2)
    noise_data_list = np.array(noise_data_list).transpose(1, 0, 2)
    scratch_list = np.array(scratch_list).transpose(1, 0, 2)
    drop_seg_list = np.array(drop_seg_list).transpose(1, 0, 2)
    crop_list = np.array(crop_list).transpose(1, 0, 2)
    print(nomal_data_list.shape, noise_data_list.shape)
    data = np.concatenate(
        (nomal_data_list, noise_data_list, scratch_list, drop_seg_list, crop_list),
        axis=0,
    )
    print(data.shape)
    label = np.array(label)

    # gen_data_pt(data, label, data_name)

    # gen_bridge("C24170018_01")
    # gen_all_data("Bridge")

    # gen_roadbank('C23240002_06')


def plot_origin():
    task_name_list = [
        "C23240002_01",
        "C23240002_02",
        "C23240002_05",
        "C23240002_06",
        "C23240002_07",
    ]
    all_sensors = []
    for name_item in task_name_list:
        with open(
            f"./data/{name_item}.txt",
            "r",
            encoding="utf-8",
        ) as f:
            data = []
            fns = []
            dim_max = float("-inf")
            dim_min = float("inf")
            pdf = PdfPages(f"plots/{name_item}.pdf")
            idx = 0

            for line in f:
                idx += 1
                line = line.strip().split(":")
                data_list = eval(line[1])
                data_list = np.array(data_list)
                data_list = data_list[:1500]
                data_list = data_list[::3]
                data.append(data_list)
                if idx > 122:
                    break
            all_sensors.append(data)
    all_sensors = np.array(all_sensors)
    print(all_sensors.shape)
    import matplotlib.pyplot as plt

    # 创建一个包含 5 个子图的画布
    fig, axes = plt.subplots(all_sensors.shape[0], 1, figsize=(20, 15))
    plt.title("RoadBank Normal Sambles")
    # 绘制每个子图
    for i in range(all_sensors.shape[0]):
        axes[i].plot(all_sensors[i, 121, :])  # 取出每个样本 (每个子图)
        axes[i].set_title(f"dim {i+1}")  # 设置每个子图的标题
        axes[i].set_xlabel("Time Step")  # 设置x轴标签
        axes[i].set_ylabel("Value")  # 设置y轴标签

    # 调整布局，避免子图之间的重叠
    plt.tight_layout()

    # 保存图形为 PNG 格式
    plt.savefig("./multi_sample_plot.png")

    # 显示图形
    plt.show()


def plot_abnormaly():
    task_name_list = [
        "C23240002_01",
        "C23240002_02",
        "C23240002_05",
        "C23240002_06",
        "C23240002_07",
    ]
    all_sensors = []
    for name_item in task_name_list:
        with open(
            f"./data1/{name_item}.txt",
            "r",
            encoding="utf-8",
        ) as f:
            data = []
            fns = []
            dim_max = float("-inf")
            dim_min = float("inf")
            pdf = PdfPages(f"plots/{name_item}.pdf")
            idx = 0

            for line in f:
                idx += 1
                line = line.strip().split(":")
                data_list = eval(line[1])
                data_list = np.array(data_list)
                data_list = data_list[:1500]
                data_list = data_list[::3]
                data.append(data_list)
                if idx > 122:
                    break
            all_sensors.append(data)
    all_sensors = np.array(all_sensors)
    print(all_sensors.shape)
    import matplotlib.pyplot as plt

    # 创建一个包含 8 个子图的画布
    fig, axes = plt.subplots(8, 1, figsize=(20, 24))
    plt.title("RoadBank Normal Sambles")

    # 正常图
    axes[0].plot(all_sensors[1, 6, :])  # 取出每个样本 (每个子图)
    axes[0].set_title(f"normal")  # 设置每个子图的标题
    axes[0].set_xlabel("Time Step")  # 设置x轴标签
    axes[0].set_ylabel("Value")  # 设置y轴标签

    _crop_ = drop_seg(all_sensors[1, 6, :])
    axes[1].plot(_crop_)  # 取出每个样本 (每个子图)
    axes[1].set_title(f"anormaly 1")  # 设置每个子图的标题
    axes[1].set_xlabel("Time Step")  # 设置x轴标签
    axes[1].set_ylabel("Value")  # 设置y轴标签

    _crop_ = crop(all_sensors[1, 6, :])
    axes[2].plot(_crop_)  # 取出每个样本 (每个子图)
    axes[2].set_title(f"anormaly 2")  # 设置每个子图的标题
    axes[2].set_xlabel("Time Step")  # 设置x轴标签
    axes[2].set_ylabel("Value")  # 设置y轴标签

    _crop_ = crop(all_sensors[1, 6, :])
    axes[3].plot(_crop_)  # 取出每个样本 (每个子图)
    axes[3].set_title(f"anormaly 3")  # 设置每个子图的标题
    axes[3].set_xlabel("Time Step")  # 设置x轴标签
    axes[3].set_ylabel("Value")  # 设置y轴标签

    _crop_ = drop_seg(all_sensors[3, 6, :])
    axes[4].plot(_crop_)  # 取出每个样本 (每个子图)
    axes[4].set_title(f"anormaly 4")  # 设置每个子图的标题
    axes[4].set_xlabel("Time Step")  # 设置x轴标签
    axes[4].set_ylabel("Value")  # 设置y轴标签

    _crop_ = drop_seg(all_sensors[1, 6, :])
    axes[5].plot(_crop_)  # 取出每个样本 (每个子图)
    axes[5].set_title(f"anormaly 5")  # 设置每个子图的标题
    axes[5].set_xlabel("Time Step")  # 设置x轴标签
    axes[5].set_ylabel("Value")  # 设置y轴标签

    _crop_ = scratch(all_sensors[3, 6, :])
    axes[6].plot(_crop_)  # 取出每个样本 (每个子图)
    axes[6].set_title(f"anormaly 6")  # 设置每个子图的标题
    axes[6].set_xlabel("Time Step")  # 设置x轴标签
    axes[6].set_ylabel("Value")  # 设置y轴标签

    _crop_ = scratch(all_sensors[0, 6, :])
    axes[7].plot(_crop_)  # 取出每个样本 (每个子图)
    axes[7].set_title(f"anormaly 7")  # 设置每个子图的标题
    axes[7].set_xlabel("Time Step")  # 设置x轴标签
    axes[7].set_ylabel("Value")  # 设置y轴标签

    # 调整布局，避免子图之间的重叠
    plt.tight_layout()

    # 保存图形为 PNG 格式
    plt.savefig("./multi_sample_plot3.png")

    # 显示图形
    plt.show()


# plot_abnormaly()
# plot_origin()
