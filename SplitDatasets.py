# coding=utf-8

import os
import random

"""
说明：
将数据集分为train,test，但不存储，存到list中
img和label中的图像命名须按照如下格式：
    labels_dir: 0.tif,1.tif,......10.tif,11.tif,.....,100.tif,101.tif,......830.tif
    imgs_dir: 7_0.tif,7_1.tif,......7_10.tif,7_11.tif,.....,7_100.tif,7_101.tif,......7_830.tif
              8_0.tif,8_1.tif,......8_10.tif,8_11.tif,.....,8_100.tif,8_101.tif,......8_830.tif
              9_0.tif,9_1.tif,......9_10.tif,9_11.tif,.....,9_100.tif,9_101.tif,......9_830.tif
              10_0.tif,10_1.tif,......10_10.tif,10_11.tif,.....,10_100.tif,10_101.tif,......10_830.tif
"""


def SortLabelslist(labels_dir):
    label_list = os.listdir(labels_dir)
    get_key = lambda i: int(i.split('.')[0])
    label_list_sort = sorted(label_list, key=get_key)
    print('label_list_sort', label_list_sort)
    print('len(label_list_sort)', len(label_list_sort))

    return label_list_sort


def data_split(split_dir, labels_dir, ratio_1, shuffle=False):
    full_list =SortLabelslist(labels_dir)
    n_total = len(full_list)
    #  train_length
    offset_1 = int(n_total * ratio_1)
    if n_total == 0 or offset_1 < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)  # random.shuffle()是对原list做修改
    sublist_1 = full_list[:offset_1]
    sublist_2 = full_list[offset_1:]
    print('len(full_list)', len(full_list))
    print('len(sublist_1)', len(sublist_1))
    print('len(sublist_2)', len(sublist_2))

    # select first ratio_1 train_length as train set
    with open(os.path.join(split_dir, 'train.txt'), 'w') as f:
        f.writelines(line + '\n' for line in sublist_1)
    # select second ratio_2 test_length as test set
    with open(os.path.join(split_dir, 'test.txt'), 'w') as f:
        f.writelines(line + '\n' for line in sublist_2)

    return sublist_1, sublist_2


if __name__ == '__main__':
    # 将数据集分为trainlist, testlist
    split_dir = "/data/study_Irrigation/Wa_256"
    labelsets_dir = "/data/study_Irrigation/Wa_256/labels"
    trainlist, testlist = data_split(split_dir, labelsets_dir, 0.8, shuffle=True)


