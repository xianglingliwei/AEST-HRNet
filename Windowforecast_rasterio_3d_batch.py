# coding=utf-8

import os
import rasterio
import numpy as np
import datetime
import math
# import sys
import torch
import _init_paths
import models
from config import config
from config import update_config


def ReadRaster(file):
    with rasterio.open(file) as src:
        if src == None:
            print(file + "文件无法打开")
        image = src.read()
        if image.shape[0] == 1:
            image = np.squeeze(image)
        src_meta = src.meta
    return image, src_meta


def TifCroppingArray(img, block_w, block_h, area_perc):
    # 计算SideLength,即
    SideLength = int((1 - math.sqrt(area_perc)) * block_w / 2)
    #  裁剪链表
    TifArrayReturn = []
    #  列上图像块数目 即行数
    ColumnNum = int((img.shape[1] - SideLength * 2) / (block_w - SideLength * 2))
    #  行上图像块数目 即列数
    RowNum = int((img.shape[2] - SideLength * 2) / (block_h - SideLength * 2))
    for i in range(ColumnNum):
        TifArray = []
        for j in range(RowNum):
            cropped = img[:, i * (block_w - SideLength * 2): i * (block_w - SideLength * 2) + block_w,
                          j * (block_h - SideLength * 2): j * (block_h - SideLength * 2) + block_h]
            TifArray.append(cropped)
        TifArrayReturn.append(TifArray)
    #  考虑到行列会有剩余的情况，向前裁剪一行和一列
    #  向前裁剪最后一列
    for i in range(ColumnNum):
        cropped = img[:, i * (block_h - SideLength * 2): i * (block_h - SideLength * 2) + block_h,
                      (img.shape[2] - block_h): img.shape[2]]
        TifArrayReturn[i].append(cropped)
    #  向前裁剪最后一行
    TifArray = []
    for j in range(RowNum):
        cropped = img[:, (img.shape[1] - block_w): img.shape[1],
                      j * (block_w-SideLength*2): j * (block_w - SideLength * 2) + block_w]
        TifArray.append(cropped)
    #  向前裁剪右下角
    cropped = img[:, (img.shape[1] - block_w): img.shape[1], (img.shape[2] - block_h): img.shape[2]]
    TifArray.append(cropped)
    TifArrayReturn.append(TifArray)
    #  列上的剩余数
    ColumnOver = (img.shape[1] - SideLength * 2) % (block_w - SideLength * 2) + SideLength
    #  行上的剩余数
    RowOver = (img.shape[2] - SideLength * 2) % (block_h - SideLength * 2) + SideLength
    return TifArrayReturn, RowOver, ColumnOver


def Result(shape, TifArray, npyfile, block_w, area_perc, RowOver, ColumnOver):
    RepetitiveLength = int((1 - math.sqrt(area_perc)) * block_w / 2)

    result = np.zeros(shape, np.uint8)
    #  j来标记行数
    j = 0
    for i, img in enumerate(npyfile):
        #  最左侧一列特殊考虑，左边的边缘要拼接进去
        if(i % len(TifArray[0]) == 0):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0: 256 - RepetitiveLength, 0: 256-RepetitiveLength] = img[0: 256 - RepetitiveLength, 0: 256 - RepetitiveLength]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif(j == len(TifArray) - 1):
                #  原来错误的
                #result[shape[0] - ColumnOver : shape[0], 0 : 512 - RepetitiveLength] = img[0 : ColumnOver, 0 : 512 - RepetitiveLength]
                #  后来修改的
                result[shape[0] - ColumnOver - RepetitiveLength: shape[0], 0: 256 - RepetitiveLength] = img[256 - ColumnOver - RepetitiveLength: 256, 0: 256 - RepetitiveLength]
            else:
                result[j * (256 - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                       0:256-RepetitiveLength] = img[RepetitiveLength: 256 - RepetitiveLength, 0: 256 - RepetitiveLength]
        #  最右侧一列特殊考虑，右边的边缘要拼接进去
        elif(i % len(TifArray[0]) == len(TifArray[0]) - 1):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0: 256 - RepetitiveLength, shape[1] - RowOver: shape[1]] = img[0: 256 - RepetitiveLength, 256 - RowOver: 256]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver: shape[0], shape[1] - RowOver: shape[1]] = img[256 - ColumnOver: 256, 256 - RowOver: 256]
            else:
                result[j * (256 - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                       shape[1] - RowOver: shape[1]] = img[RepetitiveLength: 256 - RepetitiveLength, 256 - RowOver: 256]
            #  走完每一行的最右侧，行数+1
            j = j + 1
        #  不是最左侧也不是最右侧的情况
        else:
            #  第一行的要特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0: 256 - RepetitiveLength,
                       (i - j * len(TifArray[0])) * (256 - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(TifArray[0]) + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength
                       ] = img[0: 256 - RepetitiveLength, RepetitiveLength: 256 - RepetitiveLength]
            #  最后一行的要特殊考虑，下边的边缘要考虑进去
            if(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver: shape[0],
                       (i - j * len(TifArray[0])) * (256 - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(TifArray[0]) + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength
                       ] = img[256 - ColumnOver: 256, RepetitiveLength: 256 - RepetitiveLength]
            else:
                result[j * (256 - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                       (i - j * len(TifArray[0])) * (256 - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(TifArray[0]) + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                       ] = img[RepetitiveLength: 256 - RepetitiveLength, RepetitiveLength: 256 - RepetitiveLength]
    return result


def get_preprocessing(mean_file, std_file, img):
    """mean_file: 存储mean值的txt文件; std_file:存储std值的txt文件"""
    with open(mean_file, 'r', encoding='utf-8') as f:
        imgset_mean = np.array([line.strip('\n') for line in f]).astype('float32')
    with open(std_file, 'r', encoding='utf-8') as f:
        imgset_std = np.array([line.strip('\n') for line in f]).astype('float32')
    img_shape = img.shape
    band_num = img_shape[0]
    img_new = np.zeros(img_shape)
    for band in range(band_num):
        img_new[band, ] = (img[band, ] - imgset_mean[band]) / imgset_std[band]
    img = img_new.astype('float32')

    return img


if __name__ == '__main__':

    work_dir = "/data/study_Irrigation/segmentation_models.pytorch/Code_my"

    model_paths = ["/data/study_Irrigation/segmentation_models.pytorch/hrnet_3d_w18_Ca/118.pth"]

    config_file = os.path.join(work_dir, 'seg_hrnet_w18_train_seghrnet_3d.yaml')
    update_config(config, config_file)

    DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    inform_dir = "/data/study_Irrigation/Ca_256"
    mean_file = os.path.join(inform_dir, 'imgset_mean.txt')
    std_file = os.path.join(inform_dir, 'imgset_std.txt')

    block_w = 256
    block_h = 256
    area_perc = 0.5
    rs_dir = "/data/study_Irrigation/Ca_img_val"
    predict_dir = "/data/study_Irrigation/segmentation_models.pytorch/hrnet_3d_w18_Ca_predict_val"
    id_num_train = 152
    exclusion_list = [152]
    Prefix = "Ca_val"
    test_log = []

    for img_id in range(1, id_num_train + 1):
        if img_id in exclusion_list:
            continue
        else:
            if img_id < 10:
                str_name = '00{}'.format(img_id)
            elif 10 <= img_id < 100:
                str_name = '0{}'.format(img_id)
            else:
                str_name = str(img_id)

            img_name_7 = Prefix + "_7" + "_" + str_name + ".tif"
            img_name_8 = Prefix + "_8" + "_" + str_name + ".tif"
            img_name_9 = Prefix + "_9" + "_" + str_name + ".tif"
            img_name_10 = Prefix + "_10" + "_" + str_name + ".tif"
            TifPath_7 = os.path.join(rs_dir, img_name_7)

            if os.path.exists(TifPath_7) == False:
                continue
            TifPath_8 = os.path.join(rs_dir, img_name_8)
            TifPath_9 = os.path.join(rs_dir, img_name_9)
            TifPath_10 = os.path.join(rs_dir, img_name_10)

            img_name_predict = Prefix + "_" + str_name + ".tif"
            ResultPath = os.path.join(predict_dir, img_name_predict)

            print("开始预测..." + img_name_predict)
            test_log.append("开始预测..." + img_name_predict)
            #  获取当前时间
            start_time = datetime.datetime.now()

            img_7, src_meta = ReadRaster(TifPath_7)
            if img_7.shape == (4, 256, 256):
                continue
            # print('img_7.shape', img_7.shape)
            img_8, _ = ReadRaster(TifPath_8)
            # print('img_8.shape', img_8.shape)
            img_9, _ = ReadRaster(TifPath_9)
            # print('img_9.shape', img_9.shape)
            img_10, _ = ReadRaster(TifPath_10)
            # print('img_10.shape', img_10.shape)

            TifArray_7, RowOver, ColumnOver = TifCroppingArray(img_7, block_w, block_h, area_perc)
            TifArray_8, _, _ = TifCroppingArray(img_8, block_w, block_h, area_perc)
            TifArray_9, _, _ = TifCroppingArray(img_9, block_w, block_h, area_perc)
            TifArray_10, _, _ = TifCroppingArray(img_10, block_w, block_h, area_perc)

            img_8, img_9, img_10 = None, None, None

            predicts = []
            for i in range(len(TifArray_7)):
                for j in range(len(TifArray_7[0])):
                    image_7 = TifArray_7[i][j]
                    # print('image_7.shape', image_7.shape)
                    image_8 = TifArray_8[i][j]
                    # print('image_8.shape', image_8.shape)
                    image_9 = TifArray_9[i][j]
                    # print('image_9.shape', image_9.shape)
                    image_10 = TifArray_10[i][j]
                    # print('image_10.shape', image_10.shape)

                    image_stack = np.stack([image_7, image_8, image_9, image_10], axis=1)
                    image_stack = get_preprocessing(mean_file, std_file, image_stack)
                    image_stack = torch.from_numpy(image_stack).to(DEVICE).unsqueeze(0)
                    pred = np.zeros((1, block_w, block_h))

                    "build model"
                    model = eval('models.' + config.MODEL.NAME + '.get_seg_model')(config)
                    model.to(DEVICE)

                    for model_path in model_paths:
                        model.load_state_dict(torch.load(model_path))
                        model.eval()

                        with torch.no_grad():
                            pred = model(image_stack).cpu().numpy().squeeze(0)
                            print('pred.shape', pred.shape)

                    pred[pred >= 0.5] = 1
                    pred[pred < 0.5] = 0
                    pred = pred.astype(np.uint8)
                    # pred = pred.reshape((block_w, block_h))
                    pred = pred.squeeze()
                    predicts.append((pred))

            TifArray_8, TifArray_9, TifArray_10 = None, None, None

            result_shape = (img_7.shape[1], img_7.shape[2])
            result_data = Result(result_shape, TifArray_7, predicts, block_w, area_perc, RowOver, ColumnOver)
            src_meta.update({'count': 1, 'dtype': rasterio.uint8, 'compress': 'lzw'})
            with rasterio.open(ResultPath, 'w', **src_meta) as dst:
                dst.write(result_data, 1)

            end_time = datetime.datetime.now()
            print("模型预测完毕,耗时: " + str((end_time - start_time).seconds) + "s")
            test_log.append("耗时: " + str((end_time - start_time).seconds) + "s")

    time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
    with open(os.path.join(predict_dir, 'timelog_%s.txt' % time), 'w') as f:
        for i in range(len(test_log)):
            f.write(str(test_log[i]))
            f.write("\r\n")
