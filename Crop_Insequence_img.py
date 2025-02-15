# coding=utf-8

"""
参考了卷积算法的输出形状与输入形状的关系式
"""

import os
# import cv2
import rasterio
from rasterio import fill
from rasterio import windows
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm


def ReadRaster(raster):
    """
    读取遥感影像数据.  parameters——raster:遥感影像的路径
    return——image：遥感影像包含的array，src_meta：遥感影像的坐标
    """
    with rasterio.open(raster) as src:
        image = src.read()
        if image.shape[0] == 1:
            image = np.squeeze(image)
        src_meta = src.meta
    return image, src_meta


# def filldata(array):
#     "将影像中等于np.nan的值去掉，并插值填补"
#     "证明只适合于对异常值的处理，比如去除大于6000的值，不适合于对大面积缺失做插值"
#     # np.nan,np.NAN,np.NaN 都是 np.nan，表示Nan：Not a number，
#     # 参考：https://zhuanlan.zhihu.com/p/38712765
#     # mask = (array != np.nan) # 没用，判断array中是否有空值必须用np.isnan
#     mask = np.isnan(array)  # array中值为nan的为True，不为nan的为False
#     mask = (mask != True)  # 插值是用true的位置来插值，所以true的位置的值不能是nan，因而将前面的mask取反操作
#     # print(np.unique(mask))
#     result = fill.fillnodata(array, mask, smoothing_iterations=1)  # fillnodata只能处理二轴数组(单波段数据)
#     # result = fill.fillnodata(array, mask)
#     return result

def filldata(array):
    "将影像中大于6000的值去掉，并插值填补"
    mask = (array <= 6000)
    result = fill.fillnodata(array, mask)  # fillnodata只能处理二轴数组(单波段数据)
    return result


'''
滑动窗口裁剪函数
裁剪后保存目录需要自己在代码中改
rasterlist: 保存原始影像路径的列表
labellist： 保存原始label路径的列表，需要和rasterlist中一一对应
img_w 裁剪宽度 数组的列数
img_h 裁剪高度 数组的行数
RepetitionRate 重复率
imgsets_dir: 保存裁剪影像的路径
labelsets_dir ： 保存裁剪label的路径
'''

"只裁剪img"
def TifCrop(rasterlist, savedir, img_w, img_h, RepetitionRate, month):
    # 含有默认值或可选参数的参数不能放在其他参数的前面
    # 否则会报错： non-default argument follows default argument或positional argument follows keyword argument
    imgsets_dir = os.path.join(savedir, "imgs")  # 保存img样本集的目录
    if not os.path.exists(imgsets_dir):
        os.mkdir(imgsets_dir)
    g_count = 0
    for i in tqdm(range(len(rasterlist))):
        src_img, raster_meta = ReadRaster(rasterlist[i])
        src_img = src_img.astype(np.uint16)
        # Sentinel2 遥感影像 栅格像素值的数据类型是uint16，用arcgis按照shp范围裁剪之后会改变数据类型，在这里再改回为uint16
        # print(raster_meta)
        # band1, band2, band3 = filldata(raster_img[0]), filldata(raster_img[1]), filldata(raster_img[2])
        # mask掉数组中>6000的像素值，并插值填补
        # src_img = np.stack([band1, band2, band3], 0)
        bandNum, X_height, X_width = src_img.shape
        n_H = int((X_height - img_h) / (img_h * (1 - RepetitionRate))) + 1
        n_W = int((X_width - img_w) / (img_w * (1 - RepetitionRate))) + 1
        # 卷积的输出形状与输入形状的关系式  n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
        # n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
        for h in range(n_H):  # 遍历行
            for w in range(n_W):  # 遍历列
                vert_start = int(h * img_h * (1 - RepetitionRate))  # vert_start = h * stride
                vert_end = vert_start + img_h  # vert_end = vert_start + f
                horiz_start = int(w * img_w * (1 - RepetitionRate))  # horiz_start = w * stride
                horiz_end = horiz_start + img_w
                src_roi = src_img[:, vert_start: vert_end, horiz_start: horiz_end]
                # window = windows.Window(col_off=horiz_start, row_off=vert_start, width=img_w, height=img_h)
                window = Window.from_slices((vert_start, vert_end), (horiz_start, horiz_end))  # 与上面相同，只是这样更好理解
                # 写程序时，为了方便代码跨平台，尽量不要显式地使用分隔符。尽量使用os.path.sep，保证代码可以不经修改在其他平台run起来.
                with rasterio.open(
                        # 'E:/MyStudy/segmentation_models.pytorch-master/example_li/Crop_InSequence/imgs/{}.tif'.format(g_count),
                        imgsets_dir + os.path.sep + '{}'.format(month) + '_' + '{}.tif'.format(g_count),
                        'w',
                        driver='GTiff',
                        height=img_h,
                        width=img_w,
                        count=bandNum,
                        dtype=src_img.dtype,
                        crs=raster_meta['crs'],
                        transform=windows.transform(window, raster_meta['transform']),
                        # transform = raster.window_transform(window), # 使用数据集的window_transform 方法来访问窗口的仿射变换
                        compress='lzw',
                        # nodata,
                ) as dst:
                    dst.write(src_roi)
                g_count += 1

        #  向前裁剪最后一列
        if ((X_width - img_w) % (img_w * (1 - RepetitionRate))) != 0:
            for h in range(n_H):  # 遍历行
                vert_start = int(h * img_h * (1 - RepetitionRate))  # vert_start = h * stride
                vert_end = vert_start + img_h  # vert_end = vert_start + f
                src_roi = src_img[:, vert_start: vert_end, (X_width - img_w): X_width]
                window = Window.from_slices((vert_start, vert_end), ((X_width - img_w), X_width))
                # print(window)
                with rasterio.open(
                        # 'E:/MyStudy/segmentation_models.pytorch-master/example_li/Crop_InSequence/imgs/{}.tif'.format(g_count),
                        imgsets_dir + os.path.sep + '{}'.format(month) + '_' + '{}.tif'.format(g_count),
                        'w',
                        driver='GTiff',
                        height=img_h,
                        width=img_w,
                        count=bandNum,
                        dtype=src_img.dtype,
                        crs=raster_meta['crs'],
                        transform=windows.transform(window, raster_meta['transform']),
                        # transform = img.window_transform(window), # 使用数据集的window_transform 方法来访问窗口的仿射变换
                        compress='lzw',
                        # nodata,
                ) as dst:
                    dst.write(src_roi)
                g_count += 1

        # 向前裁剪最后一行
        if ((X_height - img_h) % (img_h * (1 - RepetitionRate))) != 0:
            for w in range(n_W):  # 遍历列
                horiz_start = int(w * img_w * (1 - RepetitionRate))  # vert_start = h * stride
                horiz_end = horiz_start + img_w  # vert_end = vert_start + f
                src_roi = src_img[:, (X_height - img_h): X_height, horiz_start: horiz_end]
                window = Window.from_slices(((X_height - img_h), X_height), (horiz_start, horiz_end))
                with rasterio.open(
                        # 'E:/MyStudy/segmentation_models.pytorch-master/example_li/Crop_InSequence/imgs/{}.tif'.format(g_count),
                        imgsets_dir + os.path.sep + '{}'.format(month) + '_' + '{}.tif'.format(g_count),
                        'w',
                        driver='GTiff',
                        height=img_h,
                        width=img_w,
                        count=bandNum,
                        dtype=src_img.dtype,
                        crs=raster_meta['crs'],
                        transform=windows.transform(window, raster_meta['transform']),
                        # transform=img.window_transform(window),  # 使用数据集的window_transform 方法来访问窗口的仿射变换
                        compress='lzw',
                        # nodata,
                ) as dst:
                    dst.write(src_roi)
                g_count += 1

        #  裁剪右下角
        if ((X_height - img_h) % (img_h * (1 - RepetitionRate))) != 0 and (
                (X_height - img_h) % (img_h * (1 - RepetitionRate))) != 0:
            src_roi = src_img[:, (X_height - img_h): X_height, (X_width - img_w): X_width]
            window = Window.from_slices(((X_height - img_h), X_height), ((X_width - img_w), X_width))
            with rasterio.open(
                    # 'E:/MyStudy/segmentation_models.pytorch-master/example_li/Crop_InSequence/imgs/{}.tif'.format(g_count),
                    imgsets_dir + os.path.sep + '{}'.format(month) + '_' + '{}.tif'.format(g_count),
                    'w',
                    driver='GTiff',
                    height=img_h,
                    width=img_w,
                    count=bandNum,
                    dtype=src_img.dtype,
                    crs=raster_meta['crs'],
                    transform=windows.transform(window, raster_meta['transform']),
                    # transform = img.window_transform(window),  # 使用数据集的window_transform 方法来访问窗口的仿射变换
                    compress='lzw',
                    # nodata,
            ) as dst:
                dst.write(src_roi)
            g_count += 1


if __name__ == '__main__':
    "Ca"
    "Ca train"
    "Ca train 中的影像所在目录"
    rs_dir_train = "/data/study_Irrigation/Ca_img_train"
    "Ca train 中的影像数目"
    id_num_train = 207  # 影像数目的总数
    "Ca train 中不用的影像的编号"
    exclusion_list_train = []  # 不要的影像的id号
    "Ca train 目录中的影像前缀"
    Prefix_train = "Ca_train"

    "Ca val"
    "Ca val 中的影像所在目录"
    rs_dir_val = "/data/study_Irrigation/Ca_img_val"
    "Ca val 中的影像数目"
    id_num_val = 152   # 影像数目的总数
    "Ca val 中不用的影像的编号"
    exclusion_list_val = [152]  # 不要的影像的id号  152尺寸为10980*10980,这里不做为验证数据
    "Ca val 目录中的影像前缀"
    Prefix_val = "Ca_val"

    "要裁剪的影像的月份"
    # month = 5
    # month = 6
    # month = 7
    # month = 8
    # month = 9
    month = 10

    "保存裁剪后的小尺寸影像的目录"
    save_dir = "/data/study_Irrigation/Ca_256"

    "Ca train"
    # 构建要裁剪的影像的路径列表
    rsPath_list_train = []
    # 这里没有使用最简单的os.listdir()而是用for img_id in range(1, id_num_train + 1),就是为了保证列表中获取的文件肯定是按顺序来的.
    for img_id in range(1, id_num_train + 1):
        if img_id in exclusion_list_train:
            continue
        else:
            if img_id < 10:
                str_name = '00{}'.format(img_id)
            elif 10 <= img_id < 100:
                str_name = '0{}'.format(img_id)
            else:
                str_name = str(img_id)
            img_name = Prefix_train + "_{}".format(month) + "_" + str_name + ".tif"
            img_path = os.path.join(rs_dir_train, img_name)
            rsPath_list_train.append(img_path)
            # list是有序列表，放的时候什么顺序，取出来就是什么顺序，因而labelPath_list和rsPath_list中的元素顺序完全一致
    print('rsPath_list_train', rsPath_list_train)
    print('len(rsPath_list_train)', len(rsPath_list_train))

    "Ca val"
    # 构建要裁剪的影像的路径列表
    rsPath_list_val = []
    for img_id in range(1, id_num_val + 1):
        if img_id in exclusion_list_val:
            continue
        else:
            if img_id < 10:
                str_name = '00{}'.format(img_id)
            elif 10 <= img_id < 100:
                str_name = '0{}'.format(img_id)
            else:
                str_name = str(img_id)
            img_name = Prefix_val + "_{}".format(month) + "_" + str_name + ".tif"
            img_path = os.path.join(rs_dir_val, img_name)
            rsPath_list_val.append(img_path)
            # list是有序列表，放的时候什么顺序，取出来就是什么顺序，因而labelPath_list和rsPath_list中的元素顺序完全一致
    print('rsPath_list_val', rsPath_list_val)
    print('len(rsPath_list_val)', len(rsPath_list_val))

    "合并rsPath_list_train 和 rsPath_list_val"
    rsPath_list = rsPath_list_train + rsPath_list_val
    print('rsPath_list', rsPath_list)
    print('len(rsPath_list)', len(rsPath_list))

    TifCrop(rsPath_list, save_dir, img_w=256, img_h=256, RepetitionRate=0, month=month)

