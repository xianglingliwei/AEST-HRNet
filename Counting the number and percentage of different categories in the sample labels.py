import os
import gdal
import numpy as np
import glob


"Ca"
"train"
# mask_dir = '/data/study_Irrigation/Ca_256_train/labels'
# save_dir = '/data/study_Irrigation/Ca_256_train'

"val"
mask_dir = '/data/study_Irrigation/Ca_256_val/labels'
save_dir = '/data/study_Irrigation/Ca_256_val'


"初始化每个类的数目"
background_num = 0
Irrigated_num = 0
# CenterPivot_num = 0
# Sprinkler_num = 0
# Wheel_Line_num = 0
# Drip_num = 0
# MicroSprinkler_num = 0
# Flood_num = 0
# Rill_num = 0
# BigGun_num = 0

"返回目标目录中包含所有后缀名为 .tif 的文件的列表"
# label_paths = glob.glob(r'D:\MyStudy_IrrigateLand\Samples\E5_new\labelsets\*.tif')
label_paths = glob.glob(os.path.join(mask_dir, '*.tif'))
# glob模块用法参考：https://zhuanlan.zhihu.com/p/71861602， https://rgb-24bit.github.io/blog/2018/glob.html
# glob.glob('*.org')  # 返回包含所有后缀名为 .org 的文件的列表
# glob.iglob('*/')  # 返回匹配所有目录的迭代器
print(len(label_paths))

"读取所有图像文件并且统计所有图像中的不同类别的数量"
for label_path in label_paths:
    label = gdal.Open(label_path).ReadAsArray(0, 0, 256, 256)
    background_num += np.sum(label == 0)
    Irrigated_num += np.sum(label == 1)
    # CenterPivot_num+= np.sum(label == 1)
    # Sprinkler_num  += np.sum(label == 2)
    # Wheel_Line_num += np.sum(label == 3)
    # Drip_num += np.sum(label == 4)
    # MicroSprinkler_num += np.sum(label == 5)
    # Flood_num += np.sum(label == 6)
    # Rill_num += np.sum(label == 7)
    # BigGun_num  += np.sum(label == 8)

print('background_num', background_num)
print('Irrigated_num', Irrigated_num)

"计算不同类别的占比"
# classes = ('Non-Irrigated', 'CenterPivot', 'Sprinkler', 'Wheel_Line', 'Drip', 'MicroSprinkler', 'Flood', 'Rill', 'BigGun')
classes = ('Non-Irrigated', 'Irrigated')
# numbers = [background_num, CenterPivot_num, Sprinkler_num, Wheel_Line_num,  Drip_num, MicroSprinkler_num, Flood_num,
#            Rill_num, BigGun_num]
numbers = [background_num, Irrigated_num]
ClassValue_Proportion = numbers / (sum(numbers))
print('ClassValue_Proportion', ClassValue_Proportion)

"保存到本地文件,不同类别的数目和不同类别的占比"
with open(os.path.join(save_dir, 'mask_class_number_ratio.txt'), 'w') as f:
    f.writelines(line.astype('str') + '\n' for line in numbers)
    f.writelines(line.astype('str') + '\n' for line in ClassValue_Proportion)

