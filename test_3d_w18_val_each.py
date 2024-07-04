# coding=utf-8


import os
from tqdm import tqdm
import xlwt
import numpy as np
import rasterio
from Evaluator import binary_Evaluator


def ReadRaster(raster):
    with rasterio.open(raster) as src:
        image = src.read()
        if image.shape[0] == 1:
            image = np.squeeze(image)
        src_meta = src.meta
    return image, src_meta


"label"
label_path = r"E:\study_Irrigation\sampleRaster\Wa_label_val"
label_id_prefix = "WaIrr_"
"Predict"
Pr_path = r"D:\study_Irrigation_code\hrnet_3d_w18_Wa_predict_val"
save_dir = Pr_path

img_list = [img for img in os.listdir(Pr_path) if img.endswith('tif')]

evaluator = binary_Evaluator(1)

book = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book.add_sheet('test', cell_overwrite_ok=True)
sheet.write(0, 0, 'ID')
sheet.write(0, 1, 'Precision')
sheet.write(0, 2, 'Recall')
sheet.write(0, 3, 'F1score')
sheet.write(0, 4, 'IoU')
sheet.write(0, 5, 'OA')
sheet.write(0, 6, 'Kappa')
sheet.write(0, 7, 'shape_size')


i = 1
for P in tqdm(range(len(img_list))):
    print("第{}个图像".format(i))
    Pre_id = img_list[P]
    label_id = label_id_prefix + "val_" + Pre_id.split('_')[4]
    label_lmg, _ = ReadRaster(os.path.join(label_path, label_id))
    print("label的唯一值 \n", np.unique(label_lmg))
    with rasterio.open(os.path.join(Pr_path, Pre_id)) as src:
        Pr_lmg = np.squeeze(src.read())
    print("预测值的唯一值 \n", np.unique(Pr_lmg))
    pixels_size = Pr_lmg.shape[0]
    metricdict = evaluator.evaluate_single(label_lmg, Pr_lmg)
    P = metricdict['p']
    R = metricdict['r']
    F1 = metricdict['F1']
    OA = metricdict['acc']
    IoU = metricdict['IOU']
    K = metricdict['kappa']

    print('P:{} R:{} F1:{} IOU:{} OA:{} K:{}'.format(P, R, F1, IoU, OA, K))
    # 将binary_Evaluator结果保存到excel表
    sheet.write(i, 0, Pre_id.split('_')[4])
    sheet.write(i, 1, P)
    sheet.write(i, 2, R)
    sheet.write(i, 3, F1)
    sheet.write(i, 4, IoU)
    sheet.write(i, 5, OA)
    sheet.write(i, 6, K)
    sheet.write(i, 7, pixels_size)
    i = i + 1

book.save(os.path.join(save_dir, 'binary_Evaluator_each.xls'))






