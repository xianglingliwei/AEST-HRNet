# coding=utf-8


import os
import sys
from tqdm import tqdm
import xlwt
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from Datapre import Dataset_all
from Evaluator import binary_Evaluator
import _init_paths
import models
from config import config
from config import update_config


parent_dir = "/data/study_Irrigation/Wa_256"
imgs_dir = os.path.join(parent_dir, "imgs")
labels_dir = os.path.join(parent_dir, "labels")
test_txt = os.path.join(parent_dir, 'test.txt')
inform_dir = parent_dir

save_path = "/data/study_Irrigation/segmentation_models.pytorch/hrnet_3d_w18_Wa"
model_path = '/data/study_Irrigation/segmentation_models.pytorch/hrnet_3d_w18_Wa/127.pth'


save_dir = '/data/study_Irrigation/segmentation_models.pytorch/hrnet_3d_w18_Wa_predict_txt'
work_dir = "/data/study_Irrigation/segmentation_models.pytorch/Code_my"

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
CLASSES = ['irr']

config_file = os.path.join(work_dir, 'seg_hrnet_w18_train_seghrnet_3d.yaml')
update_config(config, config_file)

"cudnn related setting"
cudnn.benchmark = config.CUDNN.BENCHMARK
cudnn.deterministic = config.CUDNN.DETERMINISTIC
cudnn.enabled = config.CUDNN.ENABLED

"build model"
model = eval('models.'+config.MODEL.NAME + '.get_seg_model')(config)

infer_dataset = Dataset_all(imgs_dir, labels_dir, test_txt, classes=CLASSES, preprocessing=True, augmentation=False,
                            inform_dir=inform_dir)

model.load_state_dict(torch.load(model_path))
model.to(DEVICE)
model.eval()

"for model inferring and save results"
evaluator = binary_Evaluator(1)

with torch.no_grad():
    for i in tqdm(range(len(infer_dataset))):
        image, gt_mask = infer_dataset[i]
        gt_mask = gt_mask.squeeze().astype(np.uint8)
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = model(x_tensor).squeeze().cpu().numpy()
        pr_mask[pr_mask >= 0.5] = 1
        pr_mask[pr_mask < 0.5] = 0
        pr_mask = pr_mask.astype(np.uint8)
        evaluator.add_batch(gt_mask, pr_mask)

P = evaluator.Pixel_Precision()
R = evaluator.Pixel_Recall()
F1 = evaluator.Pixel_F1score()
IoU = evaluator.Pixel_IoU()
OA = evaluator.Pixel_Accuracy()
K = evaluator.Pixel_Kappa()

print('P:{} R:{} F1:{} IOU:{} OA:{} K:{}'.format(P, R, F1, IoU, OA, K))

book = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book.add_sheet('test', cell_overwrite_ok=True)
# print(test_loss, test_score)
sheet.write(0, 0, 'Precision')
sheet.write(1, 0, P)
sheet.write(0, 1, 'Recall')
sheet.write(1, 1, R)
sheet.write(0, 2, 'F1score')
sheet.write(1, 2, F1)
sheet.write(0, 3, 'IoU')
sheet.write(1, 3, IoU)
sheet.write(0, 4, 'OA')
sheet.write(1, 4, OA)
sheet.write(0, 5, 'Kappa')
sheet.write(1, 5, K)
book.save(os.path.join(save_dir, 'binary_Evaluator_drop-attention.xls'))






