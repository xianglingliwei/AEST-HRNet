# coding=utf-8

import numpy as np


class binary_Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.data_TP = 0
        self.data_TN = 0
        self.data_FP = 0
        self.data_FN = 0

    def Pixel_Accuracy(self):
        Acc = (self.data_TN + self.data_TP) / (self.data_TP + self.data_TN + self.data_FP + self.data_FN)
        return Acc

    def Pixel_Recall(self):
        Recall = (self.data_TP) / (self.data_TP + self.data_FN)
        return Recall

    def Pixel_Precision(self):
        Precision = (self.data_TP) / (self.data_TP + self.data_FP)
        return Precision

    def Pixel_F1score(self):
        r = self.Pixel_Recall()
        p = self.Pixel_Precision()
        F1 = 2 * r * p / (r + p)
        return F1

    def Pixel_IoU(self):
        IoU = self.data_TP / (self.data_TP + self.data_FP + self.data_FN)
        return IoU

    def Pixel_Kappa(self):
        po = self.Pixel_Accuracy()
        z = self.data_TP + self.data_TN + self.data_FP + self.data_FN
        pe = ((self.data_TP + self.data_FN) / z) * ((self.data_TP + self.data_FP) / z) + (
                    (self.data_TN + self.data_FP) / z) * ((self.data_TN + self.data_FN) / z)
        kappa = (po - pe) / (1 - pe)
        return kappa

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        TP = np.sum((pre_image == 1) & (gt_image == 1))
        TN = np.sum((pre_image == 0) & (gt_image == 0))
        FN = np.sum((pre_image == 0) & (gt_image == 1))
        FP = np.sum((pre_image == 1) & (gt_image == 0))

        self.data_TP += TP
        self.data_TN += TN
        self.data_FP += FP
        self.data_FN += FN

    def reset(self):
        self.data_TP = 0
        self.data_TN = 0
        self.data_FP = 0
        self.data_FN = 0

    def evaluate_single(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        TP = np.sum((pre_image == 1) & (gt_image == 1))
        TN = np.sum((pre_image == 0) & (gt_image == 0))
        FN = np.sum((pre_image == 0) & (gt_image == 1))
        FP = np.sum((pre_image == 1) & (gt_image == 0))

        metricdict = {}
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        IOU = TP / (TP + FP + FN)
        po = acc
        z = TP + TN + FP + FN
        pe = ((TP + FN) / z) * ((TP + FP) / z) + ((TN + FP) / z) * ((TN + FN) / z)
        kappa = (po - pe) / (1 - pe)

        metricdict['p'] = p
        metricdict['r'] = r
        metricdict['F1'] = F1
        metricdict['acc'] = acc
        metricdict['IOU'] = IOU
        metricdict['kappa'] = kappa

        return metricdict

