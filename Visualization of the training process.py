# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_metric(dfhistory, metric, savepath):
    train_metrics = dfhistory["train_"+metric].astype('float16')
    val_metrics = dfhistory['val_'+metric].astype('float16')
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation' + ' ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+ metric, 'val_'+ metric])
    plt.savefig(os.path.join(savepath, '{0}_{1}.jpg').format(metric, epochs), dpi=100)
    plt.close()


def structure_df(path):
    data = pd.read_table(path, sep='\t', header=None, names=['train_loss', 'train_score', 'val_loss', 'val_score', 'epoch'])
    for i in range(len(data)):
        coordinate = data['train_loss'][i].split()
        data['train_loss'][i] = coordinate[0]
        data['train_score'][i] = coordinate[1]
        data['val_loss'][i] = coordinate[2]
        data['val_score'][i] = coordinate[3]
        data['epoch'][i] = coordinate[4]

    return data


if __name__ == '__main__':
    metric_path = r"D:\MyStudy_IrrigateLand\segmentation_models\DeepLabV3Plus\train_metric_1.log"
    savepath = r"D:\MyStudy_IrrigateLand\segmentation_models\DeepLabV3Plus"
    train_hietory = structure_df(metric_path)

    plot_metric(train_hietory, "loss", savepath)
    plot_metric(train_hietory, "score", savepath)



