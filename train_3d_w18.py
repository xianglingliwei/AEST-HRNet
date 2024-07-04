
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from torchkeras import summary
import _init_paths
import models
from config import config
from config import update_config
import segmentation_models_pytorch as smp

from Datapre import Dataset_all


def plot_metric(dfhistory, metric, save_path):
    train_metrics = dfhistory["train_"+metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.savefig(os.path.join(save_path, 'train_{}.jpg').format(metric), dpi=100)
    plt.close()


def update_lr_geometric_decline(old_lr, factor=0.9):
    new_lr = old_lr * factor
    return new_lr


def update_lr_standard(old_lr, epoch_now, total_epochs, power=0.9):
    new_lr = old_lr * (1 - float(epoch_now) / (total_epochs + 1)) ** power
    return new_lr


parent_dir = "/data/study_Irrigation/Wa_256"
imgs_dir = os.path.join(parent_dir, "imgs")
labels_dir = os.path.join(parent_dir, "labels")
train_txt = os.path.join(parent_dir, 'train.txt')
test_txt = os.path.join(parent_dir, 'test.txt')

inform_dir = parent_dir

save_path = "/data/study_Irrigation/segmentation_models.pytorch/hrnet_3d_w18_Wa"

work_dir = "/data/study_Irrigation/segmentation_models.pytorch/Code_my"

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

CLASSES = ['irr']

config_file = os.path.join(work_dir, 'seg_hrnet_w18_train_seghrnet_3d.yaml')
update_config(config, config_file)


"cudnn related setting"
cudnn.benchmark = config.CUDNN.BENCHMARK
cudnn.deterministic = config.CUDNN.DETERMINISTIC
cudnn.enabled = config.CUDNN.ENABLED


"build model"
model = eval('models.'+config.MODEL.NAME + '.get_seg_model')(config)

train_dataset = Dataset_all(imgs_dir, labels_dir, train_txt, classes=CLASSES, preprocessing=True, augmentation=True,
                            inform_dir=inform_dir)

valid_dataset = Dataset_all(imgs_dir, labels_dir, test_txt, classes=CLASSES, preprocessing=True, augmentation=False,
                            inform_dir=inform_dir)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=8)

# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
#loss = smp.utils.losses.DiceLoss()
loss = smp.utils.losses.BCELoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0005),
])

# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(model, loss=loss, metrics=metrics, optimizer=optimizer,
                                         device=DEVICE, verbose=True,)

valid_epoch = smp.utils.train.ValidEpoch(model, loss=loss, metrics=metrics, device=DEVICE, verbose=True,)

trainlog_path = os.path.join(save_path, 'train.log')
metriclog_path = os.path.join(save_path, 'train_metric.log')
trainlog = open(trainlog_path, 'w')
metriclog = open(metriclog_path, 'w')
# score = "IoU"
trainlog.write(
    "train_loss" + "  " + "train_score" + "  " + "val_loss" + "  " + "val_score" + "  " + "epoch" + "\n")
metriclog.write(
    "train_loss" + "  " + "train_score" + "  " + "val_loss" + "  " + "val_score" + "  " + "epoch" + "\n")
trainlog.flush()
metriclog.flush()

history = {}

train_epoch_best_loss = 100
val_epoch_best_score = 0
no_optim = 0
epochs = 130

for epoch in range(1, epochs+1):
    print('\nEpoch: {}'.format(epoch))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    # do something (save model, change lr, etc.)
    train_score, val_score = train_logs['iou_score'], valid_logs['iou_score']
    train_loss, val_loss = train_logs['bce_loss'], valid_logs['bce_loss']
    trainlog.write(str(train_loss) + "  " + str(train_score) + "  " + str(val_loss) + "  " + str(val_score)
                    + "  " + str(epoch) + '\n')
    metriclog.write(str(train_loss) + "  " + str(train_score) + "  " + str(val_loss) + "  " + str(val_score)
                    + "  " + str(epoch) + '\n')
    lr_epoch = optimizer.param_groups[0]['lr']
    print('<<<current learn rate:>>>', lr_epoch)
    trainlog.write("current learn rate:" + "  " + str(lr_epoch) + '\n')
    trainlog.flush()
    metriclog.flush()

    history['train_loss'] = history.get('train_loss', []) + [train_loss]
    history['train_score'] = history.get('train_score', []) + [train_score]
    history['val_loss'] = history.get('val_loss', []) + [val_loss]
    history['val_score'] = history.get('val_score', []) + [val_score]

    monitor = "val_score"
    arr_scores = history[monitor]
    best_score_idx = np.argmax(arr_scores)
    if best_score_idx == len(arr_scores) - 1:
        torch.save(model.state_dict(), os.path.join(save_path, "{}.pth").format(epoch))
        print("<< reach best {0}: {1},save model in {2} epoch>>".format(monitor, arr_scores[best_score_idx], epoch),
              file=sys.stderr)
        trainlog.write("<< reach best {0}: {1},save model in {2} epoch>>".format(monitor, arr_scores[best_score_idx],
                                                                                 epoch) + '\n')
        trainlog.flush()
    patience_score = 30
    if len(arr_scores) - best_score_idx > patience_score:
        print("<< {0} without improvement after {1} epochs, early stopping in {2} epoch >>".
              format(monitor, patience_score, epoch), file=sys.stderr)
        trainlog.write("{0} without improvement after {1} epochs, early stopping in {2} epoch"
                       .format(monitor, patience_score, epoch) + '\n')
        trainlog.flush()
        break

    patience_loss = 3
    if train_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_loss  # 保留当前epoch的loss为train_epoch_best_loss
        print("<< reach best loss: {0}, in {1} epoch>>".format(train_epoch_best_loss, epoch), file=sys.stderr)
        trainlog.write("<< reach best loss: {0}, in {1} epoch >>".format(train_epoch_best_loss, epoch) + '\n')
        trainlog.flush()
    if no_optim > patience_loss:
        if optimizer.param_groups[0]['lr'] < 1e-6:
            break
        else:
            old_lr = optimizer.param_groups[0]['lr']
            new_lr = update_lr_geometric_decline(old_lr, factor=0.9)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            print('update learning rate: %f -> %f' % (old_lr, new_lr))
            trainlog.write('update learning rate: %f -> %f' % (old_lr, new_lr) + '\n')
            trainlog.flush()
            no_optim = 0

    # patience_loss = 3
    # if train_loss >= train_epoch_best_loss:
    #     no_optim += 1
    # else:
    #     no_optim = 0
    #     train_epoch_best_loss = train_loss
    #     print("<< reach best loss: {0}, in {1} epoch>>".format(train_epoch_best_loss, epoch), file=sys.stderr)
    # if no_optim > patience_loss:
    #     if optimizer.param_groups[0]['lr'] < 1e-6:
    #         break
    # old_lr = optimizer.param_groups[0]['lr']
    # new_lr = update_lr_standard(old_lr, epoch_now=epoch, total_epochs=epochs, power=0.9)
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = new_lr
    # trainlog.write('update learning rate: %f -> %f' % (old_lr, new_lr) + '\n')
    # trainlog.flush()
    # print('update learning rate: %f -> %f' % (old_lr, new_lr))

trainlog.close()
metriclog.close()

train_DF = pd.DataFrame(history)
train_DF.to_csv(os.path.join(save_path, "train_log.csv"))
print('Training completed')
plot_metric(train_DF, "loss", save_path)
plot_metric(train_DF, "score", save_path)











