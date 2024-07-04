import os
import numpy as np
import skimage.io
import rasterio
from tqdm import tqdm


class DataInform:
    """ To get statistical information about the dataset, such as mean, std, class distribution.
        The class is employed for tackle class imbalance.
    """

    def __init__(self, img_dir, mask_dir, save_dir, img_prefix, mask_prefix, mode, sample_num=207, classes_num=2,
                 normVal=1.10, label_weight_scale_factor=1):
        """
        Args:
        img_dir: 保存图像样本的路径
        mask_dir：保存对应label的路径
        save_dir：保存最后结果的路径
        我的图像命名形式 Ca_train_10_0.tif/Wa_train_10_0.tif和Ca_train_10_0.tif/Ca_val_10_0.tif
        我的分割掩膜命名形式 CaIrr_train_0.tif/WaIrr_train_0.tif 和 CaIrr_val_0.tif/WaIrr_val_0.tif
        img_prefix (str): 图像命名的前缀,如Ca/Wa.
        mask_prefix (str): 分割掩膜命名的前缀,如CaIrr/WaIrr.
        mode (str): 是训练样本还是验证样本,如train/val,这里其实只有train,因为我们只需要计算train样本.
        classes: number of classes in the dataset
        inform_data_file: location where cached file has to be stored
        normVal: normalization value, as defined in ERFNet paper
        """
        self.save_dir = save_dir
        self.sample_num = sample_num
        self.classes_num = classes_num
        self.normVal = normVal
        self.label_weight_scale_factor = label_weight_scale_factor

        # 7,8,9,10表示月份,我用的是这4个月的影像
        self.img_list_7 = [mode + '_' + '7' + '_' + '{}.tif'.format(img_id) for img_id in range(sample_num)]
        self.img_list_8 = [mode + '_' + '8' + '_' + '{}.tif'.format(img_id) for img_id in range(sample_num)]
        self.img_list_9 = [mode + '_' + '9' + '_' + '{}.tif'.format(img_id) for img_id in range(sample_num)]
        self.img_list_10 = [mode + '_' + '10' + '_' + '{}.tif'.format(img_id) for img_id in range(sample_num)]
        self.masks_list = [mode + '_' + '{}.tif'.format(img_id) for img_id in range(sample_num)]

        if img_prefix:
            self.images_fps_7 = [os.path.join(img_dir, img_prefix + '_' + img_id) for img_id in self.img_list_7]
            self.images_fps_8 = [os.path.join(img_dir, img_prefix + '_' + img_id) for img_id in self.img_list_8]
            self.images_fps_9 = [os.path.join(img_dir, img_prefix + '_' + img_id) for img_id in self.img_list_9]
            self.images_fps_10 = [os.path.join(img_dir, img_prefix + '_' + img_id) for img_id in self.img_list_10]
        else:
            self.images_fps_7 = [os.path.join(img_dir, img_id) for img_id in self.img_list_7]
            self.images_fps_8 = [os.path.join(img_dir, img_id) for img_id in self.img_list_8]
            self.images_fps_9 = [os.path.join(img_dir, img_id) for img_id in self.img_list_9]
            self.images_fps_10 = [os.path.join(img_dir, img_id) for img_id in self.img_list_10]

        if mask_prefix:
            self.masks_fps = [os.path.join(mask_dir, mask_prefix + '_' + img_id) for img_id in self.masks_list]
        else:
            self.masks_fps = [os.path.join(mask_dir, img_id) for img_id in self.masks_list]

    def maskset_sta(self):
        maskset_hist = np.zeros(self.classes_num)  

        for i in tqdm(self.sample_num):
            mask = skimage.io.imread(self.masks_fps[i], as_gray=True)
            unique_values, unique_counts = np.unique(mask, return_counts=True)
            max_value, min_value = max(unique_values), min(unique_values)

            if max_value > (self.classes_num - 1) or min_value < 0:
                print('Labels can take value between 0 and number of classes.')
                print('Some problem with labels. Please check. label_set:', unique_values)
                print('Label Image ID: ' + self.masks_list[i])
                continue
            else:
                maskset_hist += unique_counts   

        # compute the class imbalance information
        class_ratio = maskset_hist / np.sum(maskset_hist)
        class_weights = 1 / (np.log(self.normVal + class_ratio))
        # classWeights = 1 / (class_ratio + 0.01)
        class_weights = np.power(class_weights, self.label_weight_scale_factor)

        with open(os.path.join(self.save_dir, 'class_weights.txt'), 'w') as f:
            f.writelines(line + '\n' for line in class_weights.astype('str'))
        # 为 np.array
        return class_weights

    def imgset_sta(self):
        with rasterio.open(self.images_fps_7[0]) as src:
            img_7 = src.read()
            img_shape = img_7.shape
        band_num = img_shape[0]

        imgset_mean = np.zeros(band_num)
        imgset_std = np.zeros(band_num)

        img_num = 0
        for i in tqdm(range(self.sample_num)):
            with rasterio.open(self.masks_fps[i]) as src:
                mask = src.read(1)
            unique_values = np.unique(mask)
            max_value, min_value = max(unique_values), min(unique_values)

            if max_value > (self.classes_num - 1) or min_value < 0:
                print('Labels can take value between 0 and number of classes.')
                print('Some problem with labels. Please check. label_set:', unique_values)
                print('Label Image ID: ' + i)
                continue
            else:
                with rasterio.open(self.images_fps_7[i]) as src:
                    img_7 = src.read()
                with rasterio.open(self.images_fps_8[i]) as src:
                    img_8 = src.read()
                with rasterio.open(self.images_fps_9[i]) as src:
                    img_9 = src.read()
                with rasterio.open(self.images_fps_10[i]) as src:
                    img_10 = src.read()
                image_stack = np.stack([img_7, img_8, img_9, img_10], axis=1)
                for band in range(band_num):
                    img_band = image_stack[band, :, :, :]
                    imgset_mean[band] += np.mean(img_band)
                    imgset_std[band] += np.std(img_band)
                img_num += 1

        # divide the mean and std values by the sample space size
        imgset_mean /= img_num
        imgset_std /= img_num

        with open(os.path.join(self.save_dir, 'imgset_mean.txt'), 'w') as f:
            f.writelines(line + '\n' for line in imgset_mean.astype('str'))
        with open(os.path.join(self.save_dir, 'imgset_std.txt'), 'w') as f:
            f.writelines(line + '\n' for line in imgset_std.astype('str'))
        return imgset_mean, imgset_std


if __name__ == '__main__':

    img_dir = '/home/data/study_Irrigation/Ca_256_train/imgs'
    mask_dir = '/home/data/study_Irrigation/Ca_256_train/labels'
    save_dir = '/home/data/study_Irrigation/Ca_256_train'
    img_prefix = 'Ca'
    mask_prefix = 'CaIrr'
    mode = 'train'
    sample_num = 1068
    classes_num = 2

    Inform = DataInform(img_dir, mask_dir, save_dir, img_prefix, mask_prefix, mode, sample_num, classes_num,
                        normVal=1.10, label_weight_scale_factor=1)
    imgset_mean, imgset_std = Inform.imgset_sta()
    print('imgset_mean \n', imgset_mean)
    print('imgset_std \n', imgset_std)


