# coding=utf-8

import os
import rasterio
import numpy as np
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import matplotlib.pyplot as plt


def ReadRaster(raster, src_meta=True):
    with rasterio.open(raster) as src:
        image = src.read()
        if image.shape[0] == 1:
            image = np.squeeze(image)
        if src_meta:
            src_meta = src.meta
        else:
            src_meta = None
    return image, src_meta


def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


class Dataset(BaseDataset):
    CLASSES = ['background', 'irr']

    def __init__(self, images_dir, masks_dir, img_prefix, mask_prefix, mode, sample_num=207, classes=None,
                 preprocessing=False, inform_dir=None):

        img_list_7 = [mode + '_' + '7' + '_' + '{}.tif'.format(img_id) for img_id in range(sample_num)]
        img_list_8 = [mode + '_' + '8' + '_' + '{}.tif'.format(img_id) for img_id in range(sample_num)]
        img_list_9 = [mode + '_' + '9' + '_' + '{}.tif'.format(img_id) for img_id in range(sample_num)]
        img_list_10 = [mode + '_' + '10' + '_' + '{}.tif'.format(img_id) for img_id in range(sample_num)]
        masks_list = [mode + '_' + '{}.tif'.format(img_id) for img_id in range(sample_num)]

        if img_prefix:
            self.images_fps_7 = [os.path.join(images_dir, img_prefix + '_' + img_id) for img_id in img_list_7]
            self.images_fps_8 = [os.path.join(images_dir, img_prefix + '_' + img_id) for img_id in img_list_8]
            self.images_fps_9 = [os.path.join(images_dir, img_prefix + '_' + img_id) for img_id in img_list_9]
            self.images_fps_10 = [os.path.join(images_dir, img_prefix + '_' + img_id) for img_id in img_list_10]
        else:
            self.images_fps_7 = [os.path.join(images_dir, img_id) for img_id in img_list_7]
            self.images_fps_8 = [os.path.join(images_dir, img_id) for img_id in img_list_8]
            self.images_fps_9 = [os.path.join(images_dir, img_id) for img_id in img_list_9]
            self.images_fps_10 = [os.path.join(images_dir, img_id) for img_id in img_list_10]

        if mask_prefix:
            self.masks_fps = [os.path.join(masks_dir, mask_prefix + '_' + img_id) for img_id in masks_list]
        else:
            self.masks_fps = [os.path.join(masks_dir, img_id) for img_id in masks_list]

        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.mode = mode
        self.preprocessing = preprocessing
        if self.preprocessing:
            self.inform_dir = inform_dir

    def __getitem__(self, i):
        with rasterio.open(self.images_fps_7[i]) as src:
            img_7 = src.read()
            img_shape = img_7.shape
        with rasterio.open(self.images_fps_8[i]) as src:
            img_8 = src.read()
        with rasterio.open(self.images_fps_9[i]) as src:
            img_9 = src.read()
        with rasterio.open(self.images_fps_10[i]) as src:
            img_10 = src.read()
        image_concat = np.concatenate([img_7, img_8, img_9, img_10], axis=0).transpose(1,2,0)
        img_shape = (img_shape[1], img_shape[2], img_shape[0])  # (height,width,band)

        with rasterio.open(self.masks_fps[i]) as src:
            mask = src.read(1)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=0).astype('uint8').transpose(1,2,0)

        # apply augmentations
        if self.mode == 'train':
            sample = get_training_augmentation()(image=image_concat, mask=mask)
            image_concat, mask = sample['image'], sample['mask']
        elif self.mode == 'val':
            image_concat, mask = image_concat, mask
        img_frame = np.zeros((4,) + img_shape, dtype="uint16")

        for f in range(4):
            img_frame[f, :, :, :] = np.expand_dims(image_concat[:, :, f * 4:f * 4 + 4], 0)
        image_concat = img_frame

        # apply preprocessing
        if self.preprocessing:
            mean_file = os.path.join(self.inform_dir, 'imgset_mean.txt')
            std_file = os.path.join(self.inform_dir, 'imgset_std.txt')
            image_concat, mask = get_preprocessing(mean_file, std_file, img=image_concat, mask=mask)

        return image_concat, mask

    def __len__(self):
        return len(self.images_fps_7)


class Dataset_all(BaseDataset):

    CLASSES = ['background', 'irr']

    def __init__(self, images_dir, masks_dir, txt_dir, classes=None, preprocessing=False, augmentation=False,
                inform_dir=None):

        with open(txt_dir, 'r') as f:
            img_list = f.readlines()

        img_list_7 = ['7' + '_' + img_id.strip() for img_id in img_list]
        img_list_8 = ['8' + '_' + img_id.strip() for img_id in img_list]
        img_list_9 = ['9' + '_' + img_id.strip() for img_id in img_list]
        img_list_10 = ['10' + '_' + img_id.strip() for img_id in img_list]
        masks_list = [img_id.strip() for img_id in img_list]

        self.images_fps_7 = [os.path.join(images_dir, img_id) for img_id in img_list_7]
        self.images_fps_8 = [os.path.join(images_dir, img_id) for img_id in img_list_8]
        self.images_fps_9 = [os.path.join(images_dir, img_id) for img_id in img_list_9]
        self.images_fps_10 = [os.path.join(images_dir, img_id) for img_id in img_list_10]

        self.masks_fps = [os.path.join(masks_dir, img_id) for img_id in masks_list]

        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        if self.preprocessing:
            self.inform_dir = inform_dir

    def __getitem__(self, i):
        with rasterio.open(self.images_fps_7[i]) as src:
            img_7 = src.read()
            img_shape = img_7.shape  # (band,height,width)
        with rasterio.open(self.images_fps_8[i]) as src:
            img_8 = src.read()
        with rasterio.open(self.images_fps_9[i]) as src:
            img_9 = src.read()
        with rasterio.open(self.images_fps_10[i]) as src:
            img_10 = src.read()

        image_concat = np.concatenate([img_7, img_8, img_9, img_10], axis=0).transpose(1, 2, 0)

        img_shape = (img_shape[1], img_shape[2], img_shape[0])  # (height,width,band)

        with rasterio.open(self.masks_fps[i]) as src:
            mask = src.read(1)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=0).astype('uint8').transpose(1, 2, 0)

        if self.augmentation:
            sample = get_training_augmentation()(image=image_concat, mask=mask)
            image_concat, mask = sample['image'], sample['mask']
        else:
            image_concat, mask = image_concat, mask

        img_frame = np.zeros((4,) + img_shape, dtype="uint16")

        for f in range(4):
            img_frame[f, :, :, :] = np.expand_dims(image_concat[:, :, f * 4:f * 4 + 4], 0)
        image_concat = img_frame

        # apply preprocessing
        if self.preprocessing:
            mean_file = os.path.join(self.inform_dir, 'imgset_mean.txt')
            std_file = os.path.join(self.inform_dir, 'imgset_std.txt')
            image_concat, mask = get_preprocessing(mean_file, std_file, img=image_concat, mask=mask)

        return image_concat, mask

    def __len__(self):
        return len(self.images_fps_7)


class Dataset_name(BaseDataset):
    CLASSES = ['background', 'irr_land']

    def __init__(
            self,
            images_dir,
            masks_dir,
            split_txt,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        with open(split_txt, 'r', encoding='utf-8') as f:
            img_list = [line.strip('\n') for line in f]
        self.ids = img_list
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in img_list]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in img_list]
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        imgname = self.ids[i]
        with rasterio.open(self.images_fps[i]) as src:
            image, image_meta = src.read(), src.meta
        image = image.transpose(1, 2, 0)

        with rasterio.open(self.masks_fps[i]) as src:
            mask, mask_meta = src.read(1), src.meta
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask, imgname, image_meta

    def __len__(self):
        return len(self.ids)


class Dataset_name_ImageDir(BaseDataset):
    CLASSES = ['background', 'irr_land']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocess=None,
            dataInform_dir=None
    ):

        img_list = [img for img in os.listdir(images_dir) if img.endswith('tif')]
        self.ids = img_list
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in img_list]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in img_list]
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocess = preprocess
        self.dataInform_dir = dataInform_dir

    def __getitem__(self, i):
        imgname = self.ids[i]
        with rasterio.open(self.images_fps[i]) as src:
            image, image_meta = src.read(), src.meta
        image = image.transpose(1, 2, 0)

        with rasterio.open(self.masks_fps[i]) as src:
            mask, mask_meta = src.read(1), src.meta
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocess:
            mean_file = os.path.join(self.dataInform_dir, 'imgset_mean.txt')
            std_file = os.path.join(self.dataInform_dir, 'imgset_std.txt')
            image, mask = self.preprocess(mean_file, std_file, image, mask)

        return image, mask, imgname, image_meta

    def __len__(self):
        return len(self.ids)


def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.Transpose(p=0.5),
        albu.RandomRotate90(p=0.5)
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    val_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.Transpose(p=0.5),
        albu.RandomRotate90(p=0.5)
    ]
    return albu.Compose(val_transform)


def to_tensor(img, label, **kwargs):
    return img.transpose(3, 0, 1, 2).astype('float32'), label.transpose(2, 0, 1).astype('float32')


def get_preprocessing(mean_file, std_file, img, mask):
    with open(mean_file, 'r', encoding='utf-8') as f:
        imgset_mean = np.array([line.strip('\n') for line in f]).astype('float32')
    with open(std_file, 'r', encoding='utf-8') as f:
        imgset_std = np.array([line.strip('\n') for line in f]).astype('float32')
    img, mask = to_tensor(img, mask)
    img_shape = img.shape
    band_num = img_shape[0]
    img_new = np.zeros(img_shape)
    for band in range(band_num):
        img_new[band, ] = (img[band, ] - imgset_mean[band]) / imgset_std[band]
    img = img_new.astype('float32')

    return img, mask


def toUint8_1(img_array):
    frameNum, width, height, bandNum = img_array.shape
    img_uint8 = np.zeros((frameNum, width, height, bandNum))
    for f in range(frameNum):
        for b in range(bandNum):
            img_band = img_array[f, :, :, b]
            img_uint8[f, :, :, b] = (img_band - img_band.min()) / (img_band.max() - img_band.min())
    img_uint8 = (255 * img_uint8).astype(np.uint8)
    return img_uint8


def toUint8_2(img_array, low_per=2, high_per=98):
    frames, rows, cols, bands = img_array.shape
    compress_data = np.zeros((frames, rows, cols, bands), dtype="uint8")

    for f in range(frames):
        for b in range(bands):
            data_band = img_array[f, :, :, b]
            cutmin = np.percentile(data_band, low_per)
            cutmax = np.percentile(data_band, high_per)
            data_band = np.clip(data_band, cutmin, cutmax)
            compress_data[f, :, :, b] = np.around((data_band - cutmin) * 255 / (cutmax - cutmin))
    return compress_data
