from glob import glob
import cv2
from PIL import Image
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_aug(augmentation, size, normalization=True):
    augs_train = [
        # A.Resize(width=size, height=size),
        A.RandomResizedCrop(size=(size, size),
                            scale=(0.6, 1.0),
                            ratio=(0.8, 1.25),
                            p=1.0,
                            interpolation=cv2.INTER_LINEAR,
                            mask_interpolation=cv2.INTER_NEAREST,
                            ),
        A.HorizontalFlip(p=0.5),
        # A.RandomRotate90(p=0.3), # Rorate90はあまり良くないかも

        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),

        # A.ElasticTransform(p=0.6), # これもいまいちかも
        A.GridDistortion(p=0.5),
    ]
    augs_val = [
        A.Resize(width=size, height=size),
    ]

    if augmentation:
        aa = augs_train
    else:
        aa = augs_val

    if normalization:
        aa += [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],)]
    aa += [ToTensorV2()]
    return A.Compose(aa)



BASE_DATA_DIR = 'data/'

COLOR_MAP = np.array([
    [  0,   0,   0,   0], #0 -> transparent -> background
    [255,   0,   0, 255], #1 -> red         -> iris
    [  0, 255,   0, 255], #2 -> green       -> vessel
    [  0,   0, 255, 255], #3 -> blue        -> conjunctiva
],dtype=np.uint8)

class ConjDataset(torch.utils.data.Dataset):

    def __init__(self, mode='train', size=512, augmentation=True, normalization=True):
        self.image_paths = sorted(glob(f'{BASE_DATA_DIR}/{mode}/image/*.png'))
        self.label_paths = sorted(glob(f'{BASE_DATA_DIR}/{mode}/label/*.png'))
        assert len(self.image_paths) > 0, 'Downloads dataset to data/'
        assert len(self.label_paths) > 0, 'Downloads dataset to data/'

        self.images = [Image.open(p).convert('RGB').copy() for p in self.image_paths]
        self.labels = [Image.open(p).copy() for p in self.label_paths]

        # self.transform = transforms.Compose([ transforms.ToTensor() ])
        self.albu = get_aug(augmentation, size=size, normalization=normalization)


    def __getitem__(self, idx):
        image = self.images[idx]
        image_arr = np.array(image)
        label = self.labels[idx]
        label_arr = np.array(label)

        label = np.zeros_like(label_arr[...,0], dtype= np.uint8)
        for i, j in [(0, 0), (1, 1), (2, 2), (3, 2)]:
            mask = np.all(label_arr == COLOR_MAP[i], axis=-1)
            label[mask] = j

        auged = self.albu(image=image_arr, mask=label)
        x = auged['image']
        t = auged['mask']
        return x, t.to(torch.int64)


    def __len__(self):
        return len(self.image_paths)

