import pandas as pd
from glob import glob
from tqdm import tqdm
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
        A.RandomToneCurve(scale=0.8, p=0.5),
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


COLOR_MAP = np.array([
    [  0,   0,   0,   0], #0 -> BG
    [255,   0,   0, 255], #1 -> red  : cornea
    [  0,   0, 255, 255], #2 -> blue : conjunctiva
    # [  0, 255,   0, 255], #3 -> green: vessel
],dtype=np.uint8)

class ConjDataset(torch.utils.data.Dataset):

    def __init__(self, fold, mode='train', size=512, augmentation=True, normalization=True):
        self.fold = fold
        df = pd.read_csv('./data/dataset_eye.csv')
        # Select items that have label mask
        if mode == 'train':
            target_mask = df['fold'] != fold
        else:
            target_mask = df['fold'] == fold
        df = df[target_mask]
        df = df[df['label'] > 0].copy()
        self.df = df

        self.images = []
        self.labels = []
        self.masks = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            id = row['test_ID']
            rl = row['R/L']
            fn = f'{str(id).zfill(4)}_{rl}_01.png'

            # image = Image.open(J('./data/conj/image/0001_L_01.png/'))
            image = Image.open(f'./data/conj/image/{fn}').convert('RGB')
            label = Image.open(f'./data/conj/label/{fn}').copy()
            mask = self.as_mask(np.array(label))
            self.images.append(image)
            self.labels.append(label)
            self.masks.append(mask)

        self.albu = get_aug(augmentation, size=size, normalization=normalization)

    def as_mask(self, label):
        label = np.array(label)
        mask = np.zeros(label.shape[:-1], dtype=np.uint8)
        for i in range(3):
            match = np.all(label == COLOR_MAP[i], axis=-1)
            mask[match] = i
        return mask

    def __getitem__(self, idx):
        image = np.array(self.images[idx])
        mask = self.masks[idx]

        auged = self.albu(image=image, mask=mask)
        x = auged['image']
        y = auged['mask']
        return x, y.to(torch.int64)


    def __len__(self):
        return len(self.images)

class SpotsDataset(torch.utils.data.Dataset):
    def __init__(self, fold, mode='train', size=512, augmentation=True, normalization=True):
        self.fold = fold
        df = pd.read_csv('./data/dataset_eye.csv')

        # Select items based on fold
        if mode == 'train':
            target_mask = df['fold'] != fold
        else:
            target_mask = df['fold'] == fold
        df = df[target_mask]

        # Exclude invalid labels (5) and select only rows with valid Majority_label
        df = df[df['Majority_label'].notna() & (df['Majority_label'] != 5)].copy()
        self.df = df

        self.images = []
        self.labels = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            id = row['test_ID']
            rl = row['R/L']
            fn = f'{str(id).zfill(4)}_{rl}_01.png'

            image = Image.open(f'./data/spots/{fn}').convert('RGB')
            label = int(row['Majority_label'])

            self.images.append(image)
            self.labels.append(label)

        self.albu = get_aug(augmentation, size=size, normalization=normalization)

    def __getitem__(self, idx):
        image = np.array(self.images[idx])
        label = self.labels[idx]

        auged = self.albu(image=image)
        x = auged['image']
        y = torch.tensor(label, dtype=torch.int64)
        return x, y

    def __len__(self):
        return len(self.images)

