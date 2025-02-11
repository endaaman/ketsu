import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt

from .detectors import SimpleHSVDetector, AdaptiveHSVDetector, CLAHEHSVDetector


def load_dataset(split='train'):
    assert split in ['train', 'val'], 'split must be "train" or "val"'
    df = pd.read_excel('./data/list.xlsx', index_col=0)
    df = df[df['with_vessel'] > 0]
    df = df[df['split'] == split]

    images = []
    conj_masks = []
    vessel_masks = []
    for i, row in df.iterrows():
        fn = row['file_name']
        img = np.array(Image.open(f'data/{split}/image/{fn}').convert('RGB'))
        mask = np.array(Image.open(f'data/{split}/label/{fn}'))
        blue_mask = cv2.inRange(mask, (0, 0, 255, 255), (0, 0, 255, 255))  # 結膜
        green_mask = cv2.inRange(mask, (0, 255, 0, 255), (0, 255, 0, 255))  # 血管

        conj_mask = cv2.bitwise_or(blue_mask, green_mask)

        conj_masks.append(conj_mask)  # 結膜全体のマスク
        vessel_masks.append(green_mask)  # 血管部分のマスク
        images.append(img)

    return images, conj_masks, vessel_masks


def evaluate_detector(detector, train_images, train_conj_masks, train_vessel_masks,
                     val_images, val_conj_masks, val_vessel_masks):
    # Optimize parameters
    best_params, train_iou = detector.optimize(
        train_images, train_conj_masks, train_vessel_masks, n_iter=50
    )

    # Evaluate on validation set
    val_iou_mean, val_iou_std = detector.evaluate(
        val_images, val_conj_masks, val_vessel_masks, best_params
    )

    # Print results
    print(f"\nResults:")
    print(f"Best parameters: {best_params}")
    print(f"Training IoU: {train_iou:.4f}")
    print(f"Validation IoU: {val_iou_mean:.4f} ± {val_iou_std:.4f}")
    return best_params, train_iou, val_iou_mean, val_iou_std


def main():
    train_images, train_conj_masks, train_vessel_masks = load_dataset('train')
    val_images, val_conj_masks, val_vessel_masks = load_dataset('val')

    # detector = SimpleHSVDetector()
    # detector = AdaptiveHSVDetector()
    detector = CLAHEHSVDetector()
    results = evaluate_detector(
        detector,
        train_images, train_conj_masks, train_vessel_masks,
        val_images, val_conj_masks, val_vessel_masks
    )


if __name__ == "__main__":
    main()
