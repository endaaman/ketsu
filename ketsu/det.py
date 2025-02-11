import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt

from .detectors import SimpleHSVDetector


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

def main():
    # データの読み込み
    print("Loading training data...")
    train_images, train_conj_masks, train_vessel_masks = load_dataset('train')

    print("Loading validation data...")
    val_images, val_conj_masks, val_vessel_masks = load_dataset('val')

    # detector = AdaptiveHSVDetector()
    detector = SimpleHSVDetector()

    # 学習
    print("Starting optimization...")
    result = detector.optimize(
        train_images,
        train_conj_masks,
        train_vessel_masks,
        n_iter=50
    )

    best_params = {
        'h_min': int(result.x[0]),
        'h_max': int(result.x[1]),
        's_min': int(result.x[2]),
        's_max': int(result.x[3]),
        'v_min': int(result.x[4]),
        'v_max': int(result.x[5]),
    }

    # 検証データでの評価
    print("Evaluating on validation set...")
    val_ious = []
    for image, conj_mask, vessel_mask in tqdm(zip(val_images, val_conj_masks, val_vessel_masks)):
        pred_mask = detector.detect_vessels(image, conj_mask, best_params)
        iou = detector._compute_iou(pred_mask, vessel_mask)
        val_ious.append(iou)

    # 結果の表示
    print("\nResults:")
    print(f"Best parameters: {best_params}")
    print(f"Training IoU: {-result.fun:.4f}")
    print(f"Validation IoU: {np.mean(val_ious):.4f} ± {np.std(val_ious):.4f}")

    # オプション：最良と最悪のケースを可視化
    best_idx = np.argmax(val_ious)
    worst_idx = np.argmin(val_ious)

    p = [(best_idx, "Best"), (worst_idx, "Worst")]
    pred_masks = {}
    for idx, case in p:
        pred_mask = detector.detect_vessels(
            val_images[idx],
            val_conj_masks[idx],
            best_params,
        )
        pred_masks[case] = pred_mask
        print(f"\n{case} case IoU: {val_ious[idx]:.4f}")

    for i, (idx, case) in enumerate(p):
        plt.subplot(2, 2, i*2+1)
        plt.imshow(val_images[idx])
        plt.subplot(2, 2, i*2+2)
        plt.imshow(pred_masks[case])
    plt.show()

if __name__ == "__main__":
    main()
