import numpy as np
import cv2
from skopt import gp_minimize
from skopt.space import Real, Integer
from typing import Tuple, Dict
from matplotlib import pyplot as plt

from .utils import get_global_seed


class SimpleHSVDetector:
    def __init__(self):
        self.param_space = [
            Real(0, 5, name='h_min'),      # 0-10
            Real(6, 20, name='h_max'),      # 10
            Real(50, 200, name='s_min'),   # 120
            Real(201, 255, name='s_max'),   # 255
            Real(50, 200, name='v_min'),   # 120
            Real(201, 255, name='v_max'),   # 255
        ]

    def detect_vessels(self, rgb_image, conj_mask, params):
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

        # 単純に一つの範囲だけに
        lower = np.array([int(params['h_min']), int(params['s_min']), int(params['v_min'])], dtype=np.uint8)
        upper = np.array([int(params['h_max']), int(params['s_max']), int(params['v_max'])], dtype=np.uint8)

        pred_mask = cv2.inRange(hsv_image, lower, upper)
        pred_mask = cv2.bitwise_and(pred_mask, conj_mask)
        return pred_mask

    def optimize(self, images, conj_masks, vessel_masks, n_iter=50):
        def objective_function(params):
            params_dict = {
                'h_min': params[0],
                'h_max': params[1],
                's_min': params[2],
                's_max': params[3],
                'v_min': params[4],
                'v_max': params[5],
            }

            total_iou = 0
            for image, conj_mask, vessel_mask in zip(images, conj_masks, vessel_masks):
                pred_mask = self.detect_vessels(image, conj_mask, params_dict)
                total_iou += self._compute_iou(pred_mask, vessel_mask)

            return -total_iou / len(images)

        def progress_callback(res):
            print(f"Iter: {len(res.x_iters):3d}, Best IoU: {-res.fun:.4f}")

        result = gp_minimize(
            func=objective_function,
            dimensions=self.param_space,
            n_calls=n_iter,
            n_initial_points=10,
            callback=[progress_callback],
            noise=0.1,
            random_state=get_global_seed(),
        )
        return result


class AdaptiveHSVDetector:
    def __init__(self):
        self.param_space = [
            # HSV range parameters (existing)
            Real(0, 5, name='h_min'),
            Real(6, 20, name='h_max'),
            Real(50, 200, name='s_min'),
            Real(201, 255, name='s_max'),
            Real(50, 200, name='v_min'),
            Real(201, 255, name='v_max'),
            # Adaptive parameters (new)
            Integer(3, 21, name='block_size'),  # must be odd
            Real(0, 20, name='c_value')
        ]

    def detect_vessels(self, rgb_image, conj_mask, params):
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

        # 1. HSVマスクの作成（既存の処理）
        lower = np.array([int(params['h_min']), int(params['s_min']), int(params['v_min'])], dtype=np.uint8)
        upper = np.array([int(params['h_max']), int(params['s_max']), int(params['v_max'])], dtype=np.uint8)
        hsv_mask = cv2.inRange(hsv_image, lower, upper)

        # 2. 明度(V)チャンネルに対する適応的処理
        v_channel = hsv_image[:,:,2]
        block_size = int(params['block_size']) * 2 + 1  # ensure odd
        v_adaptive = cv2.adaptiveThreshold(
            v_channel,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            params['c_value']
        )

        # 3. マスクの組み合わせ
        combined_mask = cv2.bitwise_and(hsv_mask, v_adaptive)

        # 4. 結膜マスクの適用
        final_mask = cv2.bitwise_and(combined_mask, conj_mask)

        # 5. ノイズ除去
        kernel = np.ones((3,3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

        return final_mask


class VesselDetector:
    def __init__(self):
        # パラメータの探索空間を定義
        self.param_space = [
            Real(0.5, 2.0, name='hsv_h_scale'),      # HSVのH統計量からの乗数
            Real(0.5, 2.0, name='hsv_s_scale'),      # HSVのS統計量からの乗数
            Real(0.5, 2.0, name='hsv_v_scale'),      # HSVのV統計量からの乗数
            Integer(3, 99, name='block_size'),        # Adaptive Thresholdingのブロックサイズ
            Integer(-50, 50, name='c_value'),         # Adaptive ThresholdingのC値
            Real(0.0, 1.0, name='w_hsv'),            # HSVマスクの重み
            Real(0.0, 1.0, name='w_adaptive')        # Adaptiveマスクの重み
        ]

    def get_hsv_stats(self, image, mask):
        """角膜領域のHSV統計量を計算"""
        # 実装は省略
        return h_mean, s_mean, v_mean, h_std, s_std, v_std

    def create_hsv_mask(self, image, stats, params):
        """HSVベースのマスクを生成"""
        # 実装は省略
        return hsv_mask

    def create_adaptive_mask(self, image, params):
        """Adaptive Thresholdingベースのマスクを生成"""
        # 実装は省略
        return adaptive_mask

    def combine_masks(self, hsv_mask, adaptive_mask, params):
        """マスクの統合"""
        # 実装は省略
        return final_mask

    def evaluate_mask(self, pred_mask, true_mask):
        """マスクの評価（IoUなど）"""
        # 実装は省略
        return score

    def objective_function(self, params, image, true_mask, conj_mask):
        """Bayesian Optimizationの目的関数"""
        # HSV統計量の取得
        hsv_stats = self.get_hsv_stats(image, conj_mask)

        # 各マスクの生成
        hsv_mask = self.create_hsv_mask(image, hsv_stats, params[:3])
        adaptive_mask = self.create_adaptive_mask(image, params[3:5])

        # マスクの統合
        final_mask = self.combine_masks(hsv_mask, adaptive_mask, params[5:])

        # 評価
        score = self.evaluate_mask(final_mask, true_mask)
        return -score  # 最小化問題として解く

    def optimize(self, image, true_mask, conj_mask, n_iter=50):
        """パラメータの最適化を実行"""
        optimizer = BayesianOptimization(
            f=lambda *args: self.objective_function(*args, image, true_mask, conj_mask),
            dimensions=self.param_space,
            n_calls=n_iter,
            n_initial_points=10
        )
        return optimizer.res


class AdaptiveHSVDetector:
    def __init__(self):
        # パラメータの探索空間を定義
        self.param_space = [
            Integer(3, 99, name='block_size'),        # ブロックサイズ
            Real(0.0, 1.0, name='h_weight'),         # 色相の重み
            Real(0.0, 1.0, name='s_weight'),         # 彩度の重み
            Real(0.0, 1.0, name='v_weight'),         # 明度の重み
            Real(0.5, 2.0, name='local_scale'),      # 局所統計量のスケール係数
            Integer(-50, 50, name='c_value')         # ベースとなる閾値調整値
        ]

    def detect_vessels(self, rgb_image: np.ndarray, params: Dict, conj_mask: np.ndarray) -> np.ndarray:
        """
        RGB画像から血管を検出

        Args:
            rgb_image: RGB形式の入力画像
            params: 検出パラメータ
            conj_mask: 角膜領域のマスク
        Returns:
            検出された血管のマスク画像
        """
        # RGBからHSVに変換
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

        # パラメータの展開
        block_size = params['block_size']
        h_weight = params['h_weight']
        s_weight = params['s_weight']
        v_weight = params['v_weight']
        local_scale = params['local_scale']
        c_value = params['c_value']

        # パディングサイズの計算
        pad_size = block_size // 2

        # 画像のパディング
        padded_hsv = cv2.copyMakeBorder(
            hsv_image, pad_size, pad_size, pad_size, pad_size,
            cv2.BORDER_REFLECT
        )
        padded_mask = cv2.copyMakeBorder(
            conj_mask, pad_size, pad_size, pad_size, pad_size,
            cv2.BORDER_REFLECT
        )

        # 出力用マスクの初期化
        height, width = rgb_image.shape[:2]
        vessel_mask = np.zeros((height, width), dtype=np.uint8)

        # 各ピクセルに対して適応的な処理を適用
        for y in range(height):
            for x in range(width):
                if conj_mask[y, x] == 0:  # 角膜領域外はスキップ
                    continue

                # ブロック領域の切り出し
                block_hsv = padded_hsv[y:y+block_size, x:x+block_size]
                block_mask = padded_mask[y:y+block_size, x:x+block_size]

                # マスク領域内のHSV統計量を計算
                hsv_stats = self._compute_local_hsv_stats(block_hsv, block_mask)

                # 現在のピクセルのHSV値
                current_hsv = hsv_image[y, x]

                # 適応的な閾値の計算と適用
                is_vessel = self._apply_adaptive_threshold(
                    current_hsv,
                    hsv_stats,
                    h_weight,
                    s_weight,
                    v_weight,
                    local_scale,
                    c_value
                )

                vessel_mask[y, x] = 255 if is_vessel else 0

        return vessel_mask

    def _compute_local_hsv_stats(self, block_hsv: np.ndarray, block_mask: np.ndarray) -> Dict:
        """ブロック内のHSV統計量を計算"""
        # マスク領域内のピクセルのみを対象に統計量を計算
        valid_pixels = block_hsv[block_mask > 0]

        if len(valid_pixels) == 0:
            return {
                'h_mean': 0, 'h_std': 0,
                's_mean': 0, 's_std': 0,
                'v_mean': 0, 'v_std': 0
            }

        h_values = valid_pixels[..., 0]
        s_values = valid_pixels[..., 1]
        v_values = valid_pixels[..., 2]

        return {
            'h_mean': np.mean(h_values),
            'h_std': np.std(h_values),
            's_mean': np.mean(s_values),
            's_std': np.std(s_values),
            'v_mean': np.mean(v_values),
            'v_std': np.std(v_values)
        }

    def _apply_adaptive_threshold(
        self,
        current_hsv: np.ndarray,
        hsv_stats: Dict,
        h_weight: float,
        s_weight: float,
        v_weight: float,
        local_scale: float,
        c_value: int
    ) -> bool:
        """
        HSV統計量に基づいて適応的な閾値処理を適用
        """
        # 各チャンネルの差異を計算
        h_diff = abs(current_hsv[0] - hsv_stats['h_mean'])
        s_diff = abs(current_hsv[1] - hsv_stats['s_mean'])
        v_diff = abs(current_hsv[2] - hsv_stats['v_mean'])

        # 重み付けされた差異の合計を計算
        weighted_diff = (
            h_weight * h_diff / max(hsv_stats['h_std'], 1) +
            s_weight * s_diff / max(hsv_stats['s_std'], 1) +
            v_weight * v_diff / max(hsv_stats['v_std'], 1)
        )

        # 適応的な閾値を計算
        threshold = local_scale * (
            h_weight * hsv_stats['h_std'] +
            s_weight * hsv_stats['s_std'] +
            v_weight * hsv_stats['v_std']
        ) + c_value

        return weighted_diff > threshold

    def optimize(self, train_images: list, train_masks: list, conj_masks: list, n_iter: int = 50):
        """パラメータの最適化を実行"""
        def objective_function(params):
            params_dict = {
                'block_size': int(params[0]),
                'h_weight': params[1],
                's_weight': params[2],
                'v_weight': params[3],
                'local_scale': params[4],
                'c_value': int(params[5])
            }

            total_iou = 0
            for img, mask, conj in zip(train_images, train_masks, conj_masks):
                pred_mask = self.detect_vessels(img, params_dict, conj)
                iou = self._compute_iou(pred_mask, mask)
                total_iou += iou

            return -total_iou / len(train_images)

        def progress_callback(res):
            # 現在のベストスコアを表示
            print(f"Iter: {len(res.x_iters):3d}, Best IoU: {-res.fun:.4f}")

        result = gp_minimize(
            func=objective_function,
            dimensions=self.param_space,
            n_calls=n_iter,
            n_initial_points=10,
            callback=[progress_callback],
            noise=0.1,
            random_state=get_global_seed(),
        )

        return result

    @staticmethod
    def _compute_iou(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
        """IoUスコアを計算"""
        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()
        return intersection / (union + 1e-7)

# 使用例
if __name__ == "__main__":
    # データ読み込み（RGB形式で読み込むことに注意）
    image = cv2.cvtColor(cv2.imread("eye_image.png"), cv2.COLOR_BGR2RGB)
    true_mask = cv2.imread("vessel_mask.png", 0)
    conj_mask = cv2.imread("conj_mask.png", 0)

    # 検出器の初期化と最適化
    detector = AdaptiveHSVDetector()
    result = detector.optimize(
        [image], [true_mask], [conj_mask],
        n_iter=50
    )

    # 最適パラメータの表示と保存
    print("Best parameters:", result.x)
    print("Best score:", -result.fun)
