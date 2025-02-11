from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from skopt.space import Real, Integer, Categorical
import numpy as np
from skopt import gp_minimize
from tqdm import tqdm

from ..utils import get_global_seed

class BaseVesselDetector(ABC):
    """Base class for vessel detection algorithms"""

    @property
    @abstractmethod
    def param_space(self) -> List[Any]:
        """Define the parameter space for optimization"""
        pass

    @abstractmethod
    def detect_vessels(self, rgb_image: np.ndarray, conj_mask: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Detect vessels in the given image"""
        pass

    def params_to_dict(self, params: List[float]) -> Dict[str, Any]:
        """Convert optimization parameters to a dictionary"""
        return {dim.name: self._convert_param_value(params[i], dim)
                for i, dim in enumerate(self.param_space)}

    def _convert_param_value(self, value: float, dimension: Any) -> Any:
        """Convert parameter value based on dimension type"""
        if isinstance(dimension, (Real, Integer)):
            return int(value) if isinstance(dimension, Integer) else value
        elif isinstance(dimension, Categorical):
            return dimension.categories[int(value)]
        return value

    def optimize(self, images: List[np.ndarray], conj_masks: List[np.ndarray],
                vessel_masks: List[np.ndarray], n_iter: int = 50) -> Tuple[Dict[str, Any], float]:
        """Optimize detector parameters using Bayesian optimization"""

        def objective_function(params):
            params_dict = self.params_to_dict(params)
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

        best_params = self.params_to_dict(result.x)
        best_iou = -result.fun

        return best_params, best_iou

    def evaluate(self, images: List[np.ndarray], conj_masks: List[np.ndarray],
                vessel_masks: List[np.ndarray], params: Dict[str, Any]) -> Tuple[float, float]:
        """Evaluate detector performance on a dataset"""
        ious = []
        for image, conj_mask, vessel_mask in tqdm(zip(images, conj_masks, vessel_masks)):
            pred_mask = self.detect_vessels(image, conj_mask, params)
            iou = self._compute_iou(pred_mask, vessel_mask)
            ious.append(iou)
        return np.mean(ious), np.std(ious)

    @staticmethod
    def _compute_iou(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
        """Compute Intersection over Union (IoU) between two masks"""
        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()
        return intersection / (union + 1e-7)
