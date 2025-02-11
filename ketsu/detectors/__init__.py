import cv2
import numpy as np
from skopt.space import Real, Integer
from typing import List, Dict, Any
from .base import BaseVesselDetector


class SimpleHSVDetector(BaseVesselDetector):
    @property
    def param_space(self) -> List[Real]:
        return [
            Real(0, 5, name='h_min'),
            Real(6, 20, name='h_max'),
            Real(50, 200, name='s_min'),
            Real(201, 255, name='s_max'),
            Real(50, 200, name='v_min'),
            Real(201, 255, name='v_max'),
        ]

    def detect_vessels(self, rgb_image: np.ndarray, conj_mask: np.ndarray,
                      params: Dict[str, Any]) -> np.ndarray:
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        lower = np.array([
            int(params['h_min']),
            int(params['s_min']),
            int(params['v_min'])
        ], dtype=np.uint8)
        upper = np.array([
            int(params['h_max']),
            int(params['s_max']),
            int(params['v_max'])
        ], dtype=np.uint8)

        pred_mask = cv2.inRange(hsv_image, lower, upper)
        pred_mask = cv2.bitwise_and(pred_mask, conj_mask)
        return pred_mask


class AdaptiveHSVDetector(BaseVesselDetector):
    @property
    def param_space(self) -> List[Any]:
        return [
            # HSV parameters
            Real(0, 1, name='h_min'),
            Real(1, 20, name='h_max'),
            Real(10, 200, name='s_min'),
            Real(250, 255, name='s_max'),
            Real(10, 200, name='v_min'),
            Real(200, 255, name='v_max'),
            # Adaptive threshold parameters
            Integer(1, 21, name='block_size'),
            Real(-10, 20, name='c_value')
        ]

    def detect_vessels(self, rgb_image: np.ndarray, conj_mask: np.ndarray,
                      params: Dict[str, Any]) -> np.ndarray:
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

        # HSV thresholding
        lower = np.array([
            int(params['h_min']),
            int(params['s_min']),
            int(params['v_min'])
        ], dtype=np.uint8)
        upper = np.array([
            int(params['h_max']),
            int(params['s_max']),
            int(params['v_max'])
        ], dtype=np.uint8)
        hsv_mask = cv2.inRange(hsv_image, lower, upper)

        # Adaptive thresholding on V channel
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

        # Combine masks
        combined_mask = cv2.bitwise_and(hsv_mask, v_adaptive)
        final_mask = cv2.bitwise_and(combined_mask, conj_mask)

        # Post-processing
        kernel = np.ones((3,3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

        return final_mask


class CLAHEHSVDetector(BaseVesselDetector):
    @property
    def param_space(self) -> List[Any]:
        return [
            # HSV parameters
            Real(0, 5, name='h_min'),
            Real(6, 20, name='h_max'),
            Real(50, 200, name='s_min'),
            Real(50, 200, name='v_min'),
            # CLAHE parameters
            Real(0.5, 5.0, name='clip_limit'),
            Integer(4, 16, name='tile_size')
        ]

    def detect_vessels(self, rgb_image: np.ndarray, conj_mask: np.ndarray,
                      params: Dict[str, Any]) -> np.ndarray:
        # Convert to HSV
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

        # Apply CLAHE to V channel
        clahe = cv2.createCLAHE(
            clipLimit=params['clip_limit'],
            tileGridSize=(params['tile_size'], params['tile_size'])
        )
        hsv_image[:,:,2] = clahe.apply(hsv_image[:,:,2])

        # Create HSV mask
        lower = np.array([
            int(params['h_min']),
            int(params['s_min']),
            int(params['v_min'])
        ], dtype=np.uint8)
        upper = np.array([
            int(params['h_max']),
            255,  # S max
            255   # V max
        ], dtype=np.uint8)

        # Apply HSV threshold
        vessel_mask = cv2.inRange(hsv_image, lower, upper)

        # Apply conjunctiva mask
        final_mask = cv2.bitwise_and(vessel_mask, conj_mask)

        # Optional: Minor noise removal
        kernel = np.ones((3,3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

        return final_mask
