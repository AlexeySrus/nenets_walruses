from mmdet.apis import init_detector, inference_detector
import cv2
import numpy as np
import os
from PIL import Image


class MMDetectionQueryInstInference(object):
    def __init__(self, model_config_path: str = None, device: str = 'cuda', conf: float = 0.3):
        if model_config_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '../data/epoch_12.pth')
            config_path = os.path.join(os.path.dirname(__file__), '../data/animals.py')
            model_config_path = model_path, config_path

        model_path, config_path = model_config_path
        self.model = init_detector(config_path, model_path, device=device)
        self.conf = conf

    def __call__(self, input_image: np.ndarray) -> tuple:
        image = input_image.copy()
        if max(image.shape[:2]) > 700:
            k = 700 / max(image.shape[:2])
            image = cv2.resize(
                image,
                None,
                fx=k,
                fy=k,
                interpolation=cv2.INTER_AREA
            )

        result = inference_detector(self.model, image)

        confs = [result[0][0][i][4] for i in range(len(result[1][0]))]

        res_masks = []

        for _i, c in enumerate(confs):
            if c > self.conf:
                res_masks.append(
                    np.array(result[1][0][_i]).astype(np.uint8) * 255
                )

        return res_masks, None, [1] * len(res_masks)
