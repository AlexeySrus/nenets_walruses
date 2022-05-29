import base64
import io
import os
import sys
from functools import lru_cache
from typing import List

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from pydantic import conlist

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', 'inference_utils')
)

from mmdet_inference import MMDetectionQueryInstInference
from young_classifier import YoungWalrusesClassier
from tiled_segmentation import WindowReadyImage


app = FastAPI()


class Response(BaseModel):
    centers: List[conlist(int, min_items=2, max_items=2)]
    polygons: List[List[int]]
    classes: List[bool]


class MockModel:
    """Class that is used as model and has the same format output."""

    def __init__(self):
        """Initialize mock model."""
        bg = np.zeros((100, 100), np.uint8)
        masks = [bg.copy() for i in range(3)]
        masks[0][10:20, 10:20] += 1
        masks[1][30:50, 15:30] += 1
        masks[1][50:60, 70:85] += 1
        self.masks = np.array(masks, dtype=np.uint8)

        self.boxes = np.array([
            [10, 10, 20, 20],
            [25, 10, 55, 30],
            [20, 10, 55, 30]
        ])

        self.classes = [0, 1, 0]

    def __call__(self, image: np.ndarray, *args):
        """Get mock results."""
        h, w = image.shape[:2]
        masks = []
        for mask in self.masks:
            m = mask.copy()
            m = cv2.resize(m, (w, h))
            masks.append(m)
        return masks, self.boxes, self.classes


@lru_cache()
def get_models():
    """Load ML model to process images."""
    # Not the best model. It may work on CPU.
    # segm_model = YOLACTModel()
    # Better model. Cuda only.
    segm_model = MMDetectionQueryInstInference(conf=0.5)
    # segm_model = MockModel()
    young_clasifier = YoungWalrusesClassier(conf=0.7)
    return segm_model, young_clasifier


def base64str_to_PILImage(base64str):
    base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    bytesObj = io.BytesIO(base64bytes)
    img = Image.open(bytesObj)
    return img


def predict(image: Image):
    """Get prediction for the given image."""
    segm_model, young_clasifier = get_models()
    npimage = np.array(image.convert('RGB'))
    wri = WindowReadyImage(npimage, segm_model, young_clasifier)
    polygons = [
        np.array(det.poly.exterior.xy).T.astype(int).ravel().tolist()
        for det in wri.detections
    ]
    classes = [det.cls == 1 for det in wri.detections]
    centers = [points.tolist() for points in wri.get_points()]

    return Response(
        centers=centers,
        polygons=polygons,
        classes=classes
    )


@app.post('/predict', response_model=Response)
def get_predict(image: UploadFile = File(...)):
    img = Image.open(image.file)
    response = predict(img)
    return response
