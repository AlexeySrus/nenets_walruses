from typing import Tuple, List
import cv2
import numpy as np
from shapely.geometry import Polygon
from imantics import Mask
from inference_utils.yolact_inference import YOLACTModel


def tiling_intersected(
        img: np.ndarray, tile_size: int, step: float = 3/4) -> List[Tuple[int, int]]:
    stride = int(tile_size * step)

    x0_vec = []
    y0_vec = []

    target_x = 0
    while target_x + tile_size < img.size(2):
        x0_vec.append(target_x)
        target_x += stride
    x0_vec.append(img.size(2) - tile_size - 1)

    target_y = 0
    while target_y + tile_size < img.size(1):
        y0_vec.append(target_y)
        target_y += stride
    y0_vec.append(img.size(1) - tile_size - 1)

    poses = []

    for y0 in y0_vec:
        for x0 in x0_vec:
            poses.append((x0, y0))

    return poses


class PolyDetection(object):
    def __init__(self, _mask: np.ndarray, class_num: int):
        self.cls = class_num

        wrapped_mask = Mask(_mask)
        segmentation_data = wrapped_mask.polygons().segmentation

        assert len(segmentation_data) > 0, 'Empty polygon of class {}'.format(self.cls)

        segmentation_data = segmentation_data[0]

        if len(segmentation_data) >= 7*5:
            segmentation_array = np.array(
                segmentation_data).reshape((-1, 2))[::5].astype(np.float32)
        else:
            segmentation_array = np.array(
                segmentation_data).reshape((-1, 2)).astype(np.float32)

        self.poly = Polygon(segmentation_array)


class ImageSegment(object):
    def __init__(self,
                 hole_image: np.ndarray,
                 position: Tuple[int, int],
                 size: Tuple[int, int],
                 inference_function: YOLACTModel):
        self.pos = position
        self.size = size
        self.hole_image = hole_image

        masks, _, pred_classes = inference_function(self.get_crop())

        self.detections = [
            PolyDetection(masks[_mi], pred_classes[_mi])
            for _mi in range(len(masks))
        ]

    def get_crop(self):
        return self.hole_image[
               self.pos[1]:self.pos[1] + self.size[1],
               self.pos[0]:self.pos[0] + self.size[0]
            ]

    def __len__(self):
        return len(self.detections)


class WindowReadyImage(object):
    def __init__(self,
                 image: np.ndarray,
                 inference_function: YOLACTModel,
                 tile_size: int = 512):
        tiles_positions = tiling_intersected(image, tile_size)

        self.segments = [
            ImageSegment(image, t, (tile_size, tile_size), inference_function)
            for t in tiles_positions
        ]

