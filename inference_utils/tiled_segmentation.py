from typing import Tuple, List
import cv2
import numpy as np
from shapely.geometry import Polygon
from imantics import Mask
from inference_utils.yolact_inference import YOLACTModel
from tqdm import tqdm

WINDOW_STRIDE = 4/5
POLYGONS_MATCHING_THRESHOLD = 0.05
FILTER_FALSE_DETECTIONS_THRESHOLD = 0.1
POINTS_MATCHING_THRESHOLD = 0.3


def compute_poly_incircle_center(_poly: Polygon) -> Tuple[int, int]:
    ppoints = np.array(_poly.exterior.xy).T.astype(np.float32)
    box = cv2.boundingRect(ppoints)
    ppoints -= box[:2]
    ppoints = np.expand_dims(ppoints, axis=1).astype(np.int0)

    mask = np.zeros((box[3], box[2]), dtype=np.uint8)
    mask = cv2.drawContours(mask, [ppoints], 0, 255, -1)
    dres = cv2.distanceTransform(mask, cv2.DIST_L2, 5, cv2.DIST_LABEL_PIXEL)
    _, _, _, (cx, cy) = cv2.minMaxLoc(dres, None)
    return (ppoints.squeeze(1).mean(axis=0) + box[:2]).astype(np.int32)
    # return cx + box[0], cy + box[1]


def tiling_intersected(
        img: np.ndarray,
        tile_size: int,
        step: float = WINDOW_STRIDE) -> List[List[Tuple[int, int]]]:
    stride = int(tile_size * step)

    x0_vec = []
    y0_vec = []

    target_x = 0
    while target_x + tile_size < img.shape[1]:
        x0_vec.append(target_x)
        target_x += stride
    x0_vec.append(img.shape[1] - tile_size - 1)

    target_y = 0
    while target_y + tile_size < img.shape[0]:
        y0_vec.append(target_y)
        target_y += stride
    y0_vec.append(img.shape[0] - tile_size - 1)

    poses = []

    for y0 in y0_vec:
        line_positions = []
        for x0 in x0_vec:
            line_positions.append((x0, y0))
        poses.append(line_positions)

    return poses


class PolyDetection(object):
    def __init__(self, _mask: np.ndarray, class_num: int, position: Tuple[int, int]):
        self.cls = class_num

        wrapped_mask = Mask(_mask)
        segmentation_data = wrapped_mask.polygons().segmentation

        assert len(segmentation_data) > 0, 'Empty polygon of class {}'.format(self.cls)

        segmentation_data = max(segmentation_data, key=lambda _x: len(_x))

        if len(segmentation_data) >= 7*5:
            segmentation_array = np.array(
                segmentation_data).reshape((-1, 2))[::5].astype(np.float32)
        else:
            segmentation_array = np.array(
                segmentation_data).reshape((-1, 2)).astype(np.float32)

        self.poly = Polygon(segmentation_array + position)

    def estimate_iou(self, other) -> float:
        try:
            intersection = self.poly.intersection(other.poly)
        except:
            return 0

        if intersection.area < 1E-5:
            return 0

        union = self.poly.union(other.poly)

        return intersection.area / (union.area + 1E-5)


class DetectionsCarrier(object):
    detections: List[PolyDetection] = None


def merge_carriers(
        scope_src: DetectionsCarrier,
        scope_to_add: DetectionsCarrier,
        match_threshold: float = POLYGONS_MATCHING_THRESHOLD):
    pairwise_intersections = np.array([
        [
            scope_to_add.detections[si].estimate_iou(scope_src.detections[sj])
            for sj in range(len(scope_src.detections))
        ]
        for si in range(len(scope_to_add.detections))
    ])

    for si in range(len(scope_to_add.detections)):
        search_idx = -1
        best_iou = 0
        diff_poly = Polygon(scope_to_add.detections[si].poly)
        full_skip = False

        for sj in range(pairwise_intersections.shape[1]):
            p_iou = pairwise_intersections[si][sj]

            try:
                diff_poly = diff_poly.difference(scope_src.detections[sj].poly)
            except:
                continue
                # full_skip = True
                # break

            if p_iou - match_threshold > -1E-5:
                if p_iou - best_iou > -1E-5:
                    search_idx = sj
                    best_iou = p_iou

        if full_skip:
            continue

        if search_idx == -1:
            scope_src.detections.append(scope_to_add.detections[si])
        else:
            scope_src.detections[search_idx].poly = scope_src.detections[search_idx].poly.union(scope_to_add.detections[si].poly)


class ImageSegment(DetectionsCarrier):
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
            PolyDetection(masks[_mi], pred_classes[_mi], position)
            for _mi in range(len(masks))
        ]

    def get_crop(self):
        return self.hole_image[
               self.pos[1]:self.pos[1] + self.size[1],
               self.pos[0]:self.pos[0] + self.size[0]
            ]

    def __len__(self):
        return len(self.detections)


class WindowReadyImage(DetectionsCarrier):
    def __init__(self,
                 image: np.ndarray,
                 inference_function: YOLACTModel,
                 tile_size: int = 512):
        self.segments = [
            [
                ImageSegment(image, t, (tile_size, tile_size), inference_function)
                for t in t_line
            ]
            for t_line in tqdm(tiling_intersected(image, tile_size))
        ]
        self.detections = []

        for t_line in tqdm(self.segments):
            for t in t_line:
                # self.detections += t.detections
                merge_carriers(self, t)

        self.filter_noisy_points()

    def average_polygons_area(self) -> float:
        return np.array(
            [
                d.poly.area
                for d in self.detections
            ]
        ).mean()

    def filter_noisy_points(self):
        avg_area = self.average_polygons_area()
        self.detections = [
            d
            for d in self.detections
            if d.poly.area - avg_area * FILTER_FALSE_DETECTIONS_THRESHOLD > -1E-5
        ]

    def get_points(self):
        avg_d = np.sqrt(self.average_polygons_area())
        center_points = [
            compute_poly_incircle_center(p.poly)
            for p in self.detections
        ]

        filtered_points = []

        for bp in center_points:
            in_set = False
            for p in filtered_points:
                if np.linalg.norm(p - bp) - avg_d * POINTS_MATCHING_THRESHOLD < 1E-5:
                    in_set = True
                    continue

            if not in_set:
                filtered_points.append(np.array(bp))

        return filtered_points


if __name__ == '__main__':
    import cv2


    def read_image(img_p: str) -> np.ndarray:
        _img = cv2.imread(img_p, cv2.IMREAD_COLOR)
        assert _img is not None, img_p
        return cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)


    yolact_inference = YOLACTModel()
    sample_image = read_image(
        '/media/alexey/SSDDataDisk/datasets/walruses/raw/images/191.jpg')
    pred_sample = WindowReadyImage(sample_image, yolact_inference, 700)
    pred_sample.get_points()
