from typing import Tuple, List, Any
import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from imantics import Mask
from inference_utils.yolact_inference import YOLACTModel
from inference_utils.mmdet_inference import MMDetectionQueryInstInference
from inference_utils.crop_utils import create_square_crop_by_detection
from tqdm import tqdm

WINDOW_STRIDE = 3/4
POLYGONS_MATCHING_THRESHOLD = 0.2
FILTER_FALSE_DETECTIONS_THRESHOLD = 0.15
POINTS_MATCHING_THRESHOLD = 0.3
BIG_FILTER = 3


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


def multi_to_single_polygon(_poly):
    if _poly.geom_type == 'MultiPolygon' or _poly.geom_type == 'GeometryCollection':
        return _poly.convex_hull
    return _poly


class PolyDetection(object):
    def __init__(self, _mask: np.ndarray, class_num: int, position: Tuple[int, int]):
        self.cls = class_num

        contours, _ = cv2.findContours(
            _mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        contours = list(contours)
        contours.sort(key=lambda _x: cv2.contourArea(_x))

        if len(contours) == 0:
            self.poly = None
            return

        segmentation_data = contours[0].squeeze(1)

        if len(segmentation_data) > 100:
            segmentation_data = segmentation_data[::4]
        # elif len(segmentation_data) > 30:
        #     segmentation_data = segmentation_data[::4]

        if len(segmentation_data) < 3:
            self.poly = None
            return

        segmentation_array = np.array(
            segmentation_data).astype(np.float32) + position

        self.poly = Polygon(segmentation_array)

    def get_square_crop(self, hole_image: np.ndarray) -> np.ndarray:
        box = cv2.boundingRect(np.array(self.poly.exterior.xy).T.astype(np.float32))
        x1, y1, w, h = [int(_e) for _e in box]
        x2, y2 = x1 + w, y1 + h
        return create_square_crop_by_detection(hole_image, [x1, y1, x2, y2])

    def set_class(self, class_num: int):
        self.cls = class_num

    def estimate_iou(self, other) -> float:
        intersection = self.poly.intersection(other.poly)

        if intersection.area < 1E-5:
            return 0

        union = self.poly.union(other.poly)

        return intersection.area / (union.area + 1E-5)


class DetectionsCarrier(object):
    detections: List[PolyDetection] = None

    def erase_invalids(self):
        self.detections = [
            d
            for d in self.detections
            if d.poly is not None and d.poly.is_valid
        ]


def merge_carriers(
        scope_src: DetectionsCarrier,
        scope_to_add: DetectionsCarrier,
        match_threshold: float = POLYGONS_MATCHING_THRESHOLD):
    pairwise_intersections = [
        [
            scope_to_add.detections[si].estimate_iou(scope_src.detections[sj])
            for sj in range(len(scope_src.detections))
        ]
        for si in range(len(scope_to_add.detections))
    ]

    for si in range(len(scope_to_add.detections)):
        search_idx = -1
        best_iou = 0

        diff_poly = Polygon(scope_to_add.detections[si].poly)

        for sj in range(len(pairwise_intersections[si])):
            if pairwise_intersections[si][sj] < 1E-5:
                continue

            if not diff_poly.is_valid:
                continue

            p_iou = pairwise_intersections[si][sj]

            diff_poly = diff_poly.difference(scope_src.detections[sj].poly)

            if p_iou - match_threshold > -1E-5:
                if p_iou - best_iou > -1E-5:
                    search_idx = sj
                    best_iou = p_iou

        if search_idx == -1:
            scope_src.detections.append(scope_to_add.detections[si])
            for k in range(si + 1, len(scope_to_add.detections)):
                pairwise_intersections[k].append(scope_to_add.detections[k].estimate_iou(scope_src.detections[-1]))
        else:
            if diff_poly.is_valid:
                scope_src.detections[search_idx].poly = multi_to_single_polygon(scope_src.detections[search_idx].poly.union(diff_poly))


class ImageSegment(DetectionsCarrier):
    def __init__(self,
                 hole_image: np.ndarray,
                 position: Tuple[int, int],
                 size: Tuple[int, int],
                 inference_function: callable):
        self.pos = position
        self.size = size
        self.hole_image = hole_image

        masks, _, pred_classes = inference_function(self.get_crop())

        self.detections = [
            PolyDetection(masks[_mi], pred_classes[_mi], position)
            for _mi in range(len(masks))
        ]

        self.erase_invalids()

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
                 inference_function: callable,
                 young_inference: callable,
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

        for _di in tqdm(range(len(self.detections))):
            self.detections[_di].set_class(
                young_inference(self.detections[_di].get_square_crop(image))
            )

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
            if d.poly.area - avg_area * FILTER_FALSE_DETECTIONS_THRESHOLD > -1E-5 and isinstance(d.poly, Polygon) and d.poly.area < avg_area * BIG_FILTER
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


    inference_f = MMDetectionQueryInstInference()
    sample_image = read_image(
        '/media/alexey/SSDDataDisk/datasets/walruses/raw/images/DJI_0049.jpg')
    pred_sample = WindowReadyImage(sample_image, inference_f, 700)
    pred_sample.get_points()
