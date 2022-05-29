from argparse import ArgumentParser, Namespace
import cv2
import numpy as np
from tqdm import tqdm
import os
import pandas as pd

from inference_utils.tiled_segmentation import WindowReadyImage
from inference_utils.mmdet_inference import MMDetectionQueryInstInference
from inference_utils.young_classifier import YoungWalrusesClassier


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description='Prepare csv predictions'
    )
    parser.add_argument(
        '--input', '-i', type=str, required=True,
        help='Path to folder with images'
    )
    parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='Path to output folder which will contain a csv predictions'
    )
    return parser.parse_args()


def read_image(img_p: str) -> np.ndarray:
    _img = cv2.imread(img_p, cv2.IMREAD_COLOR)
    assert _img is not None, img_p
    return cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)


def main():
    args = parse_args()

    inference_f = MMDetectionQueryInstInference(conf=0.5)
    young_f = YoungWalrusesClassier(conf=0.7)

    os.makedirs(args.output, exist_ok=True)

    for img_name in tqdm(os.listdir(args.input)):
        image_path = os.path.join(args.input, img_name)
        basename = os.path.splitext(img_name)[0]

        pred_sample = WindowReadyImage(
            read_image(image_path),
            inference_f,
            young_f,
            700,
            True
        )

        points = pred_sample.get_points()
        x_coords = [int(pp[0]) for pp in points]
        y_coords = [int(pp[1]) for pp in points]

        res_path = os.path.join(args.output, '{}.csv'.format(basename))

        res_datafarame = pd.DataFrame(
            {'x': x_coords, 'y': y_coords}
        )

        res_datafarame.to_csv(res_path, index=False)


if __name__ == '__main__':
    main()
