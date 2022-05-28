import cv2
import sys
import torch
import os
import numpy as np


class YOLACTModel(object):
    class_names = ['walrus']

    def __init__(self,
                 yolact_path: str = os.path.join(os.path.dirname(__file__), 'custom_yolact/'),
                 device: str = 'cpu'):
        sys.path.append(yolact_path)
        from eval import FastBaseTransform, main, Yolact
        from layers.output_utils import postprocess

        self.postprocess = postprocess
        self.FastBaseTransform = FastBaseTransform

        self.yolact_model = main()

        self.device = device

    def yolact_model_inference(self, input_image: np.ndarray) -> tuple:
        image = input_image.copy()
        k = 1
        if max(image.shape[:2]) > 700:
            k = 700 / max(image.shape[:2])
            image = cv2.resize(
                image,
                None,
                fx=k,
                fy=k,
                interpolation=cv2.INTER_AREA
            )

        frame = torch.from_numpy(image).to(self.device).float()
        batch = self.FastBaseTransform()(frame.unsqueeze(0))

        y = self.yolact_model(batch)

        preds = self.postprocess(
            y,
            image.shape[1],
            image.shape[0],
            interpolation_mode='bilinear',
            crop_masks=True,
            score_threshold=0.8
        )

        preds = [p.detach().to('cpu') for p in preds]

        boxes = [
            (box.astype(np.float32) / k).astype(np.int64)
            for i, box in enumerate(preds[2].detach().numpy())
        ]

        masks = np.array(
            [
                cv2.resize(
                    (mask * 255.0).astype(np.uint8),
                    tuple(input_image.shape[:2][::-1]),
                    interpolation=cv2.INTER_AREA
                )
                for i, mask in enumerate(preds[3].detach().numpy())
            ]
        )

        boxes = [
            box
            for i, box in enumerate(boxes)
        ]

        classes = [
            int(preds[0][i].detach().numpy())
            for i, box in enumerate(boxes)
        ]

        return masks, boxes, classes

    def __call__(self,input_image: np.ndarray) -> tuple:
        return self.get_instance_masks(input_image)

    def get_instance_masks(self, input_image: np.ndarray) -> tuple:
        masks, boxes, classes = self.yolact_model_inference(
            cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        )

        return masks, boxes, classes
