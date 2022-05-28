import torch
import os
import cv2
import json
import numpy as np
from PIL import Image, ImageDraw

from data.iMaterialistics import create_square_crop_by_detection


def create_mask_by_polygons(polygons, width, height):
    res_mask = np.zeros((height, width), dtype=np.uint8)

    pil_img = Image.fromarray(res_mask)
    draw = ImageDraw.Draw(pil_img)
    for poly in polygons:
        draw.polygon(poly, 255)
    del draw

    return np.array(pil_img)


class DeepFashion2Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_path: str,
                 shape: tuple = (700, 700)):
        self.annos_path = os.path.join(root_path, 'annos/')
        self.images_path = os.path.join(root_path, 'image/')

        assert os.path.isdir(self.annos_path)
        assert os.path.isdir(self.images_path)
        assert len(os.listdir(self.annos_path)) == \
               len(os.listdir(self.images_path))

        self.images_names = sorted(os.listdir(self.images_path))
        self.shape = shape

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        img_name = self.images_names[idx]
        base_name, _ = os.path.splitext(img_name)
        anno_name = base_name + '.json'

        image = cv2.imread(
            os.path.join(self.images_path, img_name),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise RuntimeError(
                'Can\'t open image: {}'.format(
                    os.path.join(self.images_path, img_name)
                )
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image, (sx, sy) = create_square_crop_by_detection(
            image,
            [0, 0, *image.shape[:2][::-1]],
            True,
            True
        )

        original_padded_image_size = image.shape[0]

        image = cv2.resize(image, self.shape, interpolation=cv2.INTER_AREA)

        with open(os.path.join(self.annos_path, anno_name), 'r') as jf:
            anno_data = json.load(jf)

        masks = []
        classes = []
        boxes = []

        for key in anno_data.keys():
            if 'item' in key:
                cls = anno_data[key]['category_id'] - 1
                segment_polygons = anno_data[key]['segmentation']
                box = anno_data[key]['bounding_box']

                box = (np.array(box) - [sx, sy, sx, sy]) * [*self.shape,
                                                            *self.shape] // original_padded_image_size
                box = np.clip(box, 0, self.shape[0])

                segment_polygons = [
                    np.clip(
                        (
                            np.array(p).reshape((-1, 2)) - np.array(
                                [sx, sy])
                        ) * list(self.shape) // original_padded_image_size,
                        0, self.shape[0]
                    ).flatten().tolist()
                    for p in segment_polygons
                ]

                classes.append(cls)
                masks.append(
                    create_mask_by_polygons(
                        segment_polygons,
                        *self.shape
                    )
                )
                boxes.append(box)

        num_crowds = 0
        im = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
        gt = torch.FloatTensor(
            np.hstack(
                (
                    np.array(boxes) / [*self.shape, *self.shape],
                    np.expand_dims(classes, axis=1)
                )
            )
        )
        masks = torch.FloatTensor(np.array(masks)) / 255.0

        # print(
        #     {
        #         'gt': gt,
        #         'masks shape': masks.shape,
        #         'img shape': im.shape,
        #         'base name': base_name
        #     }
        # )

        return im, (gt, masks, num_crowds)
