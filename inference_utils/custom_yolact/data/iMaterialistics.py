import torch
import os
import cv2
import numpy as np
import pandas as pd

from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def create_square_crop_by_detection(
        frame: np.ndarray,
        box: list,
        return_shifts: bool = False,
        zero_pad: bool = False):
    """
    Rebuild detection box to square shape
    Args:
        frame: rgb image in np.uint8 format
        box: list with follow structure: [x1, y1, x2, y2]
        return_shifts: if set True then function return tuple of image crop
           and (x, y) tuple of shift coordinates
        zero_pad: pad result image by zeros values

    Returns:
        Image crop by box with square shape or tuple of crop and shifted coords
    """
    w = box[2] - box[0]
    h = box[3] - box[1]
    cx = box[0] + w // 2
    cy = box[1] + h // 2
    radius = max(w, h) // 2
    exist_box = []
    pads = []

    # y top
    if cy - radius >= 0:
        exist_box.append(cy - radius)
        pads.append(0)
    else:
        exist_box.append(0)
        pads.append(-(cy - radius))

    # y bottom
    if cy + radius >= frame.shape[0]:
        exist_box.append(frame.shape[0] - 1)
        pads.append(cy + radius - frame.shape[0] + 1)
    else:
        exist_box.append(cy + radius)
        pads.append(0)
    # x left
    if cx - radius >= 0:
        exist_box.append(cx - radius)
        pads.append(0)

    else:
        exist_box.append(0)
        pads.append(-(cx - radius))

    # x right
    if cx + radius >= frame.shape[1]:
        exist_box.append(frame.shape[1] - 1)
        pads.append(cx + radius - frame.shape[1] + 1)
    else:
        exist_box.append(cx + radius)
        pads.append(0)

    exist_crop = frame[
                 exist_box[0]:exist_box[1],
                 exist_box[2]:exist_box[3]
                 ]

    if len(frame.shape) > 2:
        croped = np.pad(
            exist_crop,
            (
                (pads[0], pads[1]),
                (pads[2], pads[3]),
                (0, 0)
            ),
            'edge' if not zero_pad else 'constant'
        )
    else:
        croped = np.pad(
            exist_crop,
            (
                (pads[0], pads[1]),
                (pads[2], pads[3])
            ),
            'edge' if not zero_pad else 'constant'
        )

    if not return_shifts:
        return croped

    shift_x = exist_box[2] - pads[2]
    shift_y = exist_box[0] - pads[0]

    return croped, (shift_x, shift_y)


class SegmentationTrainTransform(object):
    def __init__(self):
        gaussian_blur_sigma_max = 1.0
        gaussian_noise_sigma_max = 0.05

        self.seq = iaa.Sequential(
            children=[
                iaa.Sequential(
                    children=[
                        iaa.Sequential(
                            children=[
                                iaa.Affine(
                                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                                    rotate=(-5, 5),
                                    shear=(-16, 16),
                                    order=iap.Choice([0, 1, 3], p=[0.15, 0.80, 0.05]),
                                    mode="constant",
                                    name="Affine"),
                                iaa.PerspectiveTransform(
                                    scale=0.0,
                                    name="PerspectiveTransform"),
                                iaa.Sometimes(
                                    p=0.1,
                                    then_list=iaa.PiecewiseAffine(
                                        scale=(0.0, 0.01),
                                        nb_rows=(4, 20),
                                        nb_cols=(4, 20),
                                        order=iap.Choice([0, 1, 3], p=[0.15, 0.80, 0.05]),
                                        mode="constant",
                                        name="PiecewiseAffine"))],
                            random_order=True,
                            name="GeomTransform"),
                        iaa.Sequential(
                            children=[
                                iaa.Sometimes(
                                    p=0.75,
                                    then_list=iaa.Add(
                                        value=(-10, 10),
                                        per_channel=0.5,
                                        name="Brightness")),
                                iaa.Sometimes(
                                    p=0.05,
                                    then_list=iaa.Emboss(
                                        alpha=(0.0, 0.5),
                                        strength=(0.5, 1.2),
                                        name="Emboss")),
                                iaa.Sometimes(
                                    p=0.1,
                                    then_list=iaa.Sharpen(
                                        alpha=(0.0, 0.5),
                                        lightness=(0.5, 1.2),
                                        name="Sharpen")),
                                iaa.Sometimes(
                                    p=0.25,
                                    then_list=iaa.contrast.LinearContrast(
                                        alpha=(0.5, 1.5),
                                        per_channel=0.5,
                                        name="ContrastNormalization"))
                            ],
                            random_order=True,
                            name="ColorTransform"),
                        iaa.Sequential(
                            children=[
                                iaa.Sometimes(
                                    p=0.5,
                                    then_list=iaa.AdditiveGaussianNoise(
                                        loc=0,
                                        scale=(0.0, 255.0 * 2.0 * gaussian_noise_sigma_max),
                                        per_channel=0.5,
                                        name="AdditiveGaussianNoise")),
                                iaa.Sometimes(
                                    p=0.1,
                                    then_list=iaa.SaltAndPepper(
                                        p=(0, 0.001),
                                        per_channel=0.5,
                                        name="SaltAndPepper"))],
                            random_order=True,
                            name="Noise"),
                        iaa.OneOf(
                            children=[
                                iaa.Sometimes(
                                    p=0.2,
                                    then_list=iaa.imgcorruptlike.DefocusBlur(
                                        name="DefocusBlur")),
                                iaa.Sometimes(
                                    p=0.2,
                                    then_list=iaa.imgcorruptlike.ZoomBlur(
                                        name="ZoomBlur"))],
                            name="CamAug")
                    ],
                    random_order=True,
                    name="MainProcess")])

        print('Augmentations used')

    def __call__(self, src_img, src_masks):
        seq_det = self.seq.to_deterministic()

        segmaps = [
            SegmentationMapsOnImage(
                src_mask // 255,
                shape=src_img.shape
            ) for src_mask in src_masks
        ]

        used_segmentation_to_masks = [
            src_mask.sum() > 0
            for src_mask in src_masks
        ]

        img_and_masks = [
            seq_det(image=src_img, segmentation_maps=segmap)
            if i == 0 or used_segmentation_to_masks[i] else
            (None, src_masks[i])
            for i, segmap in enumerate(segmaps)
        ]

        img = img_and_masks[0][0]

        masks = [mask[1] for mask in img_and_masks]

        # for i, mask in enumerate(masks):
        #     if used_segmentation_to_masks[i]:
        #         a = mask.draw(size=src_img.shape[:2])
        #         a = a[..., 0].astype(np.uint8)

        masks = np.array([
            mask.draw(size=src_img.shape[:2])[0][..., 0].astype(np.uint8)
            if used_segmentation_to_masks[i] else
            src_masks[i]
            for i, mask in enumerate(masks)
        ])

        masks_elements = masks >= 100

        masks[masks_elements] = 255
        masks[np.bitwise_not(masks_elements)] = 0

        return img, masks


class iMaterialisticFashionDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_path: str,
                 shape: tuple = (700, 700),
                 for_train: bool = True,
                 augmentations: bool = False):
        self.images_path = os.path.join(root_path, 'train/')
        self.masks_path = os.path.join(root_path, 'train_masks/')

        self.images_names = os.listdir(self.images_path)

        with open(
            os.path.join(root_path, 'checked_names.txt'),
            'r'
        ) as f:
            self.masks_names = [line.rstrip() for line in f]

        assert len(self.images_names) >= len(self.masks_names)

        n = int(len(self.masks_names)*0.9)

        if for_train:
            self.masks_names = self.masks_names[:n]
        else:
            self.masks_names = self.masks_names[n:]

        # TODO: Add base names in folders assertion

        self.train_csv = pd.read_csv(
            os.path.join(root_path, 'train.csv'),
            index_col=['ImageId']
        )

        self.shape = shape

        self.ids = [
            os.path.splitext(mask_name)[0]
            for mask_name in self.masks_names
        ]

        self.classes_wrap = [
            [0, 28, 30, 31, 33, 41],
            [1, 27, 30, 31, 33, 35, 41],
            [2, 30, 31, 35, 41],
            [3, 30, 31, 35, 41],
            [4, 27, 29, 30, 31, 35, 41],
            [5, 30, 31, 35],
            [6, 32, 42],
            [7, 32, 42],
            [8, 42],
            [9, 29, 30, 31, 35, 41],
            [10, 31, 32, 35, 37, 38, 39, 44],
            [11, 31],
            [12, 27],
            [13, 31, 33],
            [14],
            [15],
            [16],
            [17],
            [18],
            [19],
            [20],
            [21],
            [22],
            [23, 34],
            [24],
            [25],
            [26],
        ]

        self.max_size = 1000

        self.augmentations = None if not augmentations else \
            SegmentationTrainTransform()

    def get_real_class(self, cl: int, classes: list) -> int:
        sleeve_wrap_order = [9, 10, 11, 4, 3, 2, 0, 1, 5]
        neckline_wrap_order = [2, 3, 0, 1, 4, 5]

        if cl == 31 and cl in classes:
            for idx in sleeve_wrap_order:
                if idx in classes:
                    return idx

        if cl == 33 and cl in classes:
            for idx in neckline_wrap_order:
                if idx in classes:
                    return idx

        for idx, class_wrap in enumerate(self.classes_wrap):
            if cl in class_wrap and self.classes_wrap[idx][0] in classes:
                return idx
        return -1

    def create_mask(self, df: pd.DataFrame):
        num_categories = 45 + 1  # (add 1 for background)

        mask_h = df.loc[:, 'Height'][0]
        mask_w = df.loc[:, 'Width'][0]
        mask = np.full(mask_w * mask_h, num_categories - 1, dtype=np.int32)

        for encode_pixels, encode_labels in zip(df.EncodedPixels.values,
                                                df.ClassId.values):
            pixels = list(map(int, encode_pixels.split(' ')))
            for i in range(0, len(pixels), 2):
                start_pixel = pixels[i] - 1  # index from 0
                len_mask = pixels[i + 1] - 1
                end_pixel = start_pixel + len_mask
                if int(encode_labels) < num_categories - 1:
                    mask[start_pixel:end_pixel] = self.get_real_class(
                        int(encode_labels),
                        df.ClassId.values
                    )

        mask = mask.reshape((mask_h, mask_w), order='F')
        return mask

    def __len__(self):
        return len(self.masks_names)

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, (target, masks, num_crowds))
        """
        base_name = self.masks_names[idx]

        image_full_path = os.path.join(
            self.images_path,
            '{}.jpg'.format(base_name)
        )
        # mask_full_path = os.path.join(
        #     self.masks_path,
        #     '{}.png'.format(base_name)
        # )

        image = cv2.imread(image_full_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError('Can\'t open image: {}'.format(image_full_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # mask = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
        # if mask is None:
        #     raise RuntimeError('Can\'t open mask: {}'.format(mask_full_path))

        df = self.train_csv.loc[base_name]
        if "Series" in str(type(df)):
            df = pd.DataFrame(
                [df.to_list()],
                columns=[
                    'EncodedPixels',
                    'Height',
                    'Width',
                    'ClassId',
                    'AttributesIds'
                ]
            )

        try:
            mask = self.create_mask(df)
        except Exception as e:
            print(e)
            raise RuntimeError('Can\' build mask for: {}'.format(base_name))

        indexes = set(
            [
                ind
                for ind in set(df.ClassId.to_list())
                if ind < 27
            ]
        )

        if len(indexes) > 0:
            masks = [
                (mask == class_index).astype(np.uint8) * 255
                for class_index in indexes
            ]
        else:
            # masks = [np.array((image.shape[0], image.shape[1]), dtype=np.uint8)]
            # indexes = [28]
            masks = []
            indexes = []
            print('AAAAAAAAAAAAAAAAAAAAAAAAA empty mask!')

        reshape = False
        if max(image.shape[:2]) > self.max_size:
            resize_coeff = self.max_size / max(image.shape[:2])
            reshape = True

        for k in range(len(masks)):
            assert masks[k].shape[:2] == image.shape[:2]

            if reshape:
                masks[k] = cv2.resize(
                    masks[k],
                    None,
                    fx=resize_coeff,
                    fy=resize_coeff,
                    interpolation=cv2.INTER_AREA
                )

        if reshape:
            image = cv2.resize(
                image,
                None,
                fx=resize_coeff,
                fy=resize_coeff,
                interpolation=cv2.INTER_AREA
            )

        if self.augmentations is not None:
            image, masks = self.augmentations(image, masks)
            masks = [m for m in masks]

        for k in range(len(masks)):
            assert masks[k].shape[:2] == image.shape[:2]

            masks[k] = create_square_crop_by_detection(
                masks[k],
                [0, 0, *masks[k].shape[:2][::-1]],
                zero_pad=True
            )

            masks[k] = cv2.resize(
                masks[k],
                self.shape,
                interpolation=cv2.INTER_AREA
            )

            _, masks[k] = cv2.threshold(masks[k], 127, 255, cv2.THRESH_BINARY)

        image = create_square_crop_by_detection(
            image,
            [0, 0, *image.shape[:2][::-1]],
            zero_pad=True
        )

        image = cv2.resize(
            image,
            self.shape,
            interpolation=cv2.INTER_AREA
        )

        num_crowds = 0
        im = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
        indexes = list(indexes)

        boxes = []
        res_masks = []
        res_indexes = []
        for i, m in enumerate(masks):
            x, y, w, h = cv2.boundingRect(m)
            x2, y2 = x + w, y + h

            if w*h < 5:
                # print('Skipped box {}'.format([x, y, x2, y2]))
                continue

            boxes.append(
                [
                    x / self.shape[0],
                    y / self.shape[1],
                    x2 / self.shape[0],
                    y2 / self.shape[1]
                ]
            )
            res_masks.append(masks[i])
            res_indexes.append(indexes[i])

        boxes = np.array(boxes)
        res_indexes = np.array(res_indexes) - 1 # Indexes start with 0

        # print('INDEX: {}'.format(indexes))
        # print('INDEX SHAPE: {}'.format(np.expand_dims(indexes, axis=1).shape))

        # print(boxes, np.expand_dims(res_indexes, axis=1))

        gt = torch.FloatTensor(np.hstack((boxes, np.expand_dims(res_indexes, axis=1))))
        masks = torch.FloatTensor(res_masks) / 255.0

        # print(
        #     {
        #         'gt': gt,
        #         'masks shape': masks.shape,
        #         'img shape': im.shape,
        #         'base name': base_name
        #     }
        # )
        return im, (gt, masks, num_crowds)


if __name__ == '__main__':
    masks_colors = [
        (255, 76, 148),
        (76, 255, 183),
        (76, 148, 255),
        (255, 183, 76)
    ]

    vis_masks = False

    dataset = iMaterialisticFashionDataset(
        root_path='./iMaterialistFashion/',
        for_train=True,
        augmentations=True
    )

    iwname = 'Image'
    cv2.namedWindow(iwname, cv2.WINDOW_NORMAL)

    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        if vis_masks:
            masks_names = ['Mask {}'.format(i) for i in range(len(sample[1][1]))]
            for m in masks_names:
                cv2.namedWindow(m, cv2.WINDOW_NORMAL)

        image = (
                sample[0] * 255.0
        ).permute(1, 2, 0).to('cpu').numpy().astype(np.uint8)

        masks = [
            (sample[1][1][i] * 255.0).to('cpu').numpy().astype(np.uint8)
            for i in range(len(sample[1][1]))
        ]

        for i, mask in enumerate(masks):
            class_index = int(sample[1][0][i][-1])
            colorfull_mask = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
            colorfull_mask += np.array(
                masks_colors[class_index % len(masks_colors)],
                dtype=np.uint8
            )

            image[mask > 0] = (
                    image[mask > 0].astype(np.float16) * 0.5 + colorfull_mask[
                [mask > 0]].astype(np.float16) * 0.5
            ).astype(np.uint8)

        for i in range(len(masks)):
            class_index = int(sample[1][0][i][-1])
            box = sample[1][0][i][:4].to('cpu').numpy() * [
                *image.shape[:2][::-1],
                *image.shape[:2][::-1]
            ]
            box = box.astype(np.int32)

            image = cv2.rectangle(
                image,
                tuple(box[:2]),
                tuple(box[2:]),
                masks_colors[class_index % len(masks_colors)],
                4
            )

        cv2.imshow(iwname, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if vis_masks:
            for i in range(len(masks)):
                cv2.imshow(masks_names[i], masks[i])

        k = cv2.waitKey(0)
        if vis_masks:
            for m in masks_names:
                cv2.destroyWindow(m)

        if k == 27:
            break

    cv2.destroyWindow(iwname)
