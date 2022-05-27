from argparse import ArgumentParser
import os
import cv2
from PIL import Image
import numpy as np
import json
from multiprocessing import Pool
from tqdm import tqdm
from shapely.geometry import Polygon


classes_names = [
    'walrus'
]

CROP_SIZE = 700
FILTER_COEFF = 0.2


def imap_unordered_bar(func, args, n_processes=8):
    p = Pool(n_processes)
    res_list = []
    with tqdm(total=len(args)) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list


def init_coco_dict() -> dict:
    return {
        "info": {
            "description": "COCO 2017 Dataset",
            "url": "http://cocodataset.org",
            "version": "1.0",
            "year": 2017,
            "contributor": "COCO Consortium",
            "date_created": "2017/09/01"
        },
        "licenses": [
            {
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"
            },
            {
                "url": "http://creativecommons.org/licenses/by-nc/2.0/",
                "id": 2,
                "name": "Attribution-NonCommercial License"
            },
            {
                "url": "http://creativecommons.org/licenses/by-nc-nd/2.0/",
                "id": 3,
                "name": "Attribution-NonCommercial-NoDerivs License"
            },
            {
                "url": "http://creativecommons.org/licenses/by/2.0/",
                "id": 4,
                "name": "Attribution License"
            },
            {
                "url": "http://creativecommons.org/licenses/by-sa/2.0/",
                "id": 5,
                "name": "Attribution-ShareAlike License"
            },
            {
                "url": "http://creativecommons.org/licenses/by-nd/2.0/",
                "id": 6,
                "name": "Attribution-NoDerivs License"
            },
            {
                "url": "http://flickr.com/commons/usage/",
                "id": 7,
                "name": "No known copyright restrictions"
            },
            {
                "url": "http://www.usa.gov/copyright.shtml",
                "id": 8,
                "name": "United States Government Work"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }


def get_image_description(idx: int, image: Image) -> dict:
    return {
        "license": 1,
        "file_name": '{}.jpg'.format(idx),
        "coco_url": '',
        "height": image.size[1],
        "width": image.size[0],
        "date_captured": '',
        "flickr_url": '',
        "id": idx
    }


def linear_separate_mask(mask: np.ndarray) -> list:
    """
    Separate mask to connected divided components
    Args:
        mask: mask in HW uint8 format

    Returns:
        Sorted by mask area list with separate masks
    """
    num_labels, labels_mask = cv2.connectedComponents(mask)

    sep_masks = [
        (labels_mask == i).astype(np.uint8) * 255
        for i in range(1, num_labels)
    ]

    sep_masks.sort(key=lambda m: (m // 255.0).sum(), reverse=True)

    return sep_masks


def worker_mask_generation_function(worker_arg):
    global classes_names

    worker_idx, arg = worker_arg
    res = []

    loop_generator = tqdm(arg) if worker_idx == 0 else arg

    for arg_sample in loop_generator:
        idx, image_path, masks_json_path, save_image_path = arg_sample

        image = Image.open(image_path).convert('RGB')
        with open(masks_json_path, 'r') as jf:
            markup_data = json.load(jf)

        result = []

        crop_x1 = np.random.randint(0, image.size[0] - CROP_SIZE)
        crop_y1 = np.random.randint(0, image.size[1] - CROP_SIZE)
        crop_x2 = crop_x1 + CROP_SIZE
        crop_y2 = crop_y1 + CROP_SIZE

        image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        poly_areas = []

        for i, m in enumerate(markup_data):
            poly_points = m['segmentation_poly']
            if len(poly_points) == 1:
                poly_points = poly_points[0]

            if len(poly_points) < 3 * 2:
                continue

            poly_areas.append(
                Polygon(np.array(poly_points).reshape(-1, 2)).area
            )

        if len(poly_areas) > 0:
            poly_avg_area = np.mean(poly_areas)
        else:
            poly_avg_area = None

        for i, m in enumerate(markup_data):
            target_class = 0
            # wrapped_mask = Mask(masks[i])
            poly_points = m['segmentation_poly']
            if len(poly_points) == 1:
                poly_points = poly_points[0]

            if len(poly_points) < 3 * 2:
                # print(masks_json_path)
                # print(m)
                continue

            poly_points = np.array(poly_points).reshape(-1, 2) - [crop_x1, crop_y1]
            poly_points_x = np.clip(poly_points[:, 0], 0, CROP_SIZE)
            poly_points_y = np.clip(poly_points[:, 1], 0, CROP_SIZE)

            poly_points = np.stack((poly_points_x, poly_points_y), axis=1)

            poly = Polygon(poly_points)

            area = poly.area

            if area <= 1E-5:
                continue

            if area - poly_avg_area * FILTER_COEFF < -1E-5:
                continue
            # segmentation_data = wra
            # pped_mask.polygons().segmentation
            # box = cv2.boundingRect(m)
            box = cv2.boundingRect(np.expand_dims(poly_points.astype(np.float32), 1))

            # segmentation_data = [
            #     np.array(segm_elem).reshape((-1, 2))[::5].astype(
            #         np.float32).flatten().tolist()
            #     for segm_elem in segmentation_data
            #     if len(segm_elem) >= 7*5
            # ]

            segmentation_data = np.array(m['segmentation_poly'])

            if len(segmentation_data) == 0:
                continue

            # if box[2]*box[3] < 150:
            #     continue

            result.append(
                {
                    'segmentation': [[float(p) for p in poly_points.flatten()]],
                    'bbox': [float(b) for b in box],
                    'area': float(area),
                    'iscrowd': 0,
                    'category_id': int(target_class) + 1,
                    'image_id': idx,
                    'id': -1
                }
            )

        if len(result) == 0:
            # print('Skipped index: {}'.format(idx))
            res.append([idx, None, None])
            continue

        image_description = get_image_description(idx, image)
        image.save(save_image_path)

        res.append([idx, result, image_description])

    return res


def parse_args():
    parser = ArgumentParser(description='DeepFashion2 dataset converter')
    parser.add_argument(
        '--images-folder',
        required=True, type=str
    )
    parser.add_argument(
        '--markup-folder',
        required=True, type=str
    )
    parser.add_argument(
        '--result-coco-path', required=True, type=str,
        help='Path to created dataset root dir in COCO like type.'
    )
    parser.add_argument(
        '--val-part', required=False, type=float, default=0.1
    )
    parser.add_argument(
        '--njobs', required=False, type=int, default=24
    )
    parser.add_argument(
        '--ncrops', required=False, type=int, default=50
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.result_coco_path):
        os.makedirs(args.result_coco_path)

    if not os.path.isdir(os.path.join(args.result_coco_path, 'images/')):
        os.makedirs(os.path.join(args.result_coco_path, 'images/'))

    if not os.path.isdir(os.path.join(args.result_coco_path, 'annotations/')):
        os.makedirs(os.path.join(args.result_coco_path, 'annotations/'))

    dataset_images_names = [
        os.path.splitext(img_name)[0]
        for img_name in os.listdir(args.images_folder)
    ]
    dataset_images_names.sort()
    np.random.shuffle(dataset_images_names)
    dataset_size = len(dataset_images_names)

    n = int(dataset_size * (1 - args.val_part)) * args.ncrops

    train_coco = init_coco_dict()
    val_coco = init_coco_dict()

    common_image_index = 1

    print('############### TRAIN PART ###############')
    print('Configure train data for processing')
    masks_building_tasks_data = []

    for _ in range(n):
        idx = common_image_index
        save_image_path = os.path.join(
            args.result_coco_path,
            'images/',
            '{}.jpg'.format(idx)
        )

        image_path = os.path.join(
            args.images_folder,
            dataset_images_names[(common_image_index - 1) // args.ncrops] + '.jpg'
        )
        masks_path = os.path.join(
            args.markup_folder,
            dataset_images_names[(common_image_index - 1) // args.ncrops] + '.json'
        )

        masks_building_tasks_data.append(
            [
                idx,
                image_path,
                masks_path,
                save_image_path
            ]
        )

        common_image_index += 1

    worker_chunks = [
        l.tolist()
        for l in np.array_split(np.array(masks_building_tasks_data), args.njobs)
    ]

    for i in range(len(worker_chunks)):
        for j in range(len(worker_chunks[i])):
            worker_chunks[i][j][0] = int(worker_chunks[i][j][0])

    print('Configure train data for processing')
    masks_building_tasks_results = imap_unordered_bar(
        worker_mask_generation_function,
        [[i, w] for i, w in enumerate(worker_chunks)],
        args.njobs
    )

    masks_building_tasks_results = [
        sample
        for batch in masks_building_tasks_results
        for sample in batch
    ]

    assert masks_building_tasks_results is not None
    masks_building_tasks_results.sort(key=lambda x: x[0])

    object_index = 1

    print('Building train data')
    for elem in tqdm(masks_building_tasks_results):
        idx, objects_data, image_description = elem
        if objects_data is None:
            continue

        if len(objects_data) == 0:
            continue

        train_coco['images'].append(image_description)

        for obj_elem in objects_data:
            train_coco['annotations'].append(obj_elem)
            train_coco['annotations'][-1]['id'] = object_index

            train_coco['categories'].append(
                {
                    'id': object_index,
                    'name': classes_names[obj_elem['category_id'] - 1],
                    'supercategory': 'animal'
                }
            )

            object_index += 1

    with open(
            os.path.join(
                args.result_coco_path,
                'annotations/',
                'instances_train2017.json'
            ),
            'w'
    ) as jf:
        json.dump(train_coco, jf, indent=4)

    print('Train dataset part saved')

    print('############### VALIDATION PART ###############')
    print('Configure validation data for processing')
    masks_building_tasks_data = []

    for _ in range(dataset_size - n):
        idx = common_image_index

        save_image_path = os.path.join(
            args.result_coco_path,
            'images/',
            '{}.jpg'.format(idx)
        )

        image_path = os.path.join(
            args.images_folder,
            dataset_images_names[(common_image_index - 1) // args.ncrops] + '.jpg'
        )
        masks_path = os.path.join(
            args.markup_folder,
            dataset_images_names[(common_image_index - 1) // args.ncrops] + '.json'
        )

        masks_building_tasks_data.append(
            [
                idx,
                image_path,
                masks_path,
                save_image_path
            ]
        )

        common_image_index += 1

    worker_chunks = [
        l.tolist()
        for l in np.array_split(np.array(masks_building_tasks_data), args.njobs)
    ]

    for i in range(len(worker_chunks)):
        for j in range(len(worker_chunks[i])):
            worker_chunks[i][j][0] = int(worker_chunks[i][j][0])

    print('Configure validation data for processing')
    masks_building_tasks_results = imap_unordered_bar(
        worker_mask_generation_function,
        [[i, w] for i, w in enumerate(worker_chunks)],
        args.njobs
    )

    masks_building_tasks_results = [
        sample
        for batch in masks_building_tasks_results
        for sample in batch
    ]

    assert masks_building_tasks_results is not None
    masks_building_tasks_results.sort(key=lambda x: x[0])

    object_index = 1

    print('Building validation data')
    for elem in tqdm(masks_building_tasks_results):
        idx, objects_data, image_description = elem
        if objects_data is None:
            continue

        if len(objects_data) == 0:
            continue

        val_coco['images'].append(image_description)

        for obj_elem in objects_data:
            val_coco['annotations'].append(obj_elem)
            val_coco['annotations'][-1]['id'] = object_index

            val_coco['categories'].append(
                {
                    'id': object_index,
                    'name': classes_names[obj_elem['category_id'] - 1],
                    'supercategory': 'animal'
                }
            )

            object_index += 1

    with open(
            os.path.join(
                args.result_coco_path,
                'annotations/',
                'instances_val2017.json'
            ),
            'w'
    ) as jf:
        json.dump(val_coco, jf, indent=4)

    print('Validation dataset part saved')


if __name__ == '__main__':
    main()
