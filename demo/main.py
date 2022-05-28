import base64
import yaml
import colorsys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageDraw import ImageDraw
import streamlit as st

from requests_toolbelt import MultipartEncoder
import requests
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'inference_utils'))

from yolact_inference import YOLACTModel
from tiled_segmentation import WindowReadyImage

model = YOLACTModel(device='cpu')


def read_config():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    return config


def get_mask(b64mask, width, height) -> np.ndarray:
    mask = np.frombuffer(
        base64.decodebytes(b64mask.encode('utf-8')), dtype=np.uint8
    )
    mask = mask.reshape(height, width)
    return mask


MOCK_JSON = {
    'boxes': [
        [0.1, 0.1, 0.3, 0.3],
        [0.4, 0.5, 0.8, 0.7],
    ],
    'centers': [
        [0.2, 0.2],
        [0.6, 0.6],
    ]
}
BBOX_WIDTH = 2
CIRCLE_WIDTH = 2
COEF = 0.002




def visualize_results(
        image: Image,
        json_results,
        draw_boxes=False,
        draw_centers=True,
        draw_mask=False
):
    if draw_mask:
        # TODO
        pass
        # mask = get_mask(
        #     json_results['mask']['b64mask'],
        #     json_results['mask']['width'],
        #     json_results['mask']['height']
        # )
        # image = apply_mask(image, mask)
    draw = ImageDraw(image)
    if draw_boxes:
        boxes = json_results['boxes']
        bbox_width = max(BBOX_WIDTH, int(image.width * COEF))

        for bbox in boxes:
            x1, y1, x2, y2 = bbox
            x1 *= image.width
            x2 *= image.width
            y1 *= image.height
            y2 *= image.height
            draw.rectangle(
                (x1, y1, x2, y2),
                fill=None,
                outline='white',
                width=bbox_width
            )
    if draw_centers:
        circle_width = max(CIRCLE_WIDTH, int(image.width * COEF))
        centers = json_results['centers']
        # print(centers)
        for x, y in centers:

            # x *= image.width
            # y *= image.height
            draw.ellipse(
                (x - circle_width, y - circle_width,
                 x + circle_width, y + circle_width),
                fill='red',
            )
    return image


def display_map():
    # TODO
    pass

def save_csv(image: Image, json_data: dict, dst_fname: str):
    result_str = ''
    for x, y in json_data.get('centers', []):
        # x = int(x * image.width)
        # y = int(y * image.height)
        result_str += f'{x},{y}\n'
    with open(dst_fname, 'w') as f:
        f.write('x,y\n' + result_str)


def layout():
    st.sidebar.markdown('---')
    st.sidebar.subheader('Параметры визуализации')
    draw_boxes = st.sidebar.checkbox('Рамки', value=True, disabled=True)
    draw_centers = st.sidebar.checkbox('Центры', value=True, disabled=True)
    # draw_mask = st.sidebar.checkbox('Маска сегментации', value=True)

    image_uploaded = st.file_uploader(
        'Загрузите изображение',
        type=['jpg', 'jpeg', 'png']
    )
    st.subheader('Изображение')
    container = st.empty()
    if image_uploaded is not None:
        # mp_encoder = MultipartEncoder(
        #     fields={
        #         'image': (
        #             image_uploaded.name, image_uploaded, image_uploaded.type
        #         )
        #     }
        # )
        # response = requests.post(url, data=mp_encoder, headers={
        #     'Content-Type': mp_encoder.content_type
        # })
        # json_results = response.json()
        # json_results = MOCK_JSON

        image = Image.open(image_uploaded)
        with st.spinner('Выполняется обработка...'):
            json_results = predict(np.array(image))

        fname = 'results.csv'

        save_csv(image, json_results, fname)

        with container:
            vis_image = visualize_results(
                image, json_results, draw_boxes=False, draw_centers=True, draw_mask=False
            )
            st.image(vis_image)

        st.subheader(f'Найдено {len(json_results["centers"])} моржей.')

        with open(fname, "rb") as f:
            downloaded = st.download_button(
                label="Скачать результаты",
                data=f,
                file_name="results.csv",
                mime="text/csv"
            )


def predict(image: np.ndarray):
    # return {'centers': mock_predict()}
    wri = WindowReadyImage(image, model)
    return {'centers': wri.get_points(), 'boxes': []}


def mock_predict(*args):
    return np.array([
        # [0.1, 0.2],
        # [0.3, 0.3]
        [100, 200],
        [250, 500]
    ])


def main():
    st.set_page_config(
        page_title='Мониторинг популяции ненецких моржей',
        # page_icon='./walrus-icon.png',  # config['icon'],
        layout='wide'
    )
    # print(config)
    st.title('Мониторинг популяции ненецких моржей')
    st.sidebar.image('icon.png')
    # st.sidebar.subheader('Выберите режим работы')
    # radio = st.sidebar.radio(
    #     '', options=['Одно изображение', 'Несколько изображений']
    # )
    layout()


if __name__ == '__main__':
    main()
#     img = cv2.imread('../../DJI_0005 (2).jpg')
#     h, w = img.shape[:2]
#     print(predict(img)[0].shape)
