import base64
import yaml
import colorsys
import warnings

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageDraw import ImageDraw
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from models import create_df, get_path, get_points, WalrusCoord

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
POLY_WIDTH = 1
COEF = 0.002


def visualize_results(
        image: Image,
        json_results,
        draw_centers,
        draw_polygons
):
    draw = ImageDraw(image)
    if draw_centers and 'centers' in json_results:
        circle_width = max(CIRCLE_WIDTH, int(image.width * COEF))
        centers = json_results['centers']
        if 'classes' in json_results:
            classes = json_results['classes']
        else:
            classes = [False] * len(centers)
        for (x, y), is_young in zip(centers, classes):
            draw.ellipse(
                (x - circle_width, y - circle_width,
                 x + circle_width, y + circle_width),
                fill='green' if is_young else 'red',
            )
    if draw_polygons and 'polygons' in json_results:
        polygons = json_results['polygons']
        if 'classes' in json_results:
            classes = json_results['classes']
        else:
            classes = [False] * len(polygons)
        width = max(POLY_WIDTH, int(image.width * COEF))
        for poly, is_young in zip(polygons, classes):
            draw.polygon(
                poly,
                outline='green' if is_young else 'red',
                width=width
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


from typing import List


def _prepare_coords(coords: List[WalrusCoord]) -> dict:
    result = {
        'centers': [],
        'classes': []
    }
    for c in coords:
        result['centers'].append((c.x, c.y))
        result['classes'].append(c.is_young)
    return result


def show_historic_data():
    data = create_df()
    gb = GridOptionsBuilder.from_dataframe(data)
    gb.configure_selection(use_checkbox=True)
    gridOptions = gb.build()
    grid_response = AgGrid(
        data,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT',
        update_mode='MODEL_CHANGED',
        fit_columns_on_grid_load=False,
        theme='light',  # Add theme color to the table
        enable_enterprise_modules=True,
        # height=350,
        width='100%',
        reload_data=False  # True
    )
    load_btn = st.button('Загрузить')
    image = None
    container = st.empty()
    path = ''
    coords = []
    selected = grid_response['selected_rows']
    if len(selected) > 0:
        path = get_path(selected[0]['id'])

    if load_btn and path != '':
        print(path)
        image = Image.open(path)
        coords = get_points(selected[0]['id'])
        st.text(f'Кол-во моржей на фото: {len(coords)}')
    with container:
        if image is not None:
            coords_json = _prepare_coords(coords)
            image = visualize_results(
                image,
                coords_json,
                draw_centers=True,
                draw_polygons=False
            )
            st.image(image)


def layout():
    st.sidebar.markdown('---')
    # st.sidebar.subheader('Параметры визуализации')
    # draw_poly = st.sidebar.checkbox('Объекты', value=True, disabled=False)
    # draw_centers = st.sidebar.checkbox('Центры', value=True, disabled=False)

    image_uploaded = st.file_uploader(
        'Загрузите изображение',
        type=['jpg', 'jpeg', 'png']
    )
    st.subheader('Изображение')
    container = st.empty()
    if image_uploaded is not None:
        # Image.open(image_uploaded).save(image_uploaded.name)
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

        with st.spinner('Выполняется обработка...'):
            image, json_results = process_image(image_uploaded)

        fname = 'results.csv'

        save_csv(image, json_results, fname)

        with container:
            vis_image = visualize_results(
                image, json_results, draw_centers=True, draw_polygons=True
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



def process_image(image_uploaded: str):
    image = Image.open(image_uploaded)
    json_results = predict(np.array(image))
    return image, json_results


def predict(image: np.ndarray):
    wri = WindowReadyImage(image, model)
    polygons = [
        np.array(det.poly.exterior.xy).T.astype(int).ravel().tolist()
        for det in wri.detections
    ]

    return {
        'centers': wri.get_points(),
        'boxes': [],
        'polygons': polygons
    }


def mock_predict(*args):
    return np.array([
        [100, 200],
        [250, 500]
    ])


def main():
    st.set_page_config(
        page_title='Мониторинг популяции ненецких моржей',
        layout='wide'
    )
    st.title('Мониторинг популяции ненецких моржей')
    st.sidebar.image('icon.png')
    # st.sidebar.subheader('Выберите режим работы')
    # radio = st.sidebar.radio(
    #     '', options=['Обработать изображение', 'Исторические данные']
    # )
    # if radio == 'Обработать изображение':
    layout()
    # else:
    #     show_historic_data()


if __name__ == '__main__':
    main()
