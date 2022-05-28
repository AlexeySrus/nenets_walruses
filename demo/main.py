import base64
import math
import io
import os
import sys
from typing import List, Dict
import warnings

import cv2
import folium
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw
from PIL.ImageDraw import ImageDraw
from folium.plugins import HeatMap
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_folium import st_folium

from models import (
    create_df, get_path, get_points, WalrusCoord, add_record_in_db,
    create_db_and_tables, check_db_exists
)

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', 'inference_utils')
)

from yolact_inference import YOLACTModel
from young_classifier import YoungWalrusesClassier
from tiled_segmentation import WindowReadyImage


st.set_page_config(
    page_title='Мониторинг популяции ненецких моржей',
    layout='wide'
)


class MockModel:
    """Class that is used as model and has the same format output."""

    def __init__(self):
        """Initialize mock model."""
        bg = np.zeros((100, 100), np.uint8)
        masks = [bg.copy() for i in range(3)]
        masks[0][10:20, 10:20] += 1
        masks[1][30:50, 15:30] += 1
        masks[1][50:60, 70:85] += 1
        self.masks = np.array(masks, dtype=np.uint8)

        self.boxes = np.array([
            [10, 10, 20, 20],
            [25, 10, 55, 30],
            [20, 10, 55, 30]
        ])

        self.classes = [0, 1, 0]

    def __call__(self, image: np.ndarray, *args):
        """Get mock results."""
        h, w = image.shape[:2]
        masks = []
        for mask in self.masks:
            m = mask.copy()
            m = cv2.resize(m, (w, h))
            masks.append(m)
        return masks, self.boxes, self.classes


@st.cache(
    allow_output_mutation=True,
    hash_funcs={torch._C.ScriptModule: lambda _: None}
)
def load_models():
    """Load ML model to process images."""
    model = YOLACTModel()
    # model = MMDetectionQueryInstInference(conf=0.4)
    # model = MockModel()
    yong_clasifier = YoungWalrusesClassier(conf=0.7)
    return model, yong_clasifier


MODEL, YOUNG_CLASSIFIER = load_models()

# Some constants
CIRCLE_WIDTH = 2
POLY_WIDTH = 1
COEF = 0.002


def visualize_results(
        image: Image,
        json_results: Dict,
        draw_centers: bool,
        draw_polygons: bool
):
    """Visualize results on a new Pillow image."""
    image = image.copy()
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


def save_csv(json_data: dict, dst_fname: str):
    """Save centers predicted by ML model into a file."""
    result_str = ''
    for x, y in json_data.get('centers', []):
        result_str += f'{x},{y}\n'
    with open(dst_fname, 'w') as f:
        f.write('x,y\n' + result_str)


def _prepare_coords(coords: List[WalrusCoord]) -> dict:
    result = {
        'centers': [],
        'classes': []
    }
    for c in coords:
        result['centers'].append((c.x, c.y))
        result['classes'].append(c.is_young)
    return result


def predict(image: np.ndarray):
    """Get prediction for the given image."""
    wri = WindowReadyImage(image, MODEL, YOUNG_CLASSIFIER)
    polygons = [
        np.array(det.poly.exterior.xy).T.astype(int).ravel().tolist()
        for det in wri.detections
    ]
    classes = [det.cls == 1 for det in wri.detections]

    return {
        'centers': wri.get_points(),
        'boxes': [],
        'polygons': polygons,
        'classes': classes
    }


# @st.cache(
#     allow_output_mutation=True,
#     hash_funcs={torch._C.ScriptModule: lambda _: None}
# )
@st.experimental_memo(show_spinner=False)
def process_image(image_uploaded: str):
    """Process uploaded image and get prepared results."""
    image = Image.open(image_uploaded)
    json_results = predict(np.array(image))
    return image, json_results


def layout_map():
    """Show page with map."""
    data = create_df()
    start_location = 69.45783, 58.51319
    w, h, fat_wh = 400, 400, 1.3
    m = folium.Map(location=start_location, zoom_start=13)
    heatmap = []
    for i, row in data.iterrows():
        if math.isnan(row['latitude']) or math.isnan(row['longitude']):
            continue
        fpath = get_path(row['id'])
        arr = io.BytesIO()
        image = Image.open(fpath)
        image.thumbnail((w, h))
        image.save(arr, format='JPEG')
        encoded = base64.b64encode(arr.read())
        html = f'<p>Кол-во моржей: {row["walruses_count"]}</p>'
        html += '<img src="data:image/jpg;base64,{}" ' \
                'width="{}" heigth="{}">'.format(encoded.decode('utf-8'), w, h)
        iframe = folium.IFrame(html, width=w * fat_wh, height=h * fat_wh)

        popup = folium.Popup(iframe, parse_html=True, max_width=700)
        folium.Marker(
            (row['latitude'], row['longitude']),
            tooltip="Walruses",
            popup=popup
        ).add_to(m)
        heatmap.append(
            (row['latitude'], row['longitude'], row['walruses_count'])
        )

    HeatMap(heatmap).add_to(m)
    st_folium(m, width=800, height=600)


def layout_historic_data():
    """Show historic data from database"""
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
        theme='streamlit',
        enable_enterprise_modules=True,
        # height=350,
        width='100%',
        reload_data=False
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
        image = Image.open(path)
        coords = get_points(selected[0]['id'])
        st.text(f'Кол-во моржей на фото: {len(coords)}')
        olds = sum(not c.is_young for c in coords)
        youngs = sum(c.is_young for c in coords)
        st.text(f'В том числе:\n- взрослых: {olds}\n- молодых: {youngs}')
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


def layout_process_image():
    """Show layout for image uploadting and processing."""
    st.sidebar.markdown('---')
    st.sidebar.subheader('Параметры визуализации')
    draw_poly = st.sidebar.checkbox('Объекты', value=True, disabled=False)
    draw_centers = st.sidebar.checkbox('Центры', value=True, disabled=False)

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
        save_csv(json_results, fname)

        with container:
            vis_image = visualize_results(
                image, json_results,
                draw_centers=draw_centers, draw_polygons=draw_poly
            )
            st.image(vis_image)

        olds = sum(not c for c in json_results["classes"])
        youngs = sum(c for c in json_results["classes"])
        text_result = f'В том числе:\n- молодых: {youngs}\n- взрослых: {olds}'
        st.subheader(f'Найдено {len(json_results["centers"])} моржей.')
        st.text(text_result)

        with open(fname, "rb") as f:
            st.download_button(
                label="Скачать результаты",
                data=f,
                file_name="results.csv",
                mime="text/csv"
            )
        if st.button('Сохранить в базу данных'):
            add_record_in_db(
                image,
                image_uploaded.name,
                json_results['centers'],
                json_results['classes']
            )


def main():
    """Main function."""
    if not check_db_exists():
        create_db_and_tables()
    st.title('Мониторинг популяции ненецких моржей')
    st.sidebar.image('icon.png')
    st.sidebar.subheader('Выберите режим работы')
    radio = st.sidebar.radio(
        '', options=['Обработать изображение', 'Исторические данные', 'Карта']
    )
    if radio == 'Обработать изображение':
        layout_process_image()
    elif radio == 'Исторические данные':
        layout_historic_data()
    else:
        layout_map()


if __name__ == '__main__':
    main()
