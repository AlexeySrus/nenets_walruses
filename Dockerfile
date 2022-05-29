FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

RUN apt-get update && \
    apt-get install --no-install-recommends -y curl && \
    apt-get install --no-install-recommends -y screen && \
    apt-get install --no-install-recommends -y unzip && \
    apt-get install --no-install-recommends -y xz-utils && \
    apt-get install --no-install-recommends -y python3-dev && \
    apt-get install --no-install-recommends -y python3-pip && \
    apt-get install --no-install-recommends -y git
RUN pip3 install --upgrade pip


RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean

RUN pip3 install torch==1.10 torchvision==0.11 torchaudio==0.10 --extra-index-url https://download.pytorch.org/whl/cu113

# Install MMCV
RUN pip3 install --no-cache-dir --upgrade pip wheel setuptools
RUN pip3 install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html

RUN apt install -y build-essential
RUN apt-get -y install manpages-dev
RUN pip3 install Cython
RUN pip3 install pycocotools
RUN pip3 install mmdet

COPY . /nenets_walsuses
WORKDIR /nenets_walsuses/
ENV PYTHONPATH=/nenets_walsuses

RUN pip3 install -r requirements.txt

RUN pip3 install wldhx.yadisk-direct
RUN curl -L $(yadisk-direct https://disk.yandex.ru/d/B6IScvN4Vdik-Q) -o models_data.zip
RUN unzip models_data.zip
RUN rm models_data.zip

WORKDIR /nenets_walsuses/demo/
EXPOSE 8501
EXPOSE 8001
CMD screen -dmS StreamLit bash -c "streamlit run main.py --server.port=8501";uvicorn server:app --reload --port 8001