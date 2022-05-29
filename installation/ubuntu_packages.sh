#!/bin/bash

sudo apt-get update && \
    apt-get install --no-install-recommends -y curl && \
    apt-get install --no-install-recommends -y screen && \
    apt-get install --no-install-recommends -y unzip && \
    apt-get install --no-install-recommends -y xz-utils && \
    apt-get install --no-install-recommends -y python3-dev && \
    apt-get install --no-install-recommends -y python3-pip && \
    apt-get install --no-install-recommends -y git
pip3 install --upgrade pip


sudo apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean

pip3 install torch==1.10 torchvision==0.11 torchaudio==0.10 --extra-index-url https://download.pytorch.org/whl/cu113

# Install MMCV
pip3 install --no-cache-dir --upgrade pip wheel setuptools
pip3 install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html

sudo apt install -y build-essential
sudo apt-get -y install manpages-dev
pip3 install Cython
pip3 install pycocotools
pip3 install mmdet

pip3 install -r requirements.txt

pip3 install wldhx.yadisk-direct
curl -L $(yadisk-direct https://disk.yandex.ru/d/B6IScvN4Vdik-Q) -o models_data.zip
unzip models_data.zip
rm models_data.zip
