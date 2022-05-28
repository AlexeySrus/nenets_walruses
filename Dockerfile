FROM nvidia/cuda:11.3-cudnn8-runtime-ubuntu20.04

RUN apt-get update && \
    apt-get install --no-install-recommends -y curl && \
    apt-get install --no-install-recommends -y screen && \
    apt-get install --no-install-recommends -y unzip && \
    apt-get install --no-install-recommends -y xz-utils && \
    apt-get install --no-install-recommends -y python3-dev && \
    apt-get install --no-install-recommends -y python3-pip
RUN pip3 install --upgrade pip


RUN pip3 install torch==1.10 torchvision==0.11 torchaudio==0.10 --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip3 install openmim
RUN mim install mmdet

COPY . /nenets_walsuses
WORKDIR /nenets_walsuses/
ENV PYTHONPATH=/nenets_walsuses

