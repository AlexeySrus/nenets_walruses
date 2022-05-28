FROM nvidia/cuda:11.3-cudnn8-runtime-ubuntu20.04

RUN apt-get update && \
    apt-get install --no-install-recommends -y curl && \
    apt-get install --no-install-recommends -y screen && \
    apt-get install --no-install-recommends -y unzip && \
    apt-get install --no-install-recommends -y xz-utils && \
    apt-get install --no-install-recommends -y python3-dev && \
    apt-get install --no-install-recommends -y python3-pip
RUN pip3 install --upgrade pip


