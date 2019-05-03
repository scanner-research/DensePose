FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y rsync htop git openssh-server tmux wget
RUN apt-get install python3-pip -y
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Pytorch (Caffe2 included)
RUN pip3 install torchvision_nightly
RUN pip3 install torch_nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html

# Clone the Detectron repository
RUN git clone -b python3 https://github.com/scanner-research/DensePose.git /densepose

# Install Python dependencies
RUN pip3 install -r /densepose/requirements.txt

# Install the COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi
WORKDIR /cocoapi/PythonAPI
RUN make install

# Go to Densepose root
WORKDIR /densepose

# Set up Python modules
RUN make

# [Optional] Build custom ops
#RUN make ops

# Download UV data
RUN cd DensePoseData && bash get_densepose_uv.sh

# Hacky for opencv
RUN apt-get install libglib2.0-0 libsm6 libxext6 libxrender-dev -y
# Hacky for caffe2
RUN pip3 install protobuf future

