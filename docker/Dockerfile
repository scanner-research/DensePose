# Use Caffe2 image as parent image
#FROM caffe2/caffe2:snapshot-py2-cuda9.0-cudnn7-ubuntu16.04

#RUN mv /usr/local/caffe2 /usr/local/caffe2_build
#ENV Caffe2_DIR /usr/local/caffe2_build

#ENV PYTHONPATH /usr/local/caffe2_build:${PYTHONPATH}
#ENV LD_LIBRARY_PATH /usr/local/caffe2_build/lib:${LD_LIBRARY_PATH}

FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y rsync htop git openssh-server tmux
RUN apt-get install python3-pip -y
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip3 install torchvision_nightly
RUN pip3 install torch_nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html

# Clone the Detectron repository
#RUN git clone -b py35 https://github.com/scanner-research/DensePose.git /densepose
RUN git clone -b python3 https://github.com/landersson/DensePose.git /densepose

# Install Python dependencies
RUN pip3 install -r /densepose/requirements.txt
RUN pip3 install pyyaml==3.12

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
#RUN apt-get update && apt-get install -y --no-install-recommends wget tmux
#RUN cd DensePoseData && bash get_densepose_uv.sh

# Hacky start
RUN apt-get install libglib2.0-0 libsm6 libxext6 libxrender-dev -y
RUN pip3 install protobuf future

