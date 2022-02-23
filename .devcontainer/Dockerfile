FROM nvcr.io/nvidia/tensorrt:21.09-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y \
  libssl-dev \
  libpng-dev \
  libjpeg-dev \
  libgl1-mesa-glx \
  git \
  ninja-build \
  python3.8-venv
RUN apt install -y libopencv-dev

RUN pip install \
  torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 \
  -f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN pip install \
  matplotlib \
  opencv-python

ENV TRT_LIB_PATH=”/usr/lib/x86_64-linux-gnu”
ENV TRT_INC_PATH=”/usr/include/x86_64-linux-gnu”
