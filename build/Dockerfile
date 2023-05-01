FROM huggingface/transformers-pytorch-gpu:4.23.1

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

SHELL ["/bin/bash", "-c"]

WORKDIR /workspace

# UEC Proxy
ARG HTTP_PROXY
ENV http_proxy ${HTTP_PROXY}
ENV https_proxy ${HTTP_PROXY}

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y \
    vim \
    git \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev 

RUN git config --global http.proxy http://proxy.cc.uec.ac.jp:8080 \
    && git config --global url."https://".insteadOf git:// 

COPY source/pytorch_lightning_training_script/requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip 
RUN pip install allennlp==0.9.0 pytorch-lightning==1.0.3\
    && pip install overrides==3.1.0

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES utility,compute
