FROM ubuntu:22.04

ENV DISPLAY=host.docker.internal:0.0
ENV DEBIAN_FRONTEND=noninteractive

# Update and install dependencies
RUN set -x \
    && apt-get update && apt-get install -y software-properties-common \
    && apt-get install -y \
    vim git openssh-client \
    build-essential vim \
    wget \
    git \
    pkg-config \
    zip \
    zlib1g-dev \
    unzip \
    curl \
    gnupg \
    libavcodec-dev libavformat-dev libswscale-dev libavutil-dev libgl1-mesa-dev libglu1-mesa-dev libx11-dev

# Install Python 3.11
RUN set -x \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get -y update \
    && apt-get install -y python3.11-dev python3-pip python3.11-tk \
    && apt-get -y clean \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Upgrade pip and install Python packages
RUN set -x \
    && pip install --upgrade pip \
    && pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    && pip install jaxlib[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    && pip install flax \
    && pip install tensorflow_cpu \
    && pip install keras-cv \
    && pip install opencv-python \
    && pip install pandas \
    && pip install matplotlib \
    && pip install tqdm \
    && pip install tensorboard \
    && pip install orbax-checkpoint

# Install PyTorch with CPU-only support 
RUN set -x \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN set -x \
    && pip install joblib openexr jaxwt onnx onnxsim onnxruntime scikit-learn

RUN set -x \
    && apt-get install -y ffmpeg jq

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

# Create a directory for JAX configuration
RUN mkdir -p /etc/jax

# Create JAX configuration file
RUN echo '\
import jax\n\
import os\n\
\n\
# Enable 16-bit floating point precision\n\
jax.config.update("jax_enable_x64", False)\n\
\n\
# Enable asynchronous device transfers\n\
jax.config.update("jax_transfer_guard", "allow")\n\
\n\
# Set matrix multiplication precision to bfloat16\n\
jax.config.update("jax_default_matmul_precision", "bfloat16")\n\
\n\
# Ensure JAX uses GPU\n\
os.environ["JAX_PLATFORM_NAME"] = "gpu"\n\
' > /etc/jax/config.py

# Create environment setup script
RUN printf '#!/bin/bash\n\
# JAX GPU Configuration\n\
export XLA_FLAGS="--xla_gpu_enable_fast_min_max=true"\n\
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95\n\
export JAX_PLATFORM_NAME=gpu\n\
export CUDA_DEVICE_ORDER="PCI_BUS_ID"\n\
export PYTHONSTARTUP=/etc/jax/config.py\n\
\n\
# PyTorch CPU Configuration\n\
export TORCH_USE_CUDA_DSA=0\n\
\n\
# Display configuration\n\
echo "JAX GPU and PyTorch CPU environment configured"\n\
echo "JAX devices: $(python -c '"'"'import jax; print(jax.devices())'"'"' 2>/dev/null || echo '"'"'JAX not ready'"'"')"\n\
echo "PyTorch CUDA available: $(python -c '"'"'import torch; print(torch.cuda.is_available())'"'"' 2>/dev/null || echo '"'"'PyTorch not ready'"'"')"\n' > /etc/profile.d/ml_env.sh

RUN chmod +x /etc/profile.d/ml_env.sh

# Source the environment setup in bashrc
RUN echo 'source /etc/profile.d/ml_env.sh' >> /root/.bashrc

RUN set -x \
    && pip install jupyter

ENV QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms
ENV PYTHONPATH="/code"

RUN mkdir /checkpoints/
RUN mkdir /logs/
RUN mkdir /configs/
RUN mkdir /results/
WORKDIR /code
ENTRYPOINT ["/bin/bash"]
