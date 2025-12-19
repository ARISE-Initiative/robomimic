# Base image with Python 3.9 and Linux
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH \
    MUJOCO_GL=osmesa

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    cmake \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libosmesa6-dev \
    libglfw3-dev \
    patchelf && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
# RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
#     bash /tmp/miniconda.sh -b -p /opt/conda && \
#     rm /tmp/miniconda.sh && \
#     conda clean -afy

RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.2-0-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bf -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda clean -ya

# Create and activate robomimic conda environment with Python 3.9
RUN /opt/conda/bin/conda create -n robomimic_venv python=3.9 -y

# Install PyTorch and torchvision with CPU fallback
RUN /opt/conda/bin/conda run -n robomimic_venv conda install -y pytorch==2.0.0 torchvision==0.15.0 cpuonly -c pytorch || \
    /opt/conda/bin/conda run -n robomimic_venv pip install torch==2.0.0+cpu torchvision==0.15.0+cpu

# Install robomimic from source
WORKDIR /opt
RUN git clone https://github.com/ARISE-Initiative/robomimic.git && \
    /opt/conda/bin/conda run -n robomimic_venv pip install -e ./robomimic

# 1. Upgrade pip/setuptools (Critical to find the binary wheel)
RUN /opt/conda/bin/conda run -n robomimic_venv pip install --upgrade pip setuptools wheel

# 2. Pin Numpy < 2.0 (Pro tip: Numpy 2.0 just released and breaks many robotics libraries)
RUN /opt/conda/bin/conda run -n robomimic_venv pip install "numpy<2.0"

# 2.5 FORCE install MuJoCo binary (This fixes the build error!)
# We use --only-binary to tell pip: "Do not try to compile this, if you can't find a wheel, fail."
RUN /opt/conda/bin/conda run -n robomimic_venv pip install --only-binary=mujoco "mujoco>=3.3.0"

# 3. NOW install robosuite
RUN git clone https://github.com/ARISE-Initiative/robosuite.git && \
    cd robosuite && \
    /opt/conda/bin/conda run -n robomimic_venv pip install -r requirements.txt

# Optional: Install robomimic documentation dependencies
WORKDIR /opt/robomimic
RUN /opt/conda/bin/conda run -n robomimic_venv pip install -r requirements-docs.txt

# Set the working directory
WORKDIR /workspace

# Activate Conda environment and start bash when container starts
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate robomimic_venv && bash"]
