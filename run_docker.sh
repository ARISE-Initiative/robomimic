#!/bin/bash

# The "Ultimate" Robomimic Docker Command
# Includes: GPU, GUI, Bind Mount, and Editable Install

docker run -it \
  --gpus all \
  --net=host \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="$(pwd):/app/robomimic" \
  --shm-size=16g \
  --workdir="/app/robomimic" \
  robomimic \
  /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate robomimic_venv && pip install -e . && bash"