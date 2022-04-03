#!/bin/bash -i

source ./environment

python train_myadaptor.py \
  --data_dir $DATA_DIR \
  --model_dir_root $MODEL_ROOT_DIR \
  "$@"
