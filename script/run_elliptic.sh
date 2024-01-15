#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python main_elliptic.py \
    --use_params \
    --param_path params \
    --source_target $1 \
    --backbone $2 \
    --sampling $3 \
    --device 0

