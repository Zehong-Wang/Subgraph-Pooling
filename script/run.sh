#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python main.py \
    --use_params \
    --param_path params \
    --freeze \
    --ft_last_layer \
    --ft_whole_model \
    --source_target $1 \
    --backbone $2 \
    --sampling $3 \
    --src_train_ratio 0.6 \
    --train_ratio 0.1 \
    --device 0

