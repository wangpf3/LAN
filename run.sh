#!/usr/bin/env bash

source $1

mkdir -p $save_dir

nohup python -u ./code/main.py \
    -S $save_dir \
    -D "./data/${data_dir}" \
    --use_relation $use_relation \
    --aggregate_type $aggregate_type \
    --score_function $score_function\
    --loss_function $loss_function \
    --margin $margin \
    --weight_decay $weight_decay \
    --corrupt_mode $corrupt_mode \
    --max_neighbor $max_neighbor \
    --embedding_dim $embedding_dim \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --num_epoch $num_epoch \
    --epoch_per_checkpoint $epoch_per_checkpoint \
    --gpu_device $gpu_device \
    --gpu_fraction $gpu_fraction \
    > $run_log 2>&1 &
