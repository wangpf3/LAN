#!/usr/bin/env bash

data_dir="fb15K/head-10"
aggregate_type='lstm'
iter_routing=0
use_relation=0
score_function='TransE'
n_neg=1
loss_function='margin'
margin=1.0
weight_decay=0
corrupt_mode='partial'
max_neighbor=64
embedding_dim=100
batch_size=1024
learning_rate=1e-3
num_epoch=1000
epoch_per_checkpoint=50
gpu_device="0"
gpu_fraction="0.2"
hparam="w${weight_decay}_${score_function}_${loss_function}${margin}_corrupt-${corrupt_mode}${n_neg}_e${embedding_dim}r${use_relation}_n${max_neighbor}_b${batch_size}_lr${learning_rate}"
save_dir="./checkpoints/${data_dir}/${aggregate_type}/${hparam}/"
run_log="${save_dir}run.log"
