#!/usr/bin/env bash

export PYTHONPATH=/home/rowanz/adviceeval

# RUN THIS SCRIPT LIKE nohup ./finetune.sh 11B > finetune_11B_log.txt &

####################################
model_type=${1}

if [ ${model_type} == "small" ]; then
    batch_size_per_8core=16
    model_parallelism=1
elif [ ${model_type} == "base" ]; then
    batch_size_per_8core=8
    model_parallelism=2
elif [ ${model_type} == "large" ]; then
    batch_size_per_8core=2
    model_parallelism=8
elif [ ${model_type} == "3B" ]; then
    batch_size_per_8core=2
    model_parallelism=8
elif [ ${model_type} == "11B" ]; then
    batch_size_per_8core=1
    model_parallelism=16
fi
batch_size=${batch_size_per_8core}
learning_rate=0.001
num_epochs=10

OUTPUT_DIR="fillthisin"

# you can also do this in advance
# ctpu up --tpu-size=v3-8 --tf-version 1.14 --noconf --tpu-only

python validate.py \
    --model_size=${model_type} \
    --model_dir=${OUTPUT_DIR} \
    --batch_size=${batch_size} \
    --learning_rate=${learning_rate} \
    --model_parallelism=${model_parallelism}