#!/usr/bin/env bash

# This is the script I used to finetune T5
# RUN THIS SCRIPT LIKE nohup ./finetune.sh 11B > finetune_11B_log.txt &

####################################
model_type=${1}

if [ ${model_type} == "small" ]; then
    batch_size_per_8core=4
    num_tpu_cores=256
    model_parallelism=4
elif [ ${model_type} == "base" ]; then
    batch_size_per_8core=4
    num_tpu_cores=256
    model_parallelism=4
elif [ ${model_type} == "large" ]; then
    batch_size_per_8core=2
    num_tpu_cores=512
    model_parallelism=8
elif [ ${model_type} == "3B" ]; then
    batch_size_per_8core=2
    num_tpu_cores=512
    model_parallelism=8

    # TURBO MODE
    batch_size_per_8core=1
    num_tpu_cores=1024
    model_parallelism=8
elif [ ${model_type} == "11B" ]; then
    batch_size_per_8core=1
    num_tpu_cores=1024
    model_parallelism=32
fi

# wtf
# https://github.com/google-research/text-to-text-transfer-transformer/issues/34
# https://cloud.google.com/tpu/docs/system-architecture
if [ ${num_tpu_cores} == "8" ]; then
    tpu_topology="2x2"
elif [ ${num_tpu_cores} == "32" ]; then
    tpu_topology="4x4"
elif [ ${num_tpu_cores} == "64" ]; then
    tpu_topology="4x8"
elif [ ${num_tpu_cores} == "128" ]; then
    tpu_topology="8x8"
elif [ ${num_tpu_cores} == "256" ]; then
    tpu_topology="8x16"
elif [ ${num_tpu_cores} == "512" ]; then
    tpu_topology="16x16"
elif [ ${num_tpu_cores} == "1024" ]; then
    tpu_topology="16x32"
fi


# Make sure batch size scales.
batch_size=$((${batch_size_per_8core} * ${num_tpu_cores} / 8))

# T5 tends to overfit fast. my best configurations used these configs (everything with bsize=128)
# 11B: 2.13 epochs (10k steps), lr=0.001  PPL=10.742
# 3B: 4 epochs (18k steps),     lr=0.002  PPL=11.248
# large: 8 epochs (38k steps),  lr=0.002  PPL=12.671
# base: 10 epochs (46k steps),  lr=0.002  PPL=14.855
# small: 8 epochs (38k steps),  lr=0.001. PPL=23.673
# I tried other learning rates, like 0.0004 and 0.0002 but they didn't work as well. ymmv


num_epochs=5
num_train_steps=$((599971 * ${num_epochs} / ${batch_size}))
save_checkpoint_steps=$(($((num_train_steps + 1)) / 10 ))

# you can adjust the learning rates if you want
for learning_rate in 0.001 0.002; do
    OUTPUT_DIR="gs://seri2021-advice/turingadvice/reproduction/${model_type}"

    echo "Running for ${num_epochs} epochs, or ${num_train_steps} steps at a batch size of ${batch_size}."
    echo "Storing in ${OUTPUT_DIR}"

    python train.py \
        --model_size=${model_type} \
        --model_dir=${OUTPUT_DIR} \
        --train_batch_size=${batch_size} \
        --learning_rate=${learning_rate} \
        --num_train_steps=${num_train_steps} \
        --save_checkpoints_steps=${save_checkpoint_steps} \
        --iterations_per_loop=${save_checkpoint_steps} \
        --model_parallelism=${model_parallelism} \
        --tpu_topology=${tpu_topology}
done