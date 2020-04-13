#!/usr/bin/env bash
# You can use this script to train a grover advice model.
# RUN THIS SCRIPT LIKE nohup ./finetune.sh mega > finetune_mega_log.txt &

model_type=${1}
max_seq_length=1536
num_tpu_cores=512
batch_size_per_core=1
target_bonus=9.0 # This encourages the model to learn the target, 10x more than the ctx.

####################################
init_checkpoint="gs://turingadvice/baselines/grover/grover_realnews_longsecondarypretrain_jan_1_2020/model=${model_type}~lr=2e-5~epochs=3/model.ckpt-18000"


# Make sure batch size scales.
batch_size=$((${batch_size_per_core} * ${num_tpu_cores}))

num_epochs=20
num_train_steps=$((599971 * ${num_epochs} / ${batch_size}))
save_checkpoint_steps=$(($((num_train_steps + 1)) / 2 ))

num_warmup_steps=$((${num_train_steps} / ${num_epochs} * 2)) # if num_epochs multiplies then adjust accordingly


input_file="REPLACETHISWITHTFRECORDPATH"

# feel free to mess with these
if [ ${model_type} == "base" ]; then
    LRS="2e-5"
elif [ ${model_type} == "large" ]; then
    LRS="1e-5"
elif [ ${model_type} == "mega" ]; then
    LRS="5e-6"
fi

for learning_rate in ${LRS}; do
    OUTPUT_DIR="REPLACETHIS"

    echo "Running for ${num_epochs} epochs, or ${num_train_steps} steps (${num_warmup_steps} warmup) at a batch size of ${batch_size}."
    echo "Storing in ${OUTPUT_DIR}"

    # you can also do this in advance
    # ctpu up --tpu-size=v3-512 --tf-version 1.14 --noconf --tpu-only --preemptible
    python train.py \
        --config_file=configs/${model_type}.json \
        --input_file=${input_file} \
        --output_dir=${OUTPUT_DIR} \
        --max_seq_length=${max_seq_length} \
        --train_batch_size=${batch_size} \
        --learning_rate=${learning_rate} \
        --num_train_steps=${num_train_steps} \
        --num_warmup_steps=${num_warmup_steps} \
        --save_checkpoints_steps=${save_checkpoint_steps} \
        --iterations_per_loop=${save_checkpoint_steps} \
        --use_tpu=True \
        --tpu_name=$(hostname) \
        --num_tpu_cores=${num_tpu_cores} \
        --init_checkpoint=${init_checkpoint} \
        --target_bonus=${target_bonus}
done