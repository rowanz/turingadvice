#!/usr/bin/env bash

# You can use this script to validate a grover advice model (get ppl)
# RUN THIS SCRIPT LIKE nohup ./validate_loop.sh > validate_Loop.txt &
max_seq_length=1536
save_checkpoint_steps=5000
num_tpu_cores=512
batch_size_per_core=1
target_bonus=9.0 # This encourages the model to learn the target, 10x more than the ctx.

####################################
# Make sure batch size scales.
batch_size=$((${batch_size_per_core} * ${num_tpu_cores}))

num_epochs=20
num_train_steps=$((599971 * ${num_epochs} / ${batch_size}))
save_checkpoint_steps=$(($((num_train_steps + 1)) / 2 ))

num_warmup_steps=$((${num_train_steps} / ${num_epochs} * 2)) # if num_epochs multiplies then adjust accordingly

input_file="REPLACETHISWITHTFRECORDPATH"

for model_type in mega large base; do

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
        init_checkpoint=${OUTPUT_DIR}model.ckpt-${num_train_steps}

        echo "Running for ${num_epochs} epochs, or ${num_train_steps} steps (${num_warmup_steps} warmup) at a batch size of ${batch_size}."
        echo "Storing in ${OUTPUT_DIR}"

        # ctpu up --tpu-size=v3-8 --tf-version 1.14 --noconf --tpu-only
        python validate.py \
            --config_file=configs/${model_type}.json \
            --input_file=${input_file} \
            --output_dir=${OUTPUT_DIR} \
            --max_seq_length=${max_seq_length} \
            --batch_size=8 \
            --use_tpu=True \
            --tpu_name=$(hostname) \
            --num_tpu_cores=8 \
            --init_checkpoint=${init_checkpoint} \
            --validation_name="preds.h5"
    done
done