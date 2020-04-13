#!/usr/bin/env bash

model_type=${1}
source_ckpt="gs://turingadvice/baselines/grover/${model_type}/model.ckpt-23436"

mkdir ckpt-${model_type}
gsutil cp ${source_ckpt}.data-00000-of-00001 ckpt-${model_type}/model.ckpt.data-00000-of-00001
gsutil cp ${source_ckpt}.index ckpt-${model_type}/model.ckpt.index
gsutil cp ${source_ckpt}.meta ckpt-${model_type}/model.ckpt.meta