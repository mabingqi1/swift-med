#!/bin/bash

# show env，查看申请的卡是否正常
nvidia-smi

# install, 安装一些依赖，docker如果存在可以不用安装
# apt-get update
# apt-get install -y libgl1-mesa-glx
# apt-get install -y libglib2.0-dev

# conda env, 激活自己的conda，注意conda安装路径
source /yinghepool/miniconda3/etc/profile.d/conda.sh
conda activate /yinghepool/mabingqi/envs/qwen3vl

# scripts
DIR_SCRIPTS=/yinghepool/mabingqi/ms-swift
export PYTHONPATH=$DIR_SCRIPTS:$PYTHONPATH

# DIR WORK
DIR_WORK=$DIR_SCRIPTS
cd $DIR_WORK

# delete cache files
rm -rf ~/.cache/modelscope/hub/

# RUN
python scripts/infer.py \
    --output_dir /yinghepool/mabingqi/ms-swift/output/HeadReport_tt79k-ym122k-ImpressionOnly_Qwen3-VL-4B-rslora \
    --dir_save /yinghepool/mabingqi/ms-swift/output/HeadReport_tt79k-ym122k-ImpressionOnly_Qwen3-VL-4B-rslora \
    --path_jsonl /yinghepool/mabingqi/dataset/vlm/head_report/tiantan/series_level/tt79k_QwenTemplate_ImpressionOnly-test-raw.jsonl \
    --ckpt_use /yinghepool/mabingqi/ms-swift/output/HeadReport_tt79k-ym122k-ImpressionOnly_Qwen3-VL-4B-rslora/checkpoint-10500 \