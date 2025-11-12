#!/bin/bash

# show env，查看申请的卡是否正常
nvidia-smi

# install, 安装一些依赖，docker如果存在可以不用安装
# apt-get update
# apt-get install -y libgl1-mesa-glx
# apt-get install -y libglib2.0-dev

# conda env, 激活自己的conda，注意conda安装路径
source /yinghepool/miniconda3/etc/profile.d/conda.sh
conda activate /yinghepool/mabingqi/envs/med-swift

# scripts
DIR_SCRIPTS=/yinghepool/mabingqi/ms-swift
export PYTHONPATH=$DIR_SCRIPTS:$PYTHONPATH

# DIR WORK
DIR_WORK=$DIR_SCRIPTS
cd $DIR_WORK

# delete cache files
rm -rf ~/.cache/modelscope/hub/

# RUN
python -m scripts.infer \
    --output_dir /yinghepool/mabingqi/ms-swift/output/HeadReport_tt79k_InternVL3.5-4B-lora \
    --dir_save /yinghepool/mabingqi/ms-swift/output/HeadReport_tt79k_InternVL3.5-4B-lora \
    --path_jsonl /yinghepool/mm-data/report/tiantan/20250926-tiantan10w/tiantan_head_7.9w_meta_stdWindow_clean-test.jsonl \
    --ckpt_use /yinghepool/mabingqi/ms-swift/output/HeadReport_tt79k_InternVL3.5-4B-lora/v4-20251110-104717/checkpoint-7000 \