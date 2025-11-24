#!/bin/bash
# show env，查看申请的卡是否正常
nvidia-smi
# conda env, 激活自己的conda，注意conda安装路径
source /yinghepool/miniconda3/etc/profile.d/conda.sh
conda activate /yinghepool/zhangshuheng/miniconda3/envs/ym-qwen3vl

# run
cd /yinghepool/mabingqi/ms-swift

nproc_per_node=8
# CUDA_VISIBLE_DEVICES= \
NPROC_PER_NODE=$nproc_per_node \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
VIDEO_MAX_PIXELS=157456 \
VIDEO_MAX_TOKEN_NUM=768 \
IMAGE_MAX_TOKEN_NUM=1024 \
FPS_MAX_FRAMES=64 \
swift sft \
    --model /yinghepool/zhangshuheng/models/Qwen3-VL-4B-Instruct \
    --dataset /yinghepool/mabingqi/dataset/vlm/head_report/tiantan/series_level/tt79k_ym122k_QwenTemplate_ImpressionOnly-train.jsonl \
    --val_dataset /yinghepool/mabingqi/dataset/vlm/head_report/tiantan/series_level/tt79k_QwenTemplate_ImpressionOnly-test.jsonl \
    --output_dir output/HeadReport_tt79k-ym122k-ImpressionOnly_Qwen3-VL-4B-rslora \
    --add_version false \
    --overwrite_output_dir true \
    --torch_dtype bfloat16 \
    --train_type lora \
    --freeze_vit false \
    --freeze_llm false \
    --freeze_aligner false \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --use_rslora true \
    --learning_rate 2e-4 \
    --num_train_epochs 8 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 81920 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --metric_for_best_model eval_token_acc \
    --greater_is_better true \
    --attn_impl flash_attn \
    --deepspeed zero2