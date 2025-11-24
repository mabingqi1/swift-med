#!/bin/bash
# show env，查看申请的卡是否正常
nvidia-smi
# conda env, 激活自己的conda，注意conda安装路径
source /yinghepool/miniconda3/etc/profile.d/conda.sh
conda activate /yinghepool/mabingqi/envs/med-swift

# run
cd /yinghepool/mabingqi/ms-swift

nproc_per_node=8
NPROC_PER_NODE=$nproc_per_node \
INPUT_SIZE=336 \
VIDEO_SEGMENTS=70 \
swift sft \
    --custom_register_path med_swift/med_template.py \
    --model /yinghepool/mabingqi/hf_cache/Intern3.5VL-4B \
    --model_type internvl3_5 \
    --template med-InternVL3_5 \
    --dataset /yinghepool/mabingqi/dataset/vlm/head_report/tiantan/series_level/tt79k_ym122k_QwenTemplate_ImpressionOnly-train.jsonl \
    --val_dataset /yinghepool/zhangshuheng/codes/train/our-report/case_list/20251101-tiantan-test-V1.jsonl \
    --output_dir output/HeadReport_tt79k-ym122k_InternVL3.5-4B-lora \
    --num_train_epochs 6 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-4 \
    --freeze_vit false \
    --freeze_llm false \
    --freeze_aligner false \
    --train_type lora \
    --use_rslora true \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --gradient_accumulation_steps $(expr 32 / $nproc_per_node) \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 65536 \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 8 \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --deepspeed zero2 \
    --metric_for_best_model eval_token_acc \
    --greater_is_better true 2>&1 | tee log.txt