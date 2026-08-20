#!/bin/bash

DATA_PATH="/mnt/NAS/boan/Culturalframes/train_dataset.json"
FINAL_OUTPUT_DIR="/mnt/NAS/boan/output"
PRETRAIN_MODEL="microsoft/Phi-3.5-vision-instruct"

deepspeed train_llava_reward.py \
    --save_path $FINAL_OUTPUT_DIR \
    --pretrain $PRETRAIN_MODEL \
    --dataset $DATA_PATH \
    --dataset_probs 1 \
    --train_split_ratio 0.85 \
    --eval_steps 100 \
    --save_steps -1 \
    --save_best_model 1 \
    --logging_steps 1 \
    --max_epochs 3 \
    --micro_train_batch_size 4 \
    --accumulated_gradient 4 \
    --learning_rate 1e-4 \
    --max_len 2048 \
    --bf16 \
    --flash_attn \
    --gradient_checkpointing \
    --add_cross_attention \
    --ft_projector \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.1 \
    --target_modules all-linear \
    --zero_stage 2 \
    --value_head_dim 1 \
    --group_size 1 \
    --seed 42 \