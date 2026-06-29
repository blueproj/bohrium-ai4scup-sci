#!/bin/bash

GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="<path_to_model>/MiniCPM-V-2_6" # or openbmb/MiniCPM-V-2, openbmb/MiniCPM-Llama3-V-2_5
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="train_task3_ext.json"
EVAL_DATA="test_task3.json"
LLM_TYPE="qwen2"
# if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm
#if use openbmb/MiniCPM-Llama3-V-2_5, please set LLM_TYPE=llama3

MODEL_MAX_Length=4096 # if conduct multi-images sft, please set MODEL_MAX_Length=4096

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun $DISTRIBUTED_ARGS finetune.py  \
    --model_name_or_path $MODEL \
    --llm_type $LLM_TYPE \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 true \
    --bf16_full_eval true \
    --fp16 false \
    --fp16_full_eval false \
    --do_train \
    --do_eval \
    --tune_vision true \
    --tune_llm false \
    --use_lora true \
    --lora_target_modules "llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)" \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums 9 \
    --eval_steps 5 \
    --num_train_epochs 2 \
    --output_dir ../task3_lora \
    --logging_dir ../log \
    --logging_strategy "steps" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 5 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed ds_config_zero2.json
    
    # --max_steps 10 \     --save_strategy "epoch" \
    # --q_lora true \