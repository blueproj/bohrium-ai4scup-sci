# --nnodes 1 --nproc_per_node 4 --master_port 25641

deepspeed --include localhost:0 train_yi1.5.py \
    --deepspeed ds_zero2_no_offload.json \
    --model_name_or_path /bohr/blue-tvh5/v3/ \
    --use_lora true \
    --use_deepspeed true \
    --data_path train_data_task2 \
    --bf16 true \
    --fp16 false \
    --output_dir lora-model/yi-lora-task2-0911 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --learning_rate 3e-4 \
    --logging_steps 1 \
    --model_max_length 10000
    
    # --tf32 False \
    # --report_to "tensorboard" \
    # --save_strategy "steps" \
    # --save_steps 10 \ 
    # --save_steps 1000 \


    ## model
    # <path_to_model>/Qwen1.5-14B-Chat
    # <path_to_model>/Qwen2-7B-Instruct

