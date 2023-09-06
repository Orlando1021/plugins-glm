LR=1e-5

deepspeed --include localhost:0,1,2,3 --master_port 61001 main.py \
    --do_train \
    --do_eval \
    --train_file datasets/train.json \
    --validation_file datasets/valid.json \
    --prompt_column input \
    --response_column output \
    --overwrite_cache \
    --output_dir output/classification_v1_$LR \
    --overwrite_output_dir \
    --max_source_length 1700 \
    --max_target_length 300 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --logging_dir "./logs/classification_v1_$LR" \
    --save_steps 800 \
    --eval_steps 100 \
    --learning_rate $LR \
    --model_name_or_path /share/models/chatglm-6b \
    --save_strategy "epoch" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --save_total_limit 5 \
    --preprocessing_num_workers 16 \
    --fp16 \
    --deepspeed ds_config/stage1.json \
    --gradient_checkpointing 1 
    # --predict_with_generate \
