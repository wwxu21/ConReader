#!/bin/bash

hostname
nvidia-smi
task=ConReader
for i in {3,}
do
CUDA_VISIBLE_DEVICES=1 python $task/train.py \
        --output_dir ./saved_models/ConReader-CD-base\
        --model_type roberta \
        --cache_dir ./cache\
        --model_name roberta-base \
        --data_path ./Data/3full \
        --do_train --do_eval \
        --version_2_with_negative \
        --learning_rate 5e-5 \
        --num_train_epochs 4  \
        --per_gpu_eval_batch_size=40  \
        --per_gpu_train_batch_size=16 \
        --max_seq_length 512 \
        --max_answer_length 512 \
        --doc_stride 256 \
        --save_steps 0 --logging_steps 0\
        --n_best_size 10 --reserved 30 --worker 8 --warmup_rate 0.06 --fp16\
        --overwrite_output_dir --evaluate_during_training --max_query_length 256 --CD
done
