#!/bin/bash

accelerate launch seq2seq.py \
  --mixed_precision bf16 \
  --use_lora \
  --train_tsvs /volume/ner/conll/train/*.txt \
  --dev_tsvs /volume/ner/conll/val/*.txt \
  --test_tsvs /volume/ner/conll/test/*.txt \
  --num_beams 4 \
  --num_return_sequences 1 \
  --model_name_or_path $MODEL \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 8 \
  --learning_rate 1e-4 \
  --optim adamw \
  --lr_scheduler_type cosine \
  --num_warmup_steps 400 \
  --num_train_epochs 50 \
  --eval_every_epochs 3 \
  --max_source_length 2048 \
  --max_target_length 2048 \
  --output_dir /volume/ner/outputs/$RUN_NAME/ \
  --project_name llm-ner
