#!/bin/bash

# Login
huggingface-cli login --token $(cat /etc/api-tokens/hf-token)
wandb login $(cat /etc/api-tokens/wandb-token)

# Re-install local code
pip install -U .

# Determine run name
#RUN_NAME=$(curl https://random-word-api.herokuapp.com/word?number=2 | tr '[,"]' '-' | sed 's/--//g')
if [[ -z $RUN_NAME ]]; then
  echo "Error: empty RUN_NAME"
  exit 1
fi

DEVICES=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

cd Sequence-Labeling-LLMs/ && \
  accelerate launch  --num_processes $DEVICES \
  seq2seq.py \
  $CONSTRAINED \
  --mixed_precision bf16 \
  --use_lora \
  --quantization 4 \
  --force_auto_device_map 1 \
  --use_flash_attention \
  --train_tsvs /volume/ner/conll/train/train.tsv \
  --dev_tsvs /volume/ner/conll/val/val.tsv \
  --test_tsvs /volume/ner/conll/test/test.tsv /volume/ner/conll/train/train.tsv /volume/ner/conll/val/val.tsv \
  --num_beams 1 \
  --num_return_sequences 1 \
  --model_name_or_path $MODEL \
  --per_device_train_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACCUMULATION \
  --per_device_eval_batch_size $BATCH_SIZE \
  --learning_rate 2e-5 \
  --optim adamw8bits \
  --lr_scheduler_type cosine \
  --num_warmup_steps $WARMUP_STEPS \
  --num_train_epochs $EPOCHS \
  --eval_every_epoch $EVAL_EVERY_EPOCHS \
  --max_source_length $SEQUENCE_LENGTH \
  --max_target_length $SEQUENCE_LENGTH \
  --output_dir /volume/ner/outputs/$RUN_NAME/ \
  --project_name llm-ner $FLAGS
