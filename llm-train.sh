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

cd Sequence-Labeling-LLMs/ && \
  accelerate launch  --num_processes 4 \
  seq2seq.py \
  --constrained_generation \
  --mixed_precision bf16 \
  --use_lora \
  --train_tsvs /volume/ner/conll/train/*.tsv \
  --dev_tsvs /volume/ner/conll/val/*.tsv \
  --test_tsvs /volume/ner/conll/test/*.tsv \
  --num_beams 4 \
  --num_return_sequences 1 \
  --model_name_or_path $MODEL \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 4 \
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
