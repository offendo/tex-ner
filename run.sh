#!/usr/bin/env sh

# Login
huggingface-cli login --token $(cat /etc/api-tokens/hf-token)
wandb login $(cat /etc/api-tokens/wandb-token)

# Determine run name
RUN_NAME=$(curl https://random-word-api.herokuapp.com/word?number=2 | tr '[,"]' '-' | sed 's/--//g')

echo "Beginning run $RUN_NAME"
mkdir -p "/volume/ner/outputs/$RUN_NAME"

# Run training
export WANDB_RUN_NAME="$RUN_NAME"
python src/ner_training/main.py train \
    --model FacebookAI/roberta-base \
    --definition --theorem --proof --example \
    --steps 1000 \
    --learning_rate 1.5e-5 \
    --batch_size 32 \
    --label_smoothing_factor 0.02 \
    --context_len 2048 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-6 \
    --scheduler "cosine" \
    --data_dir /volume/ner/ \
    --output_dir /volume/ner/outputs/$RUN_NAME \

# Run testing
python src/ner_training/main.py test \
    --model FacebookAI/roberta-base \
    --checkpoint /volume/ner/outputs/$RUN_NAME/checkpoint-final \
    --definition --theorem --proof --example \
    --data_dir /volume/ner \
    --output_dir /volume/ner/outputs/$RUN_NAME
