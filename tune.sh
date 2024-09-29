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
python src/ner_training/main.py tune \
    --model FacebookAI/roberta-base \
    --crf \
    --name --reference --name_or_ref_only='both' \
    --use_class_weights \
    --steps 1500 \
    --trials 50 \
    --data_dir /volume/ner/roberta-base-refs \
    --output_dir /volume/ner/outputs/$RUN_NAME \
