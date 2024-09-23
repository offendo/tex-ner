#!/usr/bin/env sh

# Login
huggingface-cli login --token $(cat /etc/api-tokens/hf-token)
wandb login $(cat /etc/api-tokens/wandb-token)

echo "Beginning run $RUN_NAME"
mkdir -p "/volume/ner/outputs/$RUN_NAME"

# Run testing
export WANDB_RUN_NAME="$RUN_NAME"
python src/ner_training/main.py test \
    --model FacebookAI/roberta-base \
    --crf \
    --checkpoint /volume/ner/outputs/$RUN_NAME/checkpoint-best \
    --definition --theorem --proof --example \
    --data_dir /volume/ner/roberta-base \
    --output_dir /volume/ner/outputs/$RUN_NAME

