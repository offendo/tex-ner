#!/usr/bin/env sh

# Login
huggingface-cli login --token $(cat /etc/api-tokens/hf-token)
wandb login $(cat /etc/api-tokens/wandb-token)

# Run training
python src/ner_training/main.py train \
    --model FacebookAI/roberta-base \
    --definition --theorem --proof --example --name \
    --epochs 25 \
    --data_dir /volume/ner/ \
    --output_dir /volume/ner/outputs/ \

# Run testing
python src/ner_training/main.py test \
    --model FacebookAI/roberta-base \
    --checkpoint /volume/ner/outputs/checkpoint-last \
    --definition --theorem --proof --example \
    --data_dir /volume/ner \
    --output_file /volume/ner/outputs/preds.json
