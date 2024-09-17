#!/usr/bin/env sh

# Login
huggingface-cli login --token $(cat /etc/api-tokens/hf-token)
wandb login $(cat /etc/api-tokens/wandb-token)

# Run training
python src/ner_training/main.py raytune \
    --model FacebookAI/roberta-base \
    --definition --theorem --proof --example \
    --steps 1000 \
    --data_dir /volume/ner/ \
    --output_dir /volume/ner/outputs/ \
