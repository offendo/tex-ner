#!/usr/bin/env sh

# Login
huggingface-cli login --token $(cat /etc/api-tokens/hf-token)
wandb login $(cat /etc/api-tokens/wandb-token)

# Update the code
pip install -U .

echo "Beginning run $RUN_NAME"
mkdir -p "/volume/ner/outputs/$RUN_NAME"
mkdir -p "/volume/ner/outputs/mmd-preds/"

# Run testing
python src/ner_training/main.py predict \
    --model FacebookAI/roberta-base \
    --crf \
    --checkpoint /volume/ner/outputs/$RUN_NAME/checkpoint-best \
    $CLASSES \
    --data_dir /volume/pdfocr/mmds/ \
    --output_dir /volume/ner/outputs/mmd-preds-2/

