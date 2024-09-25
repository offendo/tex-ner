#!/usr/bin/env sh

# Login
huggingface-cli login --token $(cat /etc/api-tokens/hf-token)
wandb login $(cat /etc/api-tokens/wandb-token)

# Determine run name
#RUN_NAME=$(curl https://random-word-api.herokuapp.com/word?number=2 | tr '[,"]' '-' | sed 's/--//g')
if [[ -z $RUN_NAME ]]; then
  echo "Error: empty RUN_NAME"
  exit 1
fi

echo "Beginning run $RUN_NAME"
mkdir -p "/volume/ner/outputs/$RUN_NAME"

# Run training
export WANDB_RUN="$RUN_NAME"
python src/ner_training/main.py train \
    --model FacebookAI/roberta-base \
    --crf \
    $CLASSES \
    --steps 1500 \
    --learning_rate 1.4e-5 \
    --batch_size 32 \
    --label_smoothing_factor 0.07 \
    --warmup_ratio 0.05 \
    --dropout 0.2 \
    --weight_decay 6e-4 \
    --scheduler "inverse_sqrt" \
    --data_dir /volume/ner/roberta-base \
    --output_dir /volume/ner/outputs/$RUN_NAME \
    --examples_as_theorems

# Run testing
python src/ner_training/main.py test \
    --model FacebookAI/roberta-base \
    --crf \
    --checkpoint /volume/ner/outputs/$RUN_NAME/checkpoint-best \
    $CLASSES \
    --data_dir /volume/ner/roberta-base \
    --output_dir /volume/ner/outputs/$RUN_NAME \
    --examples_as_theorems
