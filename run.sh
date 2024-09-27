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
    --steps 1000 \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --label_smoothing_factor $LABEL_SMOOTHING \
    --warmup_ratio $WARMUP_RATIO \
    --dropout $DROPOUT \
    --weight_decay $WEIGHT_DECAY \
    --scheduler $SCHEDULER \
    --data_dir /volume/ner/$DATASET \
    --output_dir /volume/ner/outputs/$RUN_NAME

# Run testing
python src/ner_training/main.py test \
    --model FacebookAI/roberta-base \
    --crf \
    --checkpoint /volume/ner/outputs/$RUN_NAME/checkpoint-best \
    $CLASSES \
    --data_dir /volume/ner/$DATASET \
    --output_dir /volume/ner/outputs/$RUN_NAME
