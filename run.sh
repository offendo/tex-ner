#!/usr/bin/env bash

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
if [[ -z $NTRIALS ]]; then
  NTRIALS=1
fi

if [[ $JOB_COMPLETION_INDEX != "" ]]; then
  ITER_NAME=$RUN_NAME-$JOB_COMPLETION_INDEX
else
  ITER_NAME=$RUN_NAME
  JOB_COMPLETION_INDEX=0
fi
echo "Beginning run $ITER_NAME"
mkdir -p "/volume/ner/outputs/$ITER_NAME"

# Run training
export WANDB_NAME="$RUN_NAME"
export WANDB_PROJECT="semicrf-experiments"

if [[ $DO_TRAIN = 'true' ]]; then
    python src/ner_training/main.py --run_train \
      --model_name_or_path FacebookAI/roberta-base \
      $CRF \
      $CLASSES \
      $TRAIN_FLAGS \
      --learning_rate $LEARNING_RATE \
      --per_device_train_batch_size $BATCH_SIZE \
      --per_device_eval_batch_size $BATCH_SIZE \
      --eval_steps 250 --eval_strategy "steps" \
      --logging_strategy "steps" --logging_steps 10 \
      --label_smoothing_factor $LABEL_SMOOTHING --warmup_ratio $WARMUP_RATIO --weight_decay $WEIGHT_DECAY --dropout $DROPOUT \
      --load_best_model_at_end True --metric_for_best_model "eval_f1" \
      --optim "adamw_hf" --lr_scheduler_type $SCHEDULER \
      --data_dir /volume/ner/$DATASET --output_dir /volume/ner/outputs/$ITER_NAME \
      --seed $JOB_COMPLETION_INDEX
  fi



# Run testing
if [ -d /volume/ner/outputs/$ITER_NAME/checkpoint-avg ]; then
  python src/ner_training/main.py --run_test \
      --model_name_or_path FacebookAI/roberta-base \
      $CRF \
      --checkpoint /volume/ner/outputs/$ITER_NAME/checkpoint-avg \
      $CLASSES \
      --data_dir /volume/ner/$DATASET \
      --output_dir /volume/ner/outputs/$ITER_NAME \
      --output_name "average" $PREDICT_ON_TRAIN
fi

if [ -d /volume/ner/outputs/$ITER_NAME/checkpoint-best ]; then
  python src/ner_training/main.py --run_test \
      --model_name_or_path FacebookAI/roberta-base \
      $CRF \
      --checkpoint /volume/ner/outputs/$ITER_NAME/checkpoint-best \
      $CLASSES \
      --data_dir /volume/ner/$DATASET \
      --output_dir /volume/ner/outputs/$ITER_NAME \
      --output_name "best" $PREDICT_ON_TRAIN
fi
