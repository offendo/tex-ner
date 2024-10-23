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

for ((i=1; i<=$NTRIALS; i++)); do

  if [[ $NTRIALS = 1 ]]; then
    ITER_NAME=$RUN_NAME
  else
    ITER_NAME=$RUN_NAME-$i
  fi

  echo "Beginning run $ITER_NAME"
  mkdir -p "/volume/ner/outputs/$ITER_NAME"

  # Run training
  export WANDB_RUN_NAME="$RUN_NAME"
  if [[ $DO_TRAIN = 'true' ]]; then
     python src/ner_training/main.py --run_train \
       --model_name_or_path FacebookAI/roberta-base \
       $CRF \
       $CLASSES \
       $TRAIN_FLAGS \
       --learning_rate $LEARNING_RATE \
       --per_device_train_batch_size $BATCH_SIZE \
       --per_device_eval_batch_size $BATCH_SIZE \
       --eval_steps 500 --eval_strategy "steps" \
       --logging_strategy "steps" --logging_steps 10 \
       --label_smoothing_factor $LABEL_SMOOTHING --warmup_ratio $WARMUP_RATIO --weight_decay $WEIGHT_DECAY --dropout $DROPOUT \
       --optim "adamw_hf" --lr_scheduler_type $SCHEDULER \
       --data_dir /volume/ner/$DATASET --output_dir /volume/ner/outputs/$ITER_NAME
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

done;
