#!/usr/bin/env sh

# Login
huggingface-cli login --token $(cat /etc/api-tokens/hf-token)
wandb login $(cat /etc/api-tokens/wandb-token)

pip install -U .

# Determine run name
# RUN_NAME=$(curl https://random-word-api.herokuapp.com/word?number=2 | tr '[,"]' '-' | sed 's/--//g')

echo "Beginning run $RUN_NAME"
mkdir -p "/volume/ner/outputs/$RUN_NAME"

if [[ $JOB_COMPLETION_INDEX == "" ]]; then
  JOB_COMPLETION_INDEX="0"
fi

# Run training
export WANDB_RUN_NAME="$RUN_NAME"
python src/ner_training/main.py --run_tune \
  --model_name_or_path FacebookAI/roberta-base \
  --definition --theorem --proof --example \
  $FLAGS \
  --trials 50 \
  --eval_steps 250 --eval_strategy "steps" \
  --logging_strategy "steps" --logging_steps 10 \
  --load_best_model_at_end True --metric_for_best_model "eval_f1" \
  --optim "adamw_hf" \
  --data_dir /volume/ner/$DATASET --output_dir /volume/ner/outputs/$RUN_NAME \
  --seed $JOB_COMPLETION_INDEX
