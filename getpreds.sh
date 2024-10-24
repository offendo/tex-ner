#!/bin/zsh

while (( $# )); do
  case $1 in
    --)                 shift; positional+=("${@[@]}"); break  ;;
    -h|--help)          printf "%s\n" $usage && return         ;;
    -n|--name)          shift; NAME=$1                         ;;
    -t|--type)          shift; TYPE=$1                         ;;
    -*)                 opterr $1 && return 2                  ;;
    *)                  positional+=("${@[@]}"); break         ;;
  esac
  shift
done

if [[ -z $NAME ]]; then
  echo "Error: please supply run name with -n | --name."
  exit 1;
fi

mkdir -p results/$NAME

for SPLIT in test val train; do
  if [[ $TYPE != "" ]]; then
    kubectl cp "nilay-pod":/volume/ner/outputs/$NAME/$TYPE.$SPLIT.preds.json results/$NAME/$TYPE.$SPLIT.preds.json;
  else
    kubectl cp "nilay-pod":/volume/ner/outputs/$NAME/$SPLIT.preds.json results/$NAME/default.$SPLIT.preds.json;
  fi
done

if [[ $TYPE = "" ]]; then
  TYPE="default"
fi


exec 3<<EOF 
import pandas as pd
import numpy as np
import os

softmax = lambda x: np.exp(x)/np.exp(x).sum(axis=-1).reshape(-1, 1)

for split in ['test', 'val', 'train']:
  if not os.path.exists(f'results/$NAME/$TYPE.{split}.preds.json'):
    continue
  df = pd.read_json(f'results/$NAME/$TYPE.{split}.preds.json')
  df['probs'] = df.logits.apply(lambda x: softmax(np.stack([np.array(y) for y in x], axis=0)).tolist())
  df['tokens'] = df['tokens'].apply(lambda xs: [x for x in xs if x != '<pad>'])
  df = df[['labels', 'preds', 'tokens', 'probs']].explode(['labels', 'preds', 'tokens', 'probs'])
  df['probs'] = df['probs'].apply(lambda xs: ', '.join([f"{x:0.2f}" for x in xs]))
  df.to_csv(f'{split}.outputs.csv', sep='\t', index=False, float_format=lambda x: '%.2f')

EOF

python /dev/fd/3

rye run python score.py multiclass -n $NAME -t $TYPE

column -t -s $'\t' train.outputs.csv > results/$NAME/$TYPE.train.outputs.txt
column -t -s $'\t' val.outputs.csv > results/$NAME/$TYPE.val.outputs.txt
column -t -s $'\t' test.outputs.csv > results/$NAME/$TYPE.test.outputs.txt


cat \
  results/$NAME/$TYPE.val.outputs.txt \
  results/$NAME/$TYPE.test.outputs.txt \
  > results/$NAME/$TYPE.outputs.txt

rm *.outputs.csv

