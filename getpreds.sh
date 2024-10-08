#!/bin/zsh

while (( $# )); do
  case $1 in
    --)                 shift; positional+=("${@[@]}"); break  ;;
    -h|--help)          printf "%s\n" $usage && return         ;;
    -n|--name)          shift; NAME=$1                     ;;
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

for SPLIT in test val; do
  kubectl cp nilay-pod:/volume/ner/outputs/$NAME/$SPLIT.preds.json results/$NAME/$SPLIT.preds.json;
done

exec 3<<EOF 
import pandas as pd
import numpy as np

softmax = lambda x: np.exp(x)/np.exp(x).sum(axis=-1).reshape(-1, 1)

df = pd.concat([pd.read_json(f'results/$NAME/{split}.preds.json') for split in ['test', 'val']])
df['probs'] = df.logits.apply(lambda x: softmax(np.stack([np.array(y) for y in x], axis=0)).tolist())
df = df[['labels', 'preds', 'tokens', 'probs']].explode(['labels', 'preds', 'tokens', 'probs'])
df['probs'] = df['probs'].apply(lambda xs: ', '.join([f"{x:0.2f}" for x in xs]))
df.to_csv('outputs.csv', sep='\t', index=False, float_format=lambda x: '%.2f')
EOF

python /dev/fd/3


rye run python score.py multiclass -n $NAME

column -t -s $'\t' outputs.csv > results/$NAME/outputs.txt
rm outputs.csv

