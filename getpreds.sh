#!/bin/zsh

if [[ -z $NAME ]]; then
  echo "Error: please set the \$NAME variable to the run name."
  exit 1
fi

for SPLIT in test val; do
  kubectl cp nilay-pod:/volume/ner/outputs/$NAME/$SPLIT.preds.json results/$NAME.$SPLIT.preds.json;
done

exec 3<<EOF 
import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.metrics import precision_recall_fscore_support as prfs

softmax = lambda x: np.exp(x)/np.exp(x).sum(axis=-1).reshape(-1, 1)

df = pd.concat([pd.read_json(f'results/$NAME.{split}.preds.json') for split in ['test', 'val']])
df['probs'] = df.logits.apply(lambda x: softmax(np.stack([np.array(y) for y in x], axis=0)).tolist())
df = df[['labels', 'preds', 'tokens', 'probs']].explode(['labels', 'preds', 'tokens', 'probs'])
df.to_csv('outputs.csv', sep='\t', index=False, float_format=lambda x: '%.2f')

p, r, f, s = prfs(df.labels, df.preds, average=None, labels=[l for l in df.labels.unique() if l != 'O'] )
metrics = {}
for i, l in enumerate([l for l in df.labels.unique() if l != 'O']):
  metrics[f"{l}_precision"] = p[i]
  metrics[f"{l}_recall"] = r[i]
  metrics[f"{l}_f1"] = f[i]

pprint(metrics)
EOF

python /dev/fd/3

column -t -s $'\t' outputs.csv > outputs.pretty.txt
rm outputs.csv

