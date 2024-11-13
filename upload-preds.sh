#!/bin/zsh

read -r -d '' usage << EOM
uploads prediction file (*.annos.json) to the annotation tool.

usage: 
  zsh upload-preds.sh -f <file/to/upload>
EOM


while (( $# )); do
  case $1 in
    --)                 shift; positional+=("${@[@]}"); break  ;;
    -h|--help)          printf "%s\n" $usage && return         ;;
    -f|--file)          shift; FILE=$1                         ;;
    -*)                 opterr $1 && return 2                  ;;
    *)                  positional+=("${@[@]}"); break         ;;
  esac
  shift
done

if [[ -z $FILE ]]; then
  echo "Error: please supply file with -f | --file"
  exit 1
fi

FILEID=$(basename <<< echo $FILE | sed 's/.annos.json/.mmd/g')
echo "Uploading $FILEID..."
DATA=$(jq '. | {tag, start, end}' $FILE)

exec 3<<EOF
import requests as r
import json
import math
import pandas as pd
import os
from tqdm import tqdm

data = json.loads('''$DATA''')
df = pd.DataFrame.from_dict(data)

length = df['end'].max()
chunk = int(os.environ.get("CHUNK_SIZE", length))
n_chunks = math.ceil(length / chunk)

if n_chunks == 1:
    resp = r.post(f"https://annotate.nilay.page/api/predictions?fileid=$FILEID&savename=ai-test-preds", json={'annotations': df.to_dict()})
else:
    for n in tqdm(range(n_chunks), total=n_chunks):
        chunk_data = df[(df['end'] >= chunk * n) & (df['end'] < chunk * (n+1))]
        resp = r.post(f"https://annotate.nilay.page/api/predictions?fileid=$FILEID&savename=ai-test-preds-chunk-{n}", json={'annotations': chunk_data.to_dict()})

EOF
python /dev/fd/3
