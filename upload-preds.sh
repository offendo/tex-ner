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

data = json.loads('''$DATA''')
resp = r.post("https://annotate.nilay.page/api/predictions?fileid=$FILEID&savename=ai-test-preds", json={'annotations': data})

EOF
python /dev/fd/3
