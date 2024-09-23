#!/bin/zsh
SAVES=$(http GET https://annotate.nilay.page/api/save/all\?final\=true)
URLS=$(echo $SAVES | jq '.saves[] | select(.fileid | contains("(mmd)")) | @uri "https://annotate.nilay.page/api/annotations/export?userid=\(.userid)&timestamp=\(.timestamp)&savename=\(.savename)&fileid=\(.fileid)&tokenizer=FacebookAI%2Froberta-base&ignore_annotation_endpoints=True"' | sed 's/"//g')

# Absolutely cursed way to download all the annotations and format them properly for the autolinker.
echo $URLS \
  | tr '\n' '\0' \
  | (xargs -P4 -0 -L1 http --download GET )
