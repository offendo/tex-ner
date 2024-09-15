#!/bin/zsh
SAVES=$(http GET https://annotate.nilay.page/api/save/all\?final\=true)
URLS=$(echo $SAVES | jq '.saves[] | @uri "https://annotate.nilay.page/api/annotations?userid=\(.userid)&timestamp=\(.timestamp)&fileid=\(.fileid)"' | sed 's/"//g')

# Absolutely cursed way to download all the annotations and format them properly for the autolinker.
echo $URLS \
  | tr '\n' '\0' \
  | (xargs -0 -L1 http GET | jq) \
  | jq '.annotations[].timestamp = .timestamp' \
  | jq '.annotations[].userid = .userid' \
  | jq '.annotations' \
  | jq -s 'add' \
  | jq '{annotations: .}' > annotations.json
