#!/bin/bash
# bash eval.sh lastfm/gcn/ 1 False
# bash eval.sh twitch/pt/sage/ ; bash eval.sh twitch/pt/sage/ ; bash eval.sh twitch/pt/gat/ ; bash eval.sh twitch/pt/gcn/ ; bash eval.sh twitch/fr/sage/ ; bash eval.sh twitch/fr/gat/
result_path=$1
threshold_ratio="0.8 1.0 1.2"
degree="low uncons high"
echo $result_path

for t in $threshold_ratio; do
  for d in $degree; do
    for file in "$result_path"/"$d"/*; do
      echo degree $d threshold $t
      file_name=${file##*/}
      if [[ $file_name == inf_3_result.pt || $file_name == inf_3_random_result.pt ]]; then
        python main.py --mode eval --result_path "$file" --attack_type inf --threshold_ratio $t | tee -a "$result_path"/eval
        break
      fi
      echo "============================"
    done
  done
done
echo result: "$result_path"/eval
