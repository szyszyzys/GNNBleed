#!/bin/bash

# Usage examples:
# bash eval.sh lastfm gcn attack1
# bash eval.sh twitch/pt sage attack2 4 0.01 0.3
# bash eval.sh twitch/fr gin attack3 3 0.02 0.2 uncons

# Ensure at least dataset name, model, and attack name are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <dataset_name> <model> <attack_name> [n_layer] [evo_rate] [rate_of_new_nodes_added] [degrees]"
    exit 1
fi

# Assign input arguments
dataset_name=$1
model=$2
attack_names=$3
n_layer="${4:-3}"  # Default: 3 layers
evo_rate="${5:-0.02}"  # Default: Evolution rate 0.02
rate_of_new_nodes_added="${6:-0.2}"  # Default: 0.2
degrees="${7:-uncons}"  # Default: "uncons"

# Run evaluation
for attack_name in "${attack_names[@]}"; do
  for deg in $degrees; do
      echo "Evaluating model: $model | Dataset: $dataset_name | Attack: $attack_name | Layers: $n_layer | Evolution Rate: $evo_rate | New Node Rate: $rate_of_new_nodes_added | Degree: $deg"

      python main.py --mode eval \
          --result_path outputs/model_outputs/"$dataset_name"/"$model"/"$n_layer"/"$deg"/"${attack_name}_random_dynamic_${evo_rate}_insertNode_same_neighborinsert_True_${rate_of_new_nodes_added}.pt" ----attack_type attack_name
  done
done