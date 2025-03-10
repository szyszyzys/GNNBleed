#!/bin/bash

# Usage Example:
# bash attack_dynamic_varying_ER.sh lastfm 4 3 same all inf_3 64 0.2
# bash attack_dynamic_varying_ER.sh lastfm 4 3 same all dp2 64 0.2
# bash attack_dynamic_varying_ER.sh lastfm 4 3 same all lta 64 0.2
# bash attack_dynamic_varying_ER.sh lastfm 4 3 same all infiltration 64 0.2

# Ensure at least required arguments are provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <dataset> [gpu_id] [target_layer] [insert_node_strategy] [evolve_mode] [attack_type] [n_hidden] [new_node] [root_path]"
    exit 1
fi

# Arguments with default values
dataset="${1:-lastfm}"
gpu_id="${2:-7}"
target_layer="${3:-4}"
insert_node_strategy="${4:-same}"
evolve_mode="${5:-all}"
attack_type="${6:-dp2}"
n_hidden="${7:-64}"
new_node="${8:-0.2}"
root_path="${9:-result}}"

# Static settings
target_node_degree=""
models=("gcn" "gat" "sage" "gin")
prs=("1")
dynamic_rates=("0.000001" "0.00001" "0.0001" "0.0005" "0.001" "0.005" "0.01" "0.02" "0.05")

# Function to run the attack
Run_attack() {
    local model="$1"
    local num_layers="$2"
    local dataset="$3"
    local attack_type="$4"
    local gpu_id="$5"
    local target_node_degree="$6"
    local perturb_rate="$7"
    local dynamic_rate="$8"
    local new_node="$9"
    local root_path="${10}"
    local n_hidden="${11}"

    echo "Running attack on model: $model | Dataset: $dataset | Attack Type: $attack_type | GPU: $gpu_id"

    python main.py \
        --mode attack \
        --attack_type "$attack_type" \
        --dataset "$dataset" \
        --gpuid "$gpu_id" \
        --model "$model" \
        --model_path "./${root_path}/outputs/trained_model/${model}/${dataset}/nlayer_${num_layers}_hdim_${n_hidden}_lr_0.001_epoch_300_None.pt" \
        --num_layers "$num_layers" \
        --attack_node_num 1000 \
        --attack_node_degree "$target_node_degree" \
        --dynamic \
        --dynamic_rate "$dynamic_rate" \
        --dynamic_insert_neighbor \
        --dp2_insert_node random \
        --n_neighborhood_new_node "$new_node" \
        --insert_node_strategy "$insert_node_strategy" \
        --evolving_mode "$evolve_mode" \
        --perturb_rate "$perturb_rate" \
        --h_dim "$n_hidden" \
        --root_path "$root_path"\
        --twohop
}

# Loop through models, attack types, and dynamic rates
for model in "${models[@]}"; do
    for dynamic_rate in "${dynamic_rates[@]}"; do
        for pr in "${prs[@]}"; do
            Run_attack "$model" "$target_layer" "$dataset" "$attack_type" "$gpu_id" "$target_node_degree" "$pr" "$dynamic_rate" "$new_node" "$root_path" "$n_hidden"
        done
    done
done
