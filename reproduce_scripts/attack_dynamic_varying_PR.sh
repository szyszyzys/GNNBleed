#!/bin/bash

# ========================
# Usage Examples:
# ========================
# bash attack_dynamic_varying_PR.sh lastfm 6 4 randsame 0.01 all
# bash attack_dynamic_varying_PR.sh twitch/en 6 4 same 0.01
# bash attack_dynamic_varying_PR.sh lastfm 7 4 same 0.01 all dp2 inf_3

# ========================
# Validate Input Arguments
# ========================
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <dataset> [gpu_id] [target_layer] [insert_node_strategy] [dynamic_rate] [evolve_mode] [attack_type...]"
    exit 1
fi

# ========================
# Assign Input Arguments with Defaults
# ========================
dataset="$1"
gpu_id="${2:-7}"
target_layer="${3:-3}"
insert_node_strategy="${4:-same}"
dynamic_rate="${5:-0.01}"
evolve_mode="${6:-all}"
n_hidden="${7:-64}"
new_node_rate="${8:-0.2}"
root_path="${9:-revision}"

# Assign attack types (supports multiple attacks)
shift 9  # Shift past the first 9 arguments
attack_types=("${@:-dp2}")  # Use default "dp2" if no attack type is provided

# ========================
# Static Settings
# ========================
target_node_degree="/"
models=("gin" "gcn" "gat" "sage")
prs=("0.1" "0.3" "0.5" "0.7" "0.9" "1.1")

# ========================
# Function to Execute Attack
# ========================
Run_attack() {
    local model="$1"
    local num_layers="$2"
    local dataset="$3"
    local attack_type="$4"
    local gpu_id="$5"
    local target_node_degree="$6"
    local perturb_rate="$7"

    echo "▶ Running attack on model: $model | Dataset: $dataset | Attack: $attack_type | Layers: $num_layers | GPU: $gpu_id"

    python main.py \
        --mode attack \
        --attack_type "$attack_type" \
        --dataset "$dataset" \
        --gpuid "$gpu_id" \
        --model "$model" \
        --model_path "./outputs/trained_model/${model}/${dataset}/nlayer_${num_layers}_hdim_${n_hidden}_lr_0.001_epoch_300_None.pt" \
        --num_layers "$num_layers" \
        --attack_node_num 1000 \
        --attack_node_degree "$target_node_degree" \
        --dynamic \
        --dynamic_rate "$dynamic_rate" \
        --dynamic_insert_neighbor \
        --dp2_insert_node random \
        --n_neighborhood_new_node "$new_node_rate" \
        --insert_node_strategy "$insert_node_strategy" \
        --evolving_mode "$evolve_mode" \
        --perturb_rate "$perturb_rate" \
        --root_path "$root_path" \
        --twohop
}

# ========================
# Execute Attacks for Each Model and Attack Type
# ========================
for model in "${models[@]}"; do
    for attack_t in "${attack_types[@]}"; do
        for pr in "${prs[@]}"; do
            Run_attack "$model" "$target_layer" "$dataset" "$attack_t" "$gpu_id" "$target_node_degree" "$pr"
        done
    done
done
