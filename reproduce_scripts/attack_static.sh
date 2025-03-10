#!/bin/bash

# Ensure the script is executed with at least a dataset argument
if [ -z "$1" ]; then
    echo "Usage: $0 <dataset> [gpu_id] [attack_types] [target_layer] [insert_node_strategy] [root_path] [twohop_enabled] [h_dims]"
    exit 1
fi

# Assign input arguments with default values
dataset="$1"
gpu_id="${2:-0}"
attack_types="${3:-inf_3}"
target_layer="${4:-4}"
h_dims=("${5:-"64 128 256"}")
insert_node_strategy="${6:-same}"
root_path="${7:-result}"
twohop_enabled="${8:-yes}"
models=("gcn" "gin" "sage" "gat")
p_rates=("1")
lrs=("0.001")

# Function to execute attack
run_attack() {
    local model="$1"
    local num_layers="$2"
    local dataset="$3"
    local attack_type="$4"
    local gpu_id="$5"
    local insert_node_strategy="$6"
    local perturb_rate="$7"
    local h_dim="$8"
    local lr="$9"
    local root_path="${10}"
    local twohop_enabled="${11}"

    echo "Running attack: Model=$model, Dataset=$dataset, Attack=$attack_type, Layers=$num_layers, GPU=$gpu_id"

    # Construct the command dynamically
    cmd="python main.py \
        --mode attack \
        --attack_type \"$attack_type\" \
        --dataset \"$dataset\" \
        --gpuid \"$gpu_id\" \
        --model \"$model\" \
        --model_path \"./${root_path}/outputs/trained_model/${model}/${dataset}/nlayer_${num_layers}_hdim_${h_dim}_lr_${lr}_epoch_300_None.pt\" \
        --num_layers \"$num_layers\" \
        --h_dim \"$h_dim\" \
        --attack_node_num 500 \
        --insert_node_strategy \"$insert_node_strategy\" \
        --insert_node_feature random \
        --remove_self_loop \
        --perturb_rate \"$perturb_rate\" \
        --lr \"$lr\" \
        --root_path \"$root_path\""

    # Add --twohop if enabled
    if [[ "$twohop_enabled" == "yes" ]]; then
        cmd+=" --twohop"
    fi

    # Execute the command
    eval "$cmd"
}

# Iterate over models, attack types, and hyperparameters
for model in "${models[@]}"; do
    for attack_type in "${$attack_types[@]}"; do
        for p_rate in "${p_rates[@]}"; do
            for h_dim in "${h_dims[@]}"; do
                for lr in "${lrs[@]}"; do
                    run_attack "$model" "$target_layer" "$dataset" "$attack_type" "$gpu_id" "$insert_node_strategy" "$p_rate" "$h_dim" "$lr" "$root_path" "$twohop_enabled"
                done
            done
        done
    done
done
