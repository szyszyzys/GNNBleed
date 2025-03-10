#!/bin/bash

# Default hyperparameters
default_models=("gin" "gat" "gcn" "sage")
default_layers=(3 4)
default_h_dims=(64 128 256)

# Ensure dataset and GPU ID are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <dataset> <gpu_id> [model] [num_layers] [hidden_dim]"
    exit 1
fi

dataset=$1
gpu_id=$2

# Check if optional parameters are provided; otherwise, use defaults
models=("${default_models[@]}")
n_layers=("${default_layers[@]}")
h_dims=("${default_h_dims[@]}")

if [ -n "$3" ]; then models=("$3"); fi  # If a model is provided, override defaults
if [ -n "$4" ]; then n_layers=("$4"); fi  # If layers are provided, override defaults
if [ -n "$5" ]; then h_dims=("$5"); fi  # If hidden dims are provided, override defaults

# Function to run training
Run_train() {
    local model=$1
    local num_layers=$2
    local hidden_dim=$3
    local dataset=$4
    local gpu_id=$5

    echo "Training model: $model | Layers: $num_layers | Hidden Dim: $hidden_dim | Dataset: $dataset | GPU: $gpu_id"

    python main.py --mode train \
        --model "$model" \
        --num_of_worker 4 \
        --dataset "$dataset" \
        --n_epoch 300 \
        --dropout 0.5 \
        --num_layers "$num_layers" \
        --h_dim "$hidden_dim" \
        --lr 0.001 \
        --gpuid "$gpu_id" \
        --root_path revision
}

# Loop through all combinations
for model in "${models[@]}"; do
    for n_layer in "${n_layers[@]}"; do
        for h_dim in "${h_dims[@]}"; do
            Run_train "$model" "$n_layer" "$h_dim" "$dataset" "$gpu_id"
        done
    done
done
