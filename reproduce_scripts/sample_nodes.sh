#!/bin/bash

# bash sample_nodes.sh uncons 3 1000 500
dataset='twitch/pt twitch/ru twitch/de twitch/fr twitch/en lastfm flickr'
degree_low=${1:-30}
degree_high=${2:-1000}
num_pairs=${3:-1000}
sub_path=${4:-uncons}

Run_sample() {
  python main.py --mode sample --num_node_pairs $5 --dataset $1 --sample_node_subpath $2 --degree_low $3 --degree_high $4 --twohop
}

for data in $dataset; do
  Run_sample "$data" "$sub_path" "$degree_low" "$degree_high" "$num_pairs"
done
