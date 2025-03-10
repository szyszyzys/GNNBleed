#!/bin/bash

# ==========================================
# Usage Examples:
# ==========================================
# bash eval.sh inf_3 path/to/result.pt

# Ensure at least dataset name, model, and attack name are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <attack_name> <result_path>"
    exit 1
fi

# Assign input arguments with default values for optional parameters
attack_name="$1"
result_path="${2:-results}"  # Default: "results" if not provided


# Log the execution details
echo "▶ Attack: $attack_name"
# Execute evaluation
python main.py --mode eval\
    --result_path "$result_path" \
    --attack_type "$attack_name"

