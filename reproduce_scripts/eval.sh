#!/bin/bash

# ==========================================
# Usage Examples:
# ==========================================
# bash eval.sh inf_3 path/to/result.pt
# bash eval.sh inf_3             # will use default "results" for result_path

# Ensure at least the attack name is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <attack_name> [result_path]"
    exit 1
fi

# Assign input arguments with a default value for the optional result_path
attack_name="$1"
result_path="${2:-results}"  # Default: "results" if not provided

# Log the execution details
echo "▶ Attack: $attack_name"

# Execute evaluation
python main.py --mode eval \
    --result_path "$result_path" \
    --attack_type "$attack_name"
