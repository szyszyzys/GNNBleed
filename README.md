# GNNBLEED

## Prerequisites

Ensure you have the following installed:

- Python (>=3.7)
- Required dependencies (install via `conda env create -f environment_repo.yml`)
- CUDA-enabled GPU (if running with GPU support)

## Visualize attack intuitions

## Run experiment

Run the script as follows to reproduce the full attack pipeline.
There are additional options in the scripts to explore variations of each step. Please check the individual scripts for
more details.
Ensure that the arguments across different steps remain consistent, such as model structures. From our observations,
varying parameters within reasonable limits leads to similar findings as discussed in the main paper.

## 1. train model

Script to train the target model.

```bash
bash reproduce_scripts/train.sh <dataset_name> <gpu_id> [model] [num_layers] [hidden_dim]
```

### Arguments:

- **`<dataset_name>`**: Name of the dataset (e.g., `lastfm`, `flickr`, `twitch/en`).
- **`<gpu_id>`**: ID of the GPU to use for training.
- **`[model]`** *(optional)*: Specify a GNN model (default: all models).
- **`[num_layers]`** *(optional)*: Specify the number of layers (default: all layers).
- **`[hidden_dim]`** *(optional)*: Specify the hidden dimension size (default: all sizes).

If no optional arguments are provided, the script trains **all models** with **all default configurations**.

### Default Settings:

- **Models**: `"gin"`, `"gat"`, `"gcn"`, `"sage"`
- **Number of Layers**: `3`, `4`
- **Hidden Dimensions**: `64`, `128`, `256`

You can experiment with any combination of these parameters and observe similar patterns in the following experiments as
long as the trained model is usable (i.e. converge or reach some level of utility).

## 3. visualization of intuitions

We provide the script to visualize the intuition of this work in **`visualization.ipynb`**.

## 4. Sample Node Pairs for Attack (Attack Test Set Construction).

This script samples node pairs from various datasets based on their degree. It first selects **target nodes** that meet
the specified degree constraints. Then, for each target node, it collects **other nodes** that could potentially be
connected to it (the candidate set for the target node).

In short, this script **generates node pairs for inference**.

```bash
bash reproduce_scripts/sample_nodes.sh [target_node_degree_lower_bound] [target_node_degree_upper_bound] [num_target_nodes]  [sub_path]
```

- **`[target_node_degree_lower_bound]`** (Optional, default 0): Lower bound of the target node's degree.
- **`[target_node_degree_upper_bound]`** (Optional, default 1000): Upper bound of the target node's degree.
- **`[num_target_nodes]`** (Optional, default 1000): Number of target nodes to sample.
- **`[sub_path]`** (Optional, default: uncons): Sub path used to save the current sampling, i.e. give current sampling a
  name.

## 3. Static Scenario Attack

Attack in static scenario on all the models:

```bash
bash reproduce_scripts/attack_static.sh <dataset> [gpu_id] [attack_types] [target_layer] [h_dims] [insert_node_strategy] [root_path] 
```

- **`<dataset>`**: Name of the dataset (e.g., `lastfm`, `flickr`, `twitch/en`).
- **`[gpu_id]`** *(optional, default: `0`)*: GPU ID for execution.
- **`[attack_types]`** *(optional, default: `"inf_3"`)*: Specifies the attack type to perform.
  - **Available options:**
    - `inf_4` → **INF-DIR** (as referred to in the paper)
    - `inf_3` → **INF-MAG** (as referred to in the paper)
    - `lta` → **LTA** (as referred to in the paper)
    - `infiltration` → **IIA** (as referred to in the paper)
  - Additional attack methods are included in the provided code, some of which **may outperform** the ones reported in
    the main paper.
- **`[target_layer]`** *(optional, default: `4`)*: Number of layers in the target model.
- **`[h_dims]`** *(optional, default: `64 128 256`)*: Hidden dimension size of the model.
- **`[insert_node_strategy]`** *(optional, default: `"same"`)*: Strategy for inserting nodes (`same`, `randsame`, `diff1`, `diff2`, etc.).
- **`[root_path]`** *(optional, default: `"result"`)*: Path where outputs will be stored.

## 4. Dynamic Scenario Attack

### Varying evolving rate
```bash
bash reproduce_scripts/attack_dynamic_varying_ER.sh <dataset> [gpu_id] [target_layer] [insert_node_strategy] [evolve_mode] [attack_type] [n_hidden] [new_node] [root_path]
```

- **`<dataset>`**: Name of the dataset (e.g., `lastfm`, `twitch/en`, `twitch/fr`).
- **`[gpu_id]`** *(optional, default: `7`)*: GPU ID for execution.
- **`[target_layer]`** *(optional, default: `4`)*: Number of layers in the target model.
- **`[insert_node_strategy]`** *(optional, default: `"same"`)*: Strategy for inserting nodes (e.g., `same`, `random`).
- **`[evolve_mode]`** *(optional, default: `"all"`)*: Evolution strategy to modify the graph dynamically (
  e.g., `all`, `subgraph`, `local_structure`).
- **`[attack_types]`** *(optional, default: `"inf_3"`)*: Specifies the attack type to perform.
  - **Available options:**
    - `dp2` → **INF-DIR*** (as referred to in the paper)
    - `inf_4` → **INF-DIR** (as referred to in the paper)
    - `inf_3` → **INF-MAG** (as referred to in the paper)
    - `lta` → **LTA** (as referred to in the paper)
    - `infiltration` → **IIA** (as referred to in the paper)
  - Additional attack methods are included in the provided code, some of which **may outperform** the ones reported in
    the main paper.
- **`[n_hidden]`** *(optional, default: `64`)*: Number of hidden dimensions in the model.
- **`[new_node_rate]`** *(optional, default: `0.2`)*: Rate of new nodes added.
- **`[root_path]`** *(optional, default: `"result"`)*: Path where outputs and model results will be stored.

### Varying perturbation rate

This script performs attacks by varying the perturbation rate.

```bash
bash reproduce_scripts/attack_dynamic_PR.sh <dataset> [gpu_id] [target_layer] [insert_node_strategy] [dynamic_rate] [evolve_mode] [n_hidden] [new_node_rate] [attack_type]
```

- **`<dataset>`**: Name of the dataset (e.g., `lastfm`, `twitch/en`, `twitch/fr`).
- **`[gpu_id]`** *(optional, default: `7`)*: GPU ID for execution.
- **`[target_layer]`** *(optional, default: `4`)*: Number of layers in the target model.
- **`[insert_node_strategy]`** *(optional, default: `"same"`)*: Strategy for inserting nodes (e.g., `same`, `randsame`).
- **`[dynamic_rate]`** *(optional, default: `0.01`)*: Rate of dynamic changes applied to the graph.
- **`[evolve_mode]`** *(optional, default: `"all"`)*: Evolution strategy for modifying the graph.
  - **Available options:**
    - `all`
    - `feature`,
    - `structure`
    - `local_structure`
- **`[n_hidden]`** *(optional, default: `"64"`)*: Number of hidden dimensions in the model.
- **`[new_node_rate]`** *(optional, default: `"0.2"`)*: Rate of new nodes added.
- **`[root_path]`** *(optional, default: `"result"`)*: Path where outputs and model results will be stored.
- **`[attack_type]`** *(optional, default: `"dp2"`)*: Specifies the attack type to perform.
  - **Available options:**
    - `dp2` → **INF-DIR*** (as referred to in the paper)
    - `inf_4` → **INF-DIR** (as referred to in the paper)
    - `inf_3` → **INF-MAG** (as referred to in the paper)
    - `lta` → **LTA** (as referred to in the paper)
    - `infiltration` → **IIA** (as referred to in the paper)
  - Additional attack methods are included in the provided code, some of which **may outperform** the ones reported in
    the main paper.

## 5. Analyzing statistics
This step evaluates the **performance** of those attacks based on the collected results.

### **Usage**

```bash
bash reproduce_scripts/eval.sh <dataset_name> <model> <attack_name> [n_layer] [evo_rate] [rate_of_new_nodes_added]
```

- **`<dataset_name>`**: Name of the dataset (e.g., `lastfm`, `twitch/pt`, `twitch/fr`)
- **`<model>`**: GNN model to evaluate (`gcn`, `gat`, `gin`, `sage`)
- **`<attack_names>`**: Name of the attack (must match the attack used previously, e.g. "inf_3 lta")
- **`[n_layer]`** *(optional , default: `3`)*: Number of layers in the model
- **`[evo_rate]`** *(optional, only for dynamic scenario, default: `0.02`)*: Evolution rate for dynamic changes
- **`[rate_of_new_nodes_added]`** *(optional, only for dynamic scenario, default: `0.2`)*: Rate at which new nodes were
  added in the attack