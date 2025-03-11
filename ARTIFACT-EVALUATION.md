# Artifact Appendix

Paper title: **GNNBleed: Inference Attacks to Unveil Private Edges in Graphs with Realistic Access to GNN Models**

Artifacts HotCRP Id: #7

Requested Badge: **Reproduced**

## Description

This repository contains the code for our paper. In short, we propose a new threat model and attacks to infer graph
edges during the inference phase of GNN models.

### Security/Privacy Issues and Ethical Concerns (All badges)

None.

## Basic Requirements (Only for Functional and Reproduced badges)

### Hardware Requirements

GPU.

### Software Requirements

No proprietary software.

### Estimated Time and Storage Consumption

The time and storage requirements may vary across devices but should not be significantly long. We provide script arguments to adjust the size of the test set, which linearly increases the execution time.

## Environment
- Required dependencies (install via `bash reproduce_scripts/setup_env.sh`)
- CUDA-enabled GPU

### Accessibility (All badges)

Git Repositories (https://github.com/szyszyzys/GNNBleed.git), Revision Number: 36ebafd5a9a2bdc1a527457c406f577bd2945f26.

### Set up the environment (Only for Functional and Reproduced badges)

```bash
git clone https://github.com/szyszyzys/GNNBleed.git
cd GNNBleed
bash reproduce_scripts/setup_env.sh
source ~/.bashrc  # Load Conda into the current session
conda activate GNNBLEED
```

### Testing the Environment (Only for Functional and Reproduced badges)

```bash
bash reproduce_scripts/train.sh "lastfm" 0
```

## Artifact Evaluation (Only for Functional and Reproduced badges)

### Main Results and Claims

#### Main Result 1: Illustration of why our approach is effective and why existing methods are flawed.

Illustration of why our approach is effective and why existing methods are flawed.

#### Main Result 2: Our work outperform baselines in static graph scenario.

Our work outperform baselines in static graph scenario.

#### Main Result 3: Our work outperform baselines in dynamic graph scenario.

Our work outperform baselines in dynamic graph scenario.

### Experiments

We provide detailed instruction in README. Please follow it step by step.

#### Experiment 1: Illustration of why our work works and why exiting works are flawed.

Step 2 in README.
We provide a script to visualize the intuition behind this work in **`visualization.ipynb`**.

**Results**: Our analysis shows that influence-based methods can infer the existence of edges. However, existing
approaches use a flawed design in their attack strategies.

#### Experiment 2: Our work outperform baselines in static graph scenario.

Step 4 in README.
**Results**: Our attack (INF-MAG, INF-DIR) outperform baselines (IIA, LTA).

#### Experiment 3: Our work outperform baselines in dynamic graph scenario.

Step 5 in README.
**Results**: Direction-based attacks (INF-DIR*) outperform Magnitude-based ones (INF-MAG, IIA, LTA).

## Limitations (Only for Functional and Reproduced badges)

The exact numbers may vary, but the overall pattern will remain consistent.

## Notes on Reusability (Only for Functional and Reproduced badges)

Varying the arguments of scripts for different experiment setup. Detailed are provided in README.
New dataset can be added to GraphDataModule in data_loader.py.
New GNN models can be added to model.py.