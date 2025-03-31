# AdaCGP: Online Graph Learning via Time-Vertex Adaptive Filters

This repository contains the code for AdaCGP (Adaptive Causal Graph Process), an online algorithm for adaptive estimation of the Graph Shift Operator (GSO) from multivariate time series.

[![arXiv](https://img.shields.io/badge/arXiv-2411.01567-b31b1b.svg)](https://arxiv.org/abs/2411.01567)

## Overview

Graph Signal Processing (GSP) provides a powerful framework for analysing complex, interconnected systems by modelling data as signals on graphs. Recent advances in GSP have enabled the learning of graph structures from observed signals, but these methods often struggle with time-varying systems and real-time applications. Adaptive filtering techniques, while effective for online learning, have seen limited application in graph topology estimation from a GSP perspective.

AdaCGP is an online algorithm for adaptive estimation of the Graph Shift Operator (GSO) from multivariate time series. The GSO is estimated from an adaptive time-vertex autoregressive model through recursive update formulae designed to address sparsity, shift-invariance and bias. Through simulations, we show that AdaCGP performs consistently well across various graph topologies, and achieves improvements in excess of 82% for GSO estimation compared to baseline adaptive vector autoregressive models.

Our online variable splitting approach for enforcing sparsity enables near-perfect precision in identifying causal connections while maintaining low false positive rates upon optimisation of the forecast error. Finally, AdaCGP's ability to track changes in graph structure is demonstrated on recordings of ventricular fibrillation dynamics in response to an anti-arrhythmic drug.

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@misc{jenkins2024onlinegraphlearningtimevertex,
      title={Online Graph Learning via Time-Vertex Adaptive Filters: From Theory to Cardiac Fibrillation}, 
      author={Alexander Jenkins and Thiernithi Variddhisai and Ahmed El-Medany and Fu Siong Ng and Danilo Mandic},
      year={2024},
      eprint={2411.01567},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2411.01567}, 
}
```

## Experiments

### 1. Simulation Study

#### Hyperparameter searches
To recreate our results, first run hyperparameter searches to find optimal configurations for different models and graph topologies:

**AdaCGP**
```bash
python -m experiments.sweep --config-path ../config --config-name config_sweep
```

**TISO**
```bash
python -m experiments.sweep --config-path ../config --config-name config_sweep_tiso
```

**TIRSO**
```bash
python -m experiments.sweep --config-path ../config --config-name config_sweep_tirso
```

**SD-SEM**
```bash
python -m experiments.sweep --config-path ../config --config-name config_sweep_sdsem
```

**GLasso**
```bash
python -m experiments.sweep --config-path ../config --config-name config_sweep_glasso
```

**GL-SigRep**
```bash
python -m experiments.sweep --config-path ../config --config-name config_sweep_glsigrep
```

#### Best configurations

To run the models with the best configuration and different random seeds:

**AdaCGP**
```bash
python -m experiments.best_sweep_mc_adacgp --config-path ../config --config-name config_best_sweep_mc
```

**TISO**
```bash
python -m experiments.best_sweep_mc_tiso --config-path ../config --config-name config_best_sweep_mc_tiso
```

**TIRSO**
```bash
python -m experiments.best_sweep_mc_tirso --config-path ../config --config-name config_best_sweep_mc_tirso
```

**SD-SEM**
```bash
python -m experiments.best_sweep_mc_sdsem --config-path ../config --config-name config_best_sweep_mc_sdsem
```

**GLasso**
```bash
python -m experiments.best_sweep_mc_glasso --config-path ../config --config-name config_best_sweep_mc_glasso
```

**GLSigRep**
```bash
python -m experiments.best_sweep_mc_glsigrep --config-path ../config --config-name config_best_sweep_mc_glsigrep
```

**VAR**
```bash
python -m experiments.sweep --config-path ../config --config-name config_sweep_mc_var
```

**VAR + Granger causality**
```bash
python -m experiments.sweep --config-path ../config --config-name config_sweep_mc_granger
```

After running these, our results table and figures can be recreating by running:
```bash
python generate_simulation_results_table.py
```
and
```bash
python generate_simulation_results_figures.py
```

### 2. Computational complexity

Evaluate the computational complexity over N and T for all models:
```bash
python -m experiments.sweep --config-path ../config --config-name config_complexity_sweep
```

To recreate our computation complexity figures:
```bash
python generate_complexity_results_figures.py
```

### 3. AdaCGP sparsity experiments

To recreate our sparsity experiments:

```bash
python -m experiments.sweep --config-path ../config --config-name config_sweep_sparsity
```

To recreate our sparsity figures:
```bash
python generate_sparsity_results_figures.py
```

### 4. Ventricular Fibrillation Data Analysis

Analyze real-world ventricular fibrillation (VF) data with AdaCGP:

```bash
python experiments/run_vf_analysis.py --data_path data/vf_recordings/ --drug flecainide
```

This experiment demonstrates AdaCGP's ability to track changes in graph structure in response to an anti-arrhythmic drug, identifying the stability of critical conduction patterns that may be maintaining cardiac arrhythmia.

## Installation

```bash
# Clone the repository
git clone https://github.com/username/adacgp.git
cd adacgp

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.