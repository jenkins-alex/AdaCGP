# AdaCGP: Online Graph Learning via Time-Vertex Adaptive Filters

This repository contains the official implementation of AdaCGP (Adaptive Constrained Graph Process), an online algorithm for adaptive estimation of the Graph Shift Operator (GSO) from multivariate time series.

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

Run hyperparameter searches to find optimal configurations for different graph topologies:

```bash
python experiments/run_hyperparameter_search.py
```

Best configurations for different graph types:

- **Erdős-Rényi graphs**: `alpha=0.01, beta=0.05, mu=0.1`
- **Barabási-Albert graphs**: `alpha=0.005, beta=0.01, mu=0.2`
- **Small-world graphs**: `alpha=0.01, beta=0.02, mu=0.15`

To run a simulation with the best configuration:

```bash
python experiments/run_adacgp.py --graph_type erdos_renyi --alpha 0.01 --beta 0.05 --mu 0.1
```

### 2. Benchmarking

Evaluate the computational complexity of AdaCGP compared to baselines:

```bash
python experiments/run_benchmarks.py --nodes 10 20 50 100 --time_steps 1000
```

This will generate plots comparing:
- Execution time vs. number of nodes
- Memory usage vs. number of nodes
- Convergence rate analysis

### 3. Ventricular Fibrillation Data Analysis

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