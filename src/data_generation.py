# src/data_generation.py
import torch
import numpy as np
from src.graph_filters import generate_graph_filters

def generate_data(cfg):
    """Generate data from the adaptive graph AR model

    Args:
        cfg (OmegaConf): configuration object

    Returns:
        list(torch.tensor): data, target, graph_filters_flat, weight_matrix, filter_coefficients
    """

    N, P = cfg.data.N, cfg.data.P
    process_length = cfg.data.process_length
    burn_in = cfg.data.burn_in
    noise_factor = cfg.data.noise_factor
    graph_cfg = cfg.graph

    graph_filters_flat, weight_matrix, filter_coefficients = generate_graph_filters(N, P, **graph_cfg)

    # generate data via AR model, initialise with white noise
    x = np.random.randn(P*N, 1).reshape(P, N)
    data = [x]
    target = []

    for t in range(process_length + burn_in):
        y = graph_filters_flat @ x.reshape(N*P, 1)
        next_obs = y + noise_factor * np.random.randn(N, 1)
        target.append(next_obs)

        if t == process_length + burn_in - 1:
            break

        x = np.roll(x, shift=1, axis=0)
        x[0] = next_obs.flatten()
        data.append(x)

    # remove burn-in period where dynamics have not yet stabilised
    data = np.array(data)[burn_in:]
    target = np.array(target)[burn_in:]

    return torch.tensor(data, dtype=torch.float32), torch.tensor(target, dtype=torch.float32), \
        torch.tensor(graph_filters_flat, dtype=torch.float32), torch.tensor(weight_matrix, dtype=torch.float32), \
            torch.tensor(filter_coefficients, dtype=torch.float32)