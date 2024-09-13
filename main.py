# main.py
import torch
import hydra
import pickle
import os
import numpy as np

from omegaconf import OmegaConf, DictConfig
from src.data_generation import generate_data
from src.utils import set_seed
from src.models.adaptive.AdaCGP import AdaCGP
from src.plotting import save_figures
from src.eval_metrics import save_results

def get_model(name):
    models = {
        'AdaCGP': AdaCGP
    }
    if name not in models:
        raise ValueError(f"Model {name} not implemented")
    return models[name]

def get_save_path():
    return hydra.core.hydra_config.HydraConfig.get().run.dir

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # set params
    set_seed(cfg.seed)
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    
    # generate data and move to device
    X, y, graph_filters_flat, weight_matrix, filter_coefficients = [d.to(device) for d in generate_data(cfg)]

    # Initialise model
    model = get_model(cfg.model.name)(cfg.data.N, cfg.model.hyperparams, device)
    
    # Run optimization
    results = model.run(X, y, weight_matrix, filter_coefficients, graph_filters_flat)

    # Save results
    save_path = get_save_path()
    save_results(cfg.model.hyperparams.patience, results, save_path)
    save_figures(results, weight_matrix, save_path)

if __name__ == "__main__":
    main()