# main.py
import torch
import hydra
import pickle
import os

from omegaconf import OmegaConf, DictConfig
from src.data_generation import generate_data
from src.utils import set_seed
from src.models.adaptive.AdaCGP import AdaCGP

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
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    
    # generate data and move to device
    X, y, graph_filters_flat, weight_matrix, filter_coefficients = [d.to(device) for d in generate_data(cfg)]

    # Initialise model
    model = get_model(cfg.model.name)(cfg.data.N, cfg.model.hyperparams, device)
    
    # Run optimization
    results = model.run(X, y, weight_matrix, filter_coefficients, graph_filters_flat)

    # Save results
    fpath = get_save_path()
    fpath = os.path.join(fpath, 'results.pkl')
    with open(fpath, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {fpath}")

if __name__ == "__main__":
    main()