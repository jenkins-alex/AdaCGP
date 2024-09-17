# main.py
import torch
import hydra
import pickle
import os

from omegaconf import OmegaConf, DictConfig
from src.data_generation import generate_data
from src.utils import set_seed
from src.models.adaptive.AdaCGP import AdaCGP
from src.models.adaptive.TISO import TISO
from src.models.adaptive.TIRSO import TIRSO
from src.eval_metrics import save_results

def get_model(name):
    models = {
        'AdaCGP': AdaCGP,
        'TISO': TISO,
        'TIRSO': TIRSO
    }
    if name not in models:
        raise ValueError(f"Model {name} not implemented")
    return models[name]

def get_save_path():
    sweep = hydra.core.hydra_config.HydraConfig.get().sweep
    return sweep.dir, sweep.subdir

@hydra.main(version_base=None, config_path="config", config_name="config_sweep")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # set params
    set_seed(cfg.seed)
    if cfg.get('device', None) is not None:
        device = torch.device(cfg.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    try:
        # generate data and move to device
        X, y, graph_filters_flat, weight_matrix, filter_coefficients = [d.to(device) for d in generate_data(cfg)]

        # Initialise model
        model = get_model(cfg.model.name)(cfg.data.N, cfg.model.hyperparams, device)
        
        # Run optimization
        model_inputs = {
            'X': X,
            'y': y,
            'weight_matrix': weight_matrix,
            'filter_coefficients': filter_coefficients,
            'graph_filters_flat': graph_filters_flat
        }
        results = model.run(**model_inputs)

        # Save results
        dir, subdir = get_save_path()
        save_path = os.path.join(dir, subdir)
        error_metric = save_results(cfg.model.name, cfg.model.hyperparams.patience, results, save_path, cfg.get('dump_results', False))

        if results[error_metric][-1] != results[error_metric][-1]:
            return 1e6
        return results[error_metric][-1]
    except Exception as e:
        print(e)
        return results[error_metric][-1]

if __name__ == "__main__":
    main()