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
from src.models.adaptive.TISO import TISO
from src.models.adaptive.TIRSO import TIRSO
from src.models.adaptive.SDSEM import SDSEM
from src.models.batch.GLasso import GLasso
from src.models.batch.GLSigRep import GLSigRep
from src.models.batch.GrangerVAR import GrangerVAR
from src.models.batch.VAR import VAR
from src.models.batch.PMIME import PMIME
from src.eval_metrics import save_results

def get_model(name):
    models = {
        'AdaCGP': AdaCGP,
        'TISO': TISO,
        'TIRSO': TIRSO,
        'SDSEM': SDSEM,
        'GLasso': GLasso,
        'GLSigRep': GLSigRep,
        'GrangerVAR': GrangerVAR,
        'VAR': VAR,
        'PMIME': PMIME
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
        save_path = get_save_path()
        error_metric = save_results(cfg.model.name, cfg.model.hyperparams.patience, results, save_path, cfg.get('dump_results', False))

        if results[error_metric][-1] != results[error_metric][-1]:
            return 2
        return 1
    except Exception as e:
        print(e)
        # print a stack trace
        import traceback
        traceback.print_exc()

        return 2

if __name__ == "__main__":
    main()