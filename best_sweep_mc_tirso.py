import torch
import hydra
import pickle
import os
import pandas as pd

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

def read_sweep_eval_results(results_dir):
    # walk the results dir
    cols = ['graph_type', 'N', 'lambda', 'gamma', 'seed']
    numeric_cols = ['N', 'lambda', 'gamma', 'seed']
    results_cols = ['nmse_pred_alg1', 'nmse_w_alg1', 'pce_alg1', 'p_miss_alg1', 'p_false_alarm_alg1']
    all_cols = cols + results_cols
    data = {col: [] for col in all_cols}
    data['cfg_path'] = []

    # walk the results dir
    # with folders structured as in cols
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file == 'eval_results.txt':
                # extract the parameters
                parts = root.split('/')
                cfg_path = os.path.join(root, '.hydra', 'config.yaml')
                data['cfg_path'].append(cfg_path)
                data['graph_type'].append(parts[5].split('_')[-1])
                data['N'].append(parts[6].split('_')[-1])
                data['lambda'].append(parts[7].split('_')[-1])
                data['gamma'].append(parts[8].split('_')[-1])
                data['seed'].append(parts[9].split('_')[-1])

                with open(os.path.join(root, file), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        col, result = line.split(':')
                        col = col.strip()
                        result = result.strip()
                        data[col].append(result)
    return data, (cols, numeric_cols, results_cols, all_cols)

def get_best_as_pandas(data, metric, direction, numeric_cols, results_cols):
    df = pd.DataFrame(data)
    df[results_cols] = df[results_cols].astype(float).fillna(1e10)
    df[numeric_cols] = df[numeric_cols].astype(float)
    df = df[df['N'] <= 250]
    df_best = df.groupby(by=['graph_type', 'N']).apply(lambda x: x.sort_values(by=metric, ascending=True if direction == 'min' else False).head(1))
    return df_best

@hydra.main(version_base=None, config_path="config", config_name="config_best_sweep_mc_tirso")
def main(cfg_base: DictConfig):
    print(OmegaConf.to_yaml(cfg_base))

    # load the cfg of each best config
    data, (_, numeric_cols, results_cols, _) = read_sweep_eval_results(cfg_base.sweep_results_dir)
    df_best = get_best_as_pandas(data, cfg_base.optimise_metric, cfg_base.optimise_direction, numeric_cols, results_cols)

    # loop over each row in df_best
    for cfg_path in df_best['cfg_path']:
        cfg = OmegaConf.load(cfg_path)
        cfg.data_seed = cfg_base.data_seed
        cfg.model.hyperparams.patience = cfg_base.model.hyperparams.patience

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
            results['weight_matrix'] = weight_matrix

            # Save results
            dir, subdir = get_save_path()
            save_path = os.path.join(dir, subdir)
            save_path = os.path.join(save_path, f'graph_{cfg.graph.graph_type}', f'N_{cfg.data.N}')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            error_metric = save_results(cfg.model.name, cfg.model.hyperparams.patience, results, save_path, cfg.get('dump_results', True))

            if results[error_metric][-1] != results[error_metric][-1]:
                continue

        except Exception as e:
            print(e)
            continue
    return 1

if __name__ == "__main__":
    main()