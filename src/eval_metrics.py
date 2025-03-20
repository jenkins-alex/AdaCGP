import os
import pickle
import numpy as np

def save_results(model_name, patience, results, save_path, dump_results=False):
    if dump_results:
        fpath = os.path.join(save_path, 'results.pkl')
        with open(fpath, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {fpath}")

    if model_name == 'AdaCGP':
        # compute the average errors in steady state and save to eval_results.txt
        alg2_in_steady_state = np.array(results['second_alg_converged_status'])
        alg1_status = np.array(results['first_alg_converged_status'])
        alg1_in_steady_state = np.zeros_like(alg1_status, dtype=bool)
        alg1_converged_inds = np.where(alg1_status)[0]
        alg1_converged_ind = alg1_converged_inds[0] if len(alg1_converged_inds) > 0 else None
        if alg1_converged_ind is not None:
            alg1_in_steady_state[alg1_converged_ind-patience:alg1_converged_ind] = True

        if True not in alg1_in_steady_state and True not in alg2_in_steady_state:
            alg1_in_steady_state[-patience:] = True
            alg2_in_steady_state[-patience:] = True

        metrics = {
            'nmse_pred': np.array(results['pred_error']),
            'nmse_pred_from_h': np.array(results['pred_error_from_h']),
            'nmse_w': np.array(results['w_error']),
            'pce': np.array(results['percentage_correct_elements']),
            'p_miss': np.array(results['p_miss']),
            'p_false_alarm': np.array(results['p_false_alarm'])
        }

        algorithms = {
            'alg1': alg1_in_steady_state,
            'alg2': alg2_in_steady_state
        }

        with open(os.path.join(save_path, 'eval_results.txt'), 'w') as f:
            for alg_name, alg_state in algorithms.items():
                for metric_name, metric_values in metrics.items():
                    mean_value = np.mean(metric_values[alg_state])
                    f.write(f"{metric_name}_{alg_name}: {mean_value:.9f}\n")
        return 'pred_error'

    elif model_name in ['TISO', 'TIRSO', 'SDSEM', 'GLasso', 'GLSigRep', 'GrangerVAR', 'VAR', 'PMIME']:
        # compute the average errors for the patience period
        pred_error = np.array(results['pred_error'])
        alg1_in_steady_state = np.zeros_like(pred_error, dtype=bool)
        alg1_in_steady_state[-patience:] = True

        metrics = {
            'nmse_pred': np.array(results['pred_error']),
            'nmse_w': np.array(results['w_error']),
            'pce': np.array(results['percentage_correct_elements']),
            'p_miss': np.array(results['p_miss']),
            'p_false_alarm': np.array(results['p_false_alarm'])
        }

        algorithms = {
            'alg1': alg1_in_steady_state
        }

        with open(os.path.join(save_path, 'eval_results.txt'), 'w') as f:
            for alg_name, alg_state in algorithms.items():
                for metric_name, metric_values in metrics.items():
                    mean_value = np.mean(metric_values[alg_state])
                    f.write(f"{metric_name}_{alg_name}: {mean_value:.9f}\n")
        return 'pred_error'

    else:
        raise ValueError(f"Model {model_name} not implemented")
