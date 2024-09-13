import os
import pickle
import numpy as np

def save_results(patience, results, save_path):
    fpath = os.path.join(save_path, 'results.pkl')
    with open(fpath, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {fpath}")

    # compute the average errors in steady state and save to eval_results.txt
    alg2_in_steady_state = np.array(results['second_alg_converged_status'])
    alg1_status = np.array(results['first_alg_converged_status'])
    alg1_in_steady_state = np.zeros_like(alg1_status, dtype=bool)
    alg1_converged_ind = np.where(alg1_status)[0][0]
    alg1_in_steady_state[alg1_converged_ind-patience:alg1_converged_ind] = True

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
                f.write(f"{metric_name}_{alg_name}: {mean_value:.4f}\n")