import numpy as np
from tqdm import tqdm
from sklearn.covariance import empirical_covariance, graphical_lasso

class GLasso:
    def __init__(self, N, hyperparams, device):
        self.N = N
        self.set_hyperparameters(hyperparams)

    def set_hyperparameters(self, hyperparams):
        for param, value in hyperparams.items():
            setattr(self, f"_{param}", value)
    
    def predict_topology(self, data, t):
        N = data.shape[1]

        X_window = data[:t, :]
        emp_cov = empirical_covariance(X_window, assume_centered=True)
        _, precision, costs = graphical_lasso(emp_cov.astype(float), alpha=self._alpha, return_costs=True)
        latest_obj_fn, _ = costs[-1]
        return precision, np.abs(latest_obj_fn)

    def run(self, y, weight_matrix=None, **kwargs):
        # This function computes an estimate via TISO

        results = {
            'pred_error': [], 'w_error': [], 'matrices': [],
            'percentage_correct_elements': [], 'num_non_zero_elements': [],
            'p_miss': [], 'p_false_alarm': [], 'pred_error_recursive_moving_average': []
        }

        # init params
        lowest_error = 1e10
        patience_left = self._patience
        y = np.array(y)
        weight_matrix = np.array(weight_matrix) if weight_matrix is not None else None
        m_y = y[:, :, 0]
        T, N = m_y.shape

        with tqdm(range(self._min_samples, T)) as pbar:
            for t in pbar:

                ##################################
                ######### COMPUTE W ##############
                ##################################
                try:
                    W, e = self.predict_topology(m_y, t)
                except Exception as exception:
                    W = np.zeros((N, N))
                    e = 0.0

                if len(results['pred_error_recursive_moving_average']) == 0:
                    results['pred_error_recursive_moving_average'].append(e)
                    continue
                
                ##################################
                ######### CHECK CONVERGENCE ######
                ##################################
                ma_error = results['pred_error_recursive_moving_average'][-1]
                if lowest_error != 0:
                    relative_improvement = (lowest_error - ma_error) / lowest_error
                else:
                    relative_improvement = float('inf') if ma_error < lowest_error else 0

                if relative_improvement > self._min_delta_percent:
                    lowest_error = ma_error
                    patience_left = self._patience
                else:
                    if t > self._patience:
                        patience_left -= 1

                if patience_left == 0:
                    break

                ##################################
                ######### COMPUTE ERRORS #########
                ##################################

                norm_error = e
                results['pred_error'].append(norm_error)
                ma_error = self._ma_alpha * norm_error + (1 - self._ma_alpha) * results['pred_error_recursive_moving_average'][-1]
                results['pred_error_recursive_moving_average'].append(ma_error)
        
                # Compute squared error of W estimation
                if weight_matrix is not None:
                    weight_matrix_error = weight_matrix - W
                    norm_w_error = np.linalg.norm(weight_matrix_error)**2 / np.linalg.norm(weight_matrix)**2
                    results['w_error'].append(norm_w_error)
                    results['num_non_zero_elements'].append((W != 0).sum())

                    # compute the percentage of elements correctly identified in W
                    total = (weight_matrix != 0).sum()
                    frac = ((W != 0) * (weight_matrix != 0)).sum() / total
                    results['percentage_correct_elements'].append(frac)

                    # save results for p_miss: probability of missing a non-zero element in W
                    results['p_miss'].append(((W == 0) * (weight_matrix != 0)).sum().item() / (weight_matrix != 0).sum().item())
                    results['p_false_alarm'].append(((W != 0) * (weight_matrix == 0)).sum().item() / (weight_matrix == 0).sum().item())
                pbar.set_postfix({'MA y error': ma_error})
                if 'append_all_matrices' in kwargs.keys() and kwargs['append_all_matrices']:
                    results['matrices'].append(W)
        results['matrices'].append(W)
        return results