import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.api import VAR as VAR_model


class VAR:
    """
    Implementation of VAR model for time series forecasting and causal inference.
    """
    def __init__(self, N, hyperparams, device):
        self.N = N
        self.set_hyperparameters(hyperparams)

    def set_hyperparameters(self, hyperparams):
        for param, value in hyperparams.items():
            setattr(self, f"_{param}", value)

    def identify_causal_W(self, var_model_out):
        # VAR causality
        coefs = var_model_out.coefs
        coefs_matrix = np.stack([coefs[lag] for lag in range(self._P)])
        causal = ((coefs_matrix == 0).sum(axis=0)) != self._P
        W = np.linalg.norm(coefs_matrix, ord=2, axis=0) * causal  # use magnitude of psi as weights
        return W

    def predict_next_step(self, data, t):
        N = data.shape[1]
        train_data = data[:t, :]  # Current time steps

        # Create a pandas DataFrame for the VAR model
        df = pd.DataFrame(train_data, columns=[f'var_{i}' for i in range(N)])

        model = VAR_model(df)
        results = model.fit(self._P)

        W = self.identify_causal_W(results)
        last_observations = train_data[-self._P:, :]
        y_pred = results.forecast(last_observations, steps=1)[0]
        return W, y_pred
        
    def run(self, y, weight_matrix=None, **kwargs):
        # This function computes an estimate via TISO

        results = {
            'pred_error': [], 'w_error': [], 'matrices': [],
            'percentage_correct_elements': [], 'num_non_zero_elements': [],
            'p_miss': [], 'p_false_alarm': [], 'pred_error_recursive_moving_average': [1]
        }

        # init params
        lowest_error = 1e10
        y = np.array(y)
        weight_matrix = np.array(weight_matrix) if weight_matrix is not None else None
        m_y = y[:, :, 0]
        T, N = m_y.shape

        with tqdm(range(self._min_samples, T)) as pbar:
            for t in pbar:
                ma_error = results['pred_error_recursive_moving_average'][-1]
                
                ##################################
                ######### CHECK CONVERGENCE ######
                ##################################
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
                ######### COMPUTE W ##############
                ##################################
                try:
                    W, y_pred = self.predict_next_step(m_y, t)
                except Exception as e:
                    # use empty prediction and weight matrix
                    W = np.zeros((N, N))
                    y_pred = np.zeros(N)

                ##################################
                ######### COMPUTE ERRORS #########
                ##################################

                # Compute squared error of signal forecast from graph filters
                e = m_y[t] - y_pred
                norm_error = np.linalg.norm(e)**2 / np.linalg.norm(m_y[t])**2
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