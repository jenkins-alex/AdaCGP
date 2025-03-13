import numpy as np
from tqdm import tqdm
from sklearn import linear_model

class NodeLasso:
    def __init__(self, N, hyperparams, device):
        self.N = N
        self.set_hyperparameters(hyperparams)

    def set_hyperparameters(self, hyperparams):
        for param, value in hyperparams.items():
            setattr(self, f"_{param}", value)
    
    def predict_next_step(self, data, t):
        N = data.shape[1]

        X_window = data[t-self._P_cov-1:t-1, :]  # Previous time steps
        y_window = data[t-self._P_cov:t, :]  # Current time steps

        # Current last observation (for prediction)
        last_observation = data[t-1:t, :]

        # Initialize coefficient matrix and predictions
        beta_matrix = np.zeros((N, N))
        next_step_prediction = np.zeros(N)

        # For each target node, perform Lasso regression
        for target_node in range(N):
            # Target values are the current values of the target node
            y = y_window[:, target_node]

            # Predictors are the previous values of all nodes
            X = X_window

            # Fit Lasso regression
            lasso = linear_model.Lasso(alpha=self._alpha, fit_intercept=True)
            lasso.fit(X, y)

            # Store coefficients (representing edge weights)
            beta_matrix[target_node, :] = lasso.coef_

            # Predict next value for this node
            next_step_prediction[target_node] = lasso.predict(last_observation)[0]

        return beta_matrix, next_step_prediction
        
    def run(self, y, weight_matrix=None, **kwargs):
        # This function computes an estimate via TISO

        results = {
            'pred_error': [], 'w_error': [], 'matrices': [],
            'percentage_correct_elements': [], 'num_non_zero_elements': [],
            'p_miss': [], 'p_false_alarm': [], 'pred_error_recursive_moving_average': [1]
        }

        lowest_error = 1e10
        patience_left = self._patience

        # init params
        y = np.array(y)
        weight_matrix = np.array(weight_matrix) if weight_matrix is not None else None
        m_y = y[:, :, 0]
        T, N = m_y.shape

        with tqdm(range(self._P_cov+1, T)) as pbar:
            for t in pbar:  # in paper, t=P,...
                # receive data y[t]
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
                W, y_pred = self.predict_next_step(m_y, t)

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