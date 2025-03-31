import gc
import time
import tracemalloc
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.api import VAR as VAR_model

class GrangerVAR:
    """
    Implementation of Granger VAR model for time series forecasting and causal inference.
    Uses faster VAR causality during training and computes expensive Granger causality only at the end.
    """
    def __init__(self, N, hyperparams, device):
        self.N = N
        self.set_hyperparameters(hyperparams)
        
    def set_hyperparameters(self, hyperparams):
        for param, value in hyperparams.items():
            setattr(self, f"_{param}", value)
        if not hasattr(self, "_gc_window"):
            self._gc_window = 20  # Granger causality evaluation window size
        if not hasattr(self, '_use_gc_during_training'):
            self._use_gc_during_training = True
        if not hasattr(self, '_train_steps_list'):
            self._train_steps_list = None
        if not hasattr(self, '_record_complexity'):
            self._record_complexity = False

    def identify_var_causal_W(self, var_model_out):
        # VAR causality - faster method
        coefs = var_model_out.coefs
        coefs_matrix = np.stack([coefs[lag] for lag in range(self._P)])
        causal = ((coefs_matrix == 0).sum(axis=0)) != self._P
        W = np.linalg.norm(coefs_matrix, ord=2, axis=0) * causal  # use magnitude of psi as weights
        return W

    def identify_granger_causal_W(self, var_model_out):
        # Granger causality - computationally expensive
        W = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                # Test if variable j Granger-causes variable i
                causality_test = var_model_out.test_causality(f'var_{i}', [f'var_{j}'], kind='wald')
                p_value = causality_test.pvalue
                
                # If p-value is below threshold (default 0.05), consider it causal
                if p_value < 0.05:
                    # Get coefficients for all lags of variable j affecting variable i
                    coefs = np.array([var_model_out.coefs[lag][i, j] for lag in range(self._P)])
                    
                    # Use the norm of coefficients as the weight
                    W[i, j] = np.linalg.norm(coefs)
        return W
    
    def predict_next_step(self, data, t):
        N = data.shape[1]
        train_data = data[:t, :]  # Current time steps

        # Create a pandas DataFrame for the VAR model
        df = pd.DataFrame(train_data, columns=[f'var_{i}' for i in range(N)])

        model = VAR_model(df)
        results = model.fit(self._P)

        # Use faster VAR causality during training
        if self._use_gc_during_training:
            W = self.identify_granger_causal_W(results)
        else:
            W = self.identify_var_causal_W(results)
        last_observations = train_data[-self._P:, :]
        y_pred = results.forecast(last_observations, steps=1)[0]
        return W, y_pred, results  # Return the model results for later GC testing

    def run(self, y, weight_matrix=None, **kwargs):
        # This function computes an estimate via VAR with postponed GC testing

        results = {
            'pred_error': [], 'w_error': [], 'matrices': [],
            'percentage_correct_elements': [], 'num_non_zero_elements': [],
            'p_miss': [], 'p_false_alarm': [], 'pred_error_recursive_moving_average': [1],
            'gc_window': self._gc_window
        }
        if self._record_complexity:
            results['iteration_time'] = []
            results['iteration_memory'] = []

        # initialize a ring buffer for storing the most recent VAR model outputs
        var_model_outputs = []
        buffer_size = self._gc_window  # validation window for GC testing
        
        # init params
        lowest_error = 1e10
        patience_left = self._patience
        y = np.array(y)
        weight_matrix = np.array(weight_matrix) if weight_matrix is not None else None
        m_y = y[:, :, 0] if y.ndim > 2 else y
        T, N = m_y.shape

        # training loop
        iter_range = range(self._min_samples, T) if self._train_steps_list is None else self._train_steps_list
        with tqdm(iter_range) as pbar:
            for t in pbar:

                # start measuring iteration memory and time complexity
                if self._record_complexity:
                    gc.collect()
                    tracemalloc.start()
                    start_time = time.process_time()

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
                    W, y_pred, var_out = self.predict_next_step(m_y, t)
                    
                    if not self._use_gc_during_training:
                        # Only keep recent models when approaching convergence
                        # This ensures we don't waste memory on early models
                        if patience_left < self._patience / 2:
                            # Store the VAR model output in the ring buffer
                            if len(var_model_outputs) >= buffer_size:
                                var_model_outputs.pop(0)  # Remove oldest
                            var_model_outputs.append((t, var_out))

                except Exception as e:
                    # use empty prediction and weight matrix
                    W = np.zeros((N, N))
                    y_pred = np.zeros(N)

                # end measuring iteration memory and time complexity
                if self._record_complexity:
                    end_time = time.process_time()
                    _, peak_size = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    execution_time = end_time - start_time
                    results['iteration_time'].append(execution_time)
                    results['iteration_memory'].append(peak_size / (1024 * 1024))

                ##################################
                ######### COMPUTE ERRORS #########
                ##################################

                # Compute squared error of signal forecast from graph filters
                e = m_y[t] - y_pred
                norm_error = np.linalg.norm(e)**2 / np.linalg.norm(m_y[t])**2
                results['pred_error'].append(norm_error)
                ma_error = self._ma_alpha * norm_error + (1 - self._ma_alpha) * results['pred_error_recursive_moving_average'][-1]
                results['pred_error_recursive_moving_average'].append(ma_error)
        
                # Compute squared error of W estimation (using faster VAR causality during training)
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

        if not self._use_gc_during_training:
            # Compute the final matrix using the faster method for consistency with training
            final_W_var = W
            results['matrices'].append(final_W_var)
            
            # Create temporary storage for GC results
            gc_results = {
                'w_error': [], 'percentage_correct_elements': [], 
                'num_non_zero_elements': [], 'p_miss': [], 'p_false_alarm': []
            }
            
            # Process the stored VAR model outputs with GC testing
            for t_idx, (t, var_out) in enumerate(var_model_outputs):
                try:
                    W_gc = self.identify_granger_causal_W(var_out)
                    
                    # Compute metrics using GC-based weights
                    if weight_matrix is not None:
                        weight_matrix_error = weight_matrix - W_gc
                        norm_w_error = np.linalg.norm(weight_matrix_error)**2 / np.linalg.norm(weight_matrix)**2
                        gc_results['w_error'].append(norm_w_error)
                        gc_results['num_non_zero_elements'].append((W_gc != 0).sum())

                        # Compute percentage of elements correctly identified in W
                        total = (weight_matrix != 0).sum()
                        frac = ((W_gc != 0) * (weight_matrix != 0)).sum() / total
                        gc_results['percentage_correct_elements'].append(frac)

                        # Save results for p_miss and p_false_alarm
                        gc_results['p_miss'].append(((W_gc == 0) * (weight_matrix != 0)).sum().item() / (weight_matrix != 0).sum().item())
                        gc_results['p_false_alarm'].append(((W_gc != 0) * (weight_matrix == 0)).sum().item() / (weight_matrix == 0).sum().item())
                
                except Exception as e:
                    continue
            
            # Replace the last 'gc_window' entries with GC-based metrics if available
            if gc_results['w_error']:
                # Calculate how many entries to replace (minimum of length of gc_results or gc_window)
                replace_count = min(len(gc_results['w_error']), self._gc_window)
                start_idx = len(results['w_error']) - replace_count
                
                # Replace the metrics for the last 'gc_window' number of entries
                for key in ['w_error', 'percentage_correct_elements', 'num_non_zero_elements', 'p_miss', 'p_false_alarm']:
                    for i in range(replace_count):
                        if start_idx + i < len(results[key]) and i < len(gc_results[key]):
                            results[key][start_idx + i] = gc_results[key][i]
            
            # Compute final GC-based W matrix for the most recent model output
            if var_model_outputs:
                _, final_var_out = var_model_outputs[-1]
                final_W_gc = self.identify_granger_causal_W(final_var_out)
                # Replace the last matrix with the GC-based one
                results['matrices'][-1] = final_W_gc

        return results