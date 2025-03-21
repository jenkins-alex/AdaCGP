import gc
import time
import math
import tracemalloc
import numpy as np

from tqdm import tqdm

class SDSEM:
    """
    Implementation of Switching Dynamic Structureal Equation Models from
    Baingana, B., & Giannakis, G. B. (2016).
    Tracking switched dynamic network topologies from information cascades.
    IEEE Transactions on Signal Processing, 65(4), 985-997.

    Code adapted from https://github.com/bbaingana/dynetinf

    Implemented for S=1, i.e. no switching between different weight matrices.
    """
    def __init__(self, N, hyperparams, device):
        self.N = N
        self.set_hyperparameters(hyperparams)
        self._forget_t = 0.0

    def set_hyperparameters(self, hyperparams):
        for param, value in hyperparams.items():
            setattr(self, f"_{param}", value)
        if not hasattr(self, '_use_eig_stepsize'):
            self._use_eig_stepsize = True
        if not hasattr(self, '_record_complexity'):
            self._record_complexity = False

    def softThresh(self, M, mu):
        P = np.asmatrix(np.zeros(M.shape))
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                P[i,j] = np.sign(M[i,j])*max(abs(M[i,j]) - mu, 0)
        return P

    def fista(self, X, Ptau, Qtau, Ahat, Ahat_old, \
            bhat, bhat_old, forget_t, lamb_t, kmax):
        """ fista() tracks the sequence of unknown graphs captured by
        adjacency matrices, over which cascade data Yt propagate"""

        N = Qtau.shape[0]
        K = Qtau.shape[1]

        t_seq_old = 1
        t_seq = (1 + math.sqrt(1 + 4*(t_seq_old**2)))/2 

        # Compute Lipschitz constant
        M1 = np.hstack((Ptau, Qtau*X.T))
        M2 = np.hstack((X*Qtau.T, forget_t*X*X.T))
        M3 = np.vstack((M1, M2))
        if self._use_eig_stepsize:
            L = self.maxEigVal(M3)
            stepsize = 1.0 / L
        else:
            stepsize = self._stepsize

        result_dict = {}

        for k in range(kmax):
            for i in range(N):
                curr = [i]
                indices = list(set(range(N)).difference(set(curr))) 

                # Variables using accelerating combination of last two iterates
                b_ii = bhat[i, 0] + ((t_seq_old-1)/t_seq)*(bhat[i, 0] - bhat_old[i, 0])
                a_i = Ahat[i, :] + ((t_seq_old-1)/t_seq)*(Ahat[i, :] - Ahat_old[i, :])
                a_i_tilde = a_i[:, indices].T

                # Auxiliary quantities
                p_t = Ptau[:, i]
                p_ti = p_t[indices, :]

                q_t = Qtau[i, :]
                P_ti = Ptau[indices, :]
                P_ti = P_ti[:, indices]

                Q_ti = Qtau[indices, :]
                x_i = X[i, :].T

                # Step 1: compute gradients
                nablaf_ai = (-1.0)*(p_ti - P_ti*a_i_tilde - Q_ti*x_i*b_ii)

                nablaf_bii = (-1.0)*(q_t*x_i - a_i_tilde.T*Q_ti*x_i - \
                        forget_t*b_ii*(np.linalg.norm(x_i)**2))

                # Step 2: update B (gradient descent)
                bhat_old[i, 0] = bhat[i, 0]

                bhat[i, 0] = b_ii - stepsize * nablaf_bii[0,0]

                # Step 3: update A (gradient descent + soft-thresholding)
                a_i_tilde = self.softThresh(a_i_tilde-stepsize*nablaf_ai, lamb_t/L)
                Ahat_old[i, :] = Ahat[i, :]


                Ahat[i, :] = np.hstack((a_i_tilde[0:i, :].T, \
                        np.asmatrix(np.zeros((1,1))), \
                        a_i_tilde[i:, :].T))

            t_seq_old = t_seq
            t_seq = (1 + math.sqrt(1 + 4*(t_seq_old**2)))/2
            
        result_dict['Ahat'] = Ahat
        result_dict['bhat'] = bhat
        return result_dict

    def maxEigVal(self, M):
        eigvals, eigvecs = np.linalg.eig(M)
        return max(eigvals)

    def run(self, y, weight_matrix=None, **kwargs):
        # This function computes an estimate via TISO

        results = {
            'pred_error': [], 'w_error': [], 'matrices': [],
            'percentage_correct_elements': [], 'num_non_zero_elements': [],
            'p_miss': [], 'p_false_alarm': [], 'pred_error_recursive_moving_average': [0]
        }
        if self._record_complexity:
            results['iteration_time'] = []
            results['iteration_memory'] = []

        # init params
        patience_left = self._patience
        y = np.array(y)
        weight_matrix = np.array(weight_matrix) if weight_matrix is not None else None
        T, N, C = y.shape
        assert C == 1, "Only univariate time series are supported"

        # initialization
        bhat = np.asmatrix(np.zeros((N, 1)))
        bhat_old = bhat.copy()
        Ahat = np.asmatrix(np.ones((N, N)))
        Ahat_old = Ahat.copy()

        lowest_error = 1e10
        
        with tqdm(range(T)) as pbar:
            for t in pbar:

                # start measuring iteration memory and time complexity
                if self._record_complexity:
                    gc.collect()
                    tracemalloc.start()
                    start_time = time.time()

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

                # extract data from the y array
                Yt = np.asmatrix(y[t, :, :])  # Shape (N, 1)
                m_y = y[:, :, 0]  # Shape (N,)
                X = np.zeros_like(np.asmatrix(np.zeros((N, C))))
                
                # calculate Pt (outer product of current values)
                Pt = Yt @ Yt.T
                self._forget_t = 1 + self._forget * self._forget_t
                lamb_t = self._lamb

                # recursive data updates
                if t == 0:
                    Ptau = Pt
                    Qtau = Yt
                else:
                    Ptau = self._forget * Ptau + Pt
                    Qtau = self._forget * Qtau + Yt

                ##################################
                ######### COMPUTE W ##############
                ##################################
                result_dict = self.fista(X, Ptau, Qtau, Ahat, Ahat_old, 
                                bhat, bhat_old, self._forget_t, lamb_t, self._kmax)
                W = np.array(result_dict['Ahat'])
                b = np.array(result_dict['bhat']).flatten()

                # Compute squared error of prediction
                y_pred = W @ m_y[t] + np.diag(b) @ X

                # end measuring iteration memory and time complexity
                if self._record_complexity:
                    end_time = time.time()
                    _, peak_size = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    execution_time = end_time - start_time
                    results['iteration_time'].append(execution_time)
                    results['iteration_memory'].append(peak_size / (1024 * 1024))

                ##################################
                ######### COMPUTE ERRORS #########
                ##################################
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