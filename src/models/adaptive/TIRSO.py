import torch
import numpy as np
from tqdm import tqdm
from src.utils import get_each_graph_filter

import numpy as np

class TIRSO:
    def __init__(self, N, hyperparams, device):
        self.N = N
        self.set_hyperparameters(hyperparams)
        self.initialize_parameters()

    def initialize_parameters(self):
        N, P = self.N, self._P
        self.m_A_initial = np.zeros((N, N * P))

    def set_hyperparameters(self, hyperparams):
        for param, value in hyperparams.items():
            setattr(self, f"_{param}", value)
        self._mu = 1 - self._gamma
        
    def run(self, y, weight_matrix=None, **kwargs):
        # This function computes an estimate via TISO

        results = {
            'pred_error': [], 'w_error': [], 'matrices': [],
            'percentage_correct_elements': [], 'num_non_zero_elements': [],
            'p_miss': [], 'p_false_alarm': [], 'pred_error_recursive_moving_average': [1]
        }

        lowest_error = 1e10
        a_prev = self.m_A_initial  # size N X NP

        # init params
        patience_left = self._patience
        y = np.array(y)
        weight_matrix = np.array(weight_matrix) if weight_matrix is not None else None
        m_y = y[:, :, 0].T
        N, T = m_y.shape
        
        assert a_prev.shape[0] == N and a_prev.shape[1] == N * self._P, 'A_initial should have of size N X NP'
        
        Phi_prev = self._sigma**2 * np.eye(N * self._P)  # initializing Phi
        r_prev = np.zeros((N * self._P, N))  # r has NP X N size to avoid transpose 
        m_r = np.zeros((N * self._P, N))
        t_A = np.zeros((N, N * self._P, T))

        with tqdm(range(self._P, T)) as pbar:
            for t in pbar:  # in paper, t=P,...
                # receive data y[t]
                # form g[t] via g[t]= vec([y[t-1],...,y[t-P]]^T)
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
                ######## TIRSO ALGORITHM ##########
                ##################################
                y_prev = m_y[:, t - self._P:t]
                aux = np.fliplr(y_prev).T
                g = aux.flatten()

                # compute stepsize
                R = np.outer(g, g)
                eigs = torch.lobpcg(torch.tensor(R), largest=True)
                stepsize = 2 / (eigs[0].item())
                stepsize /= (np.linalg.norm(g, ord=2)**2 + self._epsilon)

                # update Phi
                Phi_t = self._gamma * Phi_prev + self._mu * R
                
                if self._b_trace == 1:
                    if self._b_diminishing == 1:
                        stepsize = 1 / (np.trace(Phi_t) * np.sqrt(t))
                    else:
                        stepsize = 1 / np.trace(Phi_t)

                for n in range(N):
                    # update r_n
                    m_r[:, n] = self._gamma * r_prev[:, n] + self._mu * m_y[n, t] * g
                    grad_n = Phi_t @ a_prev[n, :].T - m_r[:, n]  # v_n in the paper

                    for nprime in range(N):
                        groupindices = slice((nprime - 1) * self._P, nprime * self._P)  # n,n' group indices
                        af_nnprime = a_prev[n, groupindices] - stepsize * grad_n[groupindices]
                        
                        if n != nprime:
                            t_A[n, groupindices, t] = np.maximum(0, (1 - (stepsize * self._lambda) / (np.linalg.norm(af_nnprime) + self._epsilon))) * af_nnprime  # indicator rem
                        else:
                            t_A[n, groupindices, t] = af_nnprime
                
                # to store A, Phi, and r_n
                a_prev = t_A[:, :, t]
                Phi_prev = Phi_t
                r_prev = m_r

                ##################################
                ######### COMPUTE W ##############
                ##################################

                # compute the causal graph from the VAR parameters as described in paper
                psi = get_each_graph_filter(torch.tensor(a_prev), N, self._P).numpy()
                causal = ((psi == 0).sum(axis=1)) != self._P
                W = np.linalg.norm(psi, ord=2, axis=1) * causal  # use magnitude of psi as weights
                
                ##################################
                ######### COMPUTE ERRORS #########
                ##################################

                # Compute squared error of signal forecast from graph filters
                e = m_y[:, t] - a_prev @ g
                norm_error = np.linalg.norm(e)**2 / np.linalg.norm(m_y[:, t])**2
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
        results['matrices'].append(W)
        return results