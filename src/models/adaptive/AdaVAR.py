import torch

from tqdm import tqdm
from src.line_search import wolfe_line_search
from src.utils import get_each_graph_filter, pack_graph_filters

class AdaVAR:
    """Adaptive VAR model using our online sparsity algorithm
    """
    def __init__(self, N, hyperparams, device):
        self.N = N
        self.device = device
        self.set_hyperparameters(hyperparams)
        self.initialize_parameters()

    def initialize_parameters(self):
        N, P, device = self.N, self._P, self.device
        
        self.Psi_pos = torch.zeros((N, N*P), dtype=torch.float32, device=device)
        self.Psi_neg = torch.zeros((N, N*P), dtype=torch.float32, device=device)
        self.Psi = torch.zeros((N, N*P), dtype=torch.float32, device=device)
        self.P0 = torch.zeros((N, N*P), dtype=torch.float32, device=device)
        self.Q = torch.zeros((N, N*P), dtype=torch.float32, device=device)
        self.R0 = torch.zeros((N*P, N*P), dtype=torch.float32, device=device)
        self.eye_NxN = torch.eye(N, N, device=device)
        self.ones_NxN = torch.ones_like(self.eye_NxN)
        self.eye_N = torch.eye(N, device=device)
        self.eye_P = torch.eye(P, device=device)
        self.mus = torch.tensor([self._mu]*P, dtype=torch.float32, device=device)

    def set_hyperparameters(self, hyperparams):
        for param, value in hyperparams.items():
            setattr(self, f"_{param}", value)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        private_name = f"_{name}"
        if hasattr(self, private_name):
            return getattr(self, private_name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def run(self, X, y, weight_matrix=None, filter_coefficients=None, graph_filter_matrix=None, **kwargs):
        """Run the AdaVAR model

        Args:
            X (torch.tensor): input data (T, N, P)
            y (torch.tensor): target data (T, N)
            weight_matrix (torch.tensor, optional): true weight matrix, if known. Defaults to None.
            filter_coefficients (torch.tensor, optional): true filter coefficients if known. Defaults to None.
            graph_filter_matrix (torch.tensor, optional): true graph filter matrix if known. Defaults to None.
        """

        results = {
            'pred_error': [], 'w_error': [], 'matrices': [],
            'percentage_correct_elements': [], 'num_non_zero_elements': [],
            'p_miss': [], 'p_false_alarm': [], 'pred_error_recursive_moving_average': [1]
        }
 
        lowest_error = 1e10
        process_length = X.shape[0]
        psi_loss = 0.0
        with torch.no_grad():
            with tqdm(range(process_length)) as pbar:
                for t in pbar:
                    ##################################
                    ######### GET DATA AT T ##########
                    ##################################

                    xPt = X[t, :, :].flatten()[:, None]  # (NP, 1)
                    yt = y[t]  # (N, 1)

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

                    # Psi loss update
                    psi_loss = 0.5 * (self._lambda * psi_loss + torch.norm(yt - torch.matmul(self.Psi, xPt))**2)
            
                    ############################################
                    ############## COMPUTE PSI #################
                    ############################################
        
                    self.R0 = self._lambda * self.R0 + torch.matmul(xPt, xPt.T)
                    self.P0 = self._lambda * self.P0 + torch.matmul(yt, xPt.T)
        
                    # Compute mus
                    # mu_scales = []
                    # P0_unpacked = get_each_graph_filter(self.P0, self.N, self._P)
                    # for p in range(0, self._P):
                    #     P0p = P0_unpacked[:, p, :]
                    #     infty_norm = torch.norm(P0p, p=float('inf'))
                    #     mu_scales.append(infty_norm)

                    mus_pt = self.mus# * torch.stack(mu_scales)
                    M = torch.vstack([self.ones_NxN * mus_pt[p] for p in range(0, self._P)]).T  # (N, N*P)
                    G = torch.matmul(self.Psi, self.R0) - self.P0
                    
                    # set maximum stepsize using max eigenvalue of autocorrelation matrix
                    try:
                        eigs = torch.lobpcg(self.R0, largest=True)
                        stepsize = 2 / (eigs[0].item())
                        stepsize /= (torch.linalg.norm(xPt, ord=2)**2 + self._epsilon)
                        A = self.eye_P * stepsize
                        # for p in range(self._P):
                        #     A[p, p] /= (torch.linalg.norm(xPt[p*self.N:(p+1)*self.N], ord=2)**2 + self._epsilon)
                    except:
                        stepsize = self._default_stepsize
                        A = self.eye_P * stepsize
        
                    ######### UPDATE PARAM #########
                    dPsi_pos = - (M + G) @ torch.kron(A, self.eye_N)
                    dPsi_neg = - (M - G) @ torch.kron(A, self.eye_N)
                    self.Psi_pos = self.Psi_pos + dPsi_pos
                    self.Psi_neg = self.Psi_neg + dPsi_neg

                    # projection onto non-negative space
                    self.Psi_pos[self.Psi_pos < 0] = 0
                    self.Psi_neg[self.Psi_neg < 0] = 0
                    self.Psi = self.Psi_pos - self.Psi_neg
            
                    ##################################
                    ######### COMPUTE W ##############
                    ##################################

                    # compute the causal graph from the VAR parameters as described in paper
                    psi = get_each_graph_filter(self.Psi, self.N, self._P)
                    causal = ((psi == 0).sum(axis=1)) != self._P
                    W = torch.linalg.norm(psi, ord=2, axis=1) * causal  # use magnitude of psi as weights
                    
                    ###################################
                    ######### COMPUTE RESULTS #########
                    ###################################

                    # Compute squared error of signal forecast from graph filters
                    d_hat_psi = torch.matmul(self.Psi, xPt)
                    e = yt - d_hat_psi
                    norm_error = torch.norm(e)**2 / torch.norm(yt)**2
                    results['pred_error'].append(norm_error.item())
                    ma_error = self._ma_alpha * norm_error + (1 - self._ma_alpha) * results['pred_error_recursive_moving_average'][-1]
                    results['pred_error_recursive_moving_average'].append(ma_error.item())

                    # Compute squared error of W estimation
                    if weight_matrix is not None:
                        weight_matrix_error = weight_matrix - W
                        norm_w_error = torch.norm(weight_matrix_error)**2 / torch.norm(weight_matrix)**2
                        results['w_error'].append(norm_w_error.item())
                        results['num_non_zero_elements'].append((W != 0).sum().item())

                        # compute the percentage of elements correctly identified in W
                        total = (weight_matrix != 0).sum()
                        frac = ((W != 0) * (weight_matrix != 0)).sum() / total
                        results['percentage_correct_elements'].append(frac.item())

                        # save results for p_miss: probability of missing a non-zero element in W
                        results['p_miss'].append(((W == 0) * (weight_matrix != 0)).sum().item() / (weight_matrix != 0).sum().item())
                        results['p_false_alarm'].append(((W != 0) * (weight_matrix == 0)).sum().item() / (weight_matrix == 0).sum().item())
                    pbar.set_postfix({'MA y error': ma_error.item(), 'MA W error': results['w_error'][-1]})
        results['matrices'].append(W.detach().cpu().numpy())
        return results

def comm(A, B):
    return torch.matmul(A, B) - torch.matmul(B, A)
 
def comm_term_mnlms(W, p, P):
    comm_terms = []
    for k in range(P):
        comm_ = torch.matmul(comm(W[:, p, :], W[:, k, :]), W[:, k, :].T) + torch.matmul(W[:, k, :].T, comm(W[:, p, :], W[:, k, :]))
        comm_terms.append(comm_)
    return torch.sum(torch.stack(comm_terms), dim=0)
 
def second_comm_term_mnlms(W, A, P):
    comm_terms = []
    for k in range(P):
        comm_ = torch.matmul(comm(A, W[:, k, :]), W[:, k, :].T) + torch.matmul(W[:, k, :].T, comm(A, W[:, k, :]))
        comm_terms.append(comm_)
    return torch.sum(torch.stack(comm_terms), dim=0)
 
def gradient_function_psi_mlms(psi_p_flat, Psi, mus_pt, R0, P0, gamma_, p, include_comm_term, N, P, lambda_, xPt, yt, cumulative_loss, **kwargs):
    Psi_unpacked = get_each_graph_filter(Psi, N, P)
    psi_p = psi_p_flat.reshape(N, N)
    Psi_unpacked[:, p, :] = psi_p
    Psi = pack_graph_filters([Psi_unpacked[:, i, :] for i in range(P)], N, P)
 
    if include_comm_term:
        Qs = []
        Psi_unpacked = get_each_graph_filter(Psi, N, P)
        for p in range(P):
            Qp = comm_term_mnlms(Psi_unpacked, p, P)
            Qs.append(Qp)
        Q = pack_graph_filters(Qs, N, P)
        G = torch.matmul(Psi, R0) - (P0 - gamma_ * Q)
    else:
        G = torch.matmul(Psi, R0) - P0
    
    G_unpacked = get_each_graph_filter(G, N, P)
    gp = G_unpacked[:, p, :]
    return gp.flatten() # (N*N,)
 
def objective_function_psi_mlms(psi_p_flat, Psi, mus_pt, R0, P0, gamma_, p, include_comm_term, N, P, lambda_, xPt, yt, cumulative_loss, **kwargs):
    Psi_unpacked = get_each_graph_filter(Psi, N, P)
    psi_p = psi_p_flat.reshape(N, N)
    Psi_unpacked[:, p, :] = psi_p
    Psi = pack_graph_filters([Psi_unpacked[:, i, :] for i in range(P)], N, P)
    psi_loss = 0.5 * (lambda_ * cumulative_loss + torch.norm(yt - torch.matmul(Psi, xPt))**2)
    Psi_unpacked = get_each_graph_filter(Psi, N, P)
 
    for i in range(P):
        Psi_p = Psi_unpacked[:, i, :]
        mu = mus_pt[i]
        psi_loss += mu * torch.norm(Psi_p.flatten(), p=1)
 
        if include_comm_term:
            for j in range(P):
                if i == j:
                    continue
                psi_loss += gamma_ * torch.norm(comm(Psi_unpacked[:, i, :], Psi_unpacked[:, j, :]), p='fro')**2
 
    return psi_loss
 
def update_function_psi_mlms(psi_p_flat, G, step, Psi, mus_pt, R0, P0, gamma_, p, include_comm_term, N, P, lambda_, xPt, yt, cumulative_loss, **kwargs):
    new_param = psi_p_flat - step * G
    return new_param
 
def gradient_function_psi_debias(psi_p_flat, Psi, mus_pt, R0, P0, gamma_, p, include_comm_term, N, P, lambda_, xPt, yt, cumulative_loss, mask_W, **kwargs):
    Psi_unpacked = get_each_graph_filter(Psi, N, P)
    psi_p = psi_p_flat.reshape(N, N)
    Psi_unpacked[:, p, :] = psi_p
    Psi = pack_graph_filters([Psi_unpacked[:, i, :] for i in range(P)], N, P)
 
    G = torch.matmul(Psi, R0) - P0
    masks = []
    for i in range(P):
        mask = torch.matrix_power(mask_W, i+1)
        masks.append(mask)
    mask = pack_graph_filters(masks, N, P)
    G[mask == 0] = 0
    
    G_unpacked = get_each_graph_filter(G, N, P)
    gp = G_unpacked[:, p, :]
    return gp.flatten() # (N*N,)
 
def objective_function_psi_debias(psi_p_flat, Psi, mus_pt, R0, P0, gamma_, p, include_comm_term, N, P, lambda_, xPt, yt, cumulative_loss, mask_W, **kwargs):
    Psi_unpacked = get_each_graph_filter(Psi, N, P)
    psi_p = psi_p_flat.reshape(N, N)
    Psi_unpacked[:, p, :] = psi_p
    Psi = pack_graph_filters([Psi_unpacked[:, i, :] for i in range(P)], N, P)
    psi_loss = 0.5 * (lambda_ * cumulative_loss + torch.norm(yt - torch.matmul(Psi, xPt))**2)
    Psi_unpacked = get_each_graph_filter(Psi, N, P)
 
    for i in range(P):
        Psi_p = Psi_unpacked[:, i, :]
        mu = mus_pt[i]
        psi_loss += mu * torch.norm(Psi_p, p=1)
    return psi_loss
 
def update_function_psi_debias(psi_p_flat, G, step, Psi, mus_pt, R0, P0, gamma_, p, include_comm_term, N, P, lambda_, xPt, yt, cumulative_loss, mask_W, **kwargs):
    new_param = psi_p_flat - step * G
    return new_param

def objective_function_wstep2_lms(W_flat, Psi, mus_pt, gamma_, N, P, **kwargs):
    Psi_unpacked = get_each_graph_filter(Psi, N, P)
    W = W_flat.reshape(N, N)
    Psi_1 = Psi_unpacked[:, 0, :]
    
    wstep2_loss = 0
    wstep2_loss += 0.5 * torch.norm(Psi_1 - W)**2
    wstep2_loss += mus_pt[0] * torch.norm(W, p=1)
    for i in range(1, P):
        wstep2_loss += gamma_ * torch.norm(comm(W, Psi_unpacked[:, i, :]), p='fro')**2
    return wstep2_loss
 
def gradient_function_wstep2_lms(W_flat, Psi, mus_pt, gamma_, N, P, **kwargs):
    Psi_unpacked = get_each_graph_filter(Psi, N, P)
    Psi_1 = Psi_unpacked[:, 0, :]
    W = W_flat.reshape(N, N)
 
    # compute S
    Ss = []
    Psi_unpacked = get_each_graph_filter(Psi, N, P)
    S = second_comm_term_mnlms(Psi_unpacked, W, P)
 
    # Compute gradient
    V = W - (Psi_1 - gamma_ * S)
    return V.flatten()
 
def update_function_wstep2_lms(W_flat, G, step, Psi, mus_pt, gamma_, N, P, **kwargs):
    new_param = W_flat - step * G
    return new_param
 
def objective_function_h_lms(h_flat, Ys, yt, lambda_, C, u, instant_h, nu_t, epsilon):
    h = h_flat.reshape(-1, 1)
    loss = 0
    loss += 0.5 * torch.norm(yt - torch.matmul(Ys, h))**2
    loss += nu_t * torch.norm(h, p=1)
    return loss
 
def gradient_function_h_lms(h_flat, Ys, yt, lambda_, C, u, instant_h, nu_t, epsilon):
    h = h_flat.reshape(-1, 1)
    b = torch.sign(h) / (epsilon + h)
 
    if instant_h:
        h_e = yt - torch.matmul(Ys, h)
        h_g = torch.matmul(Ys.T, h_e)
    else:
        C = lambda_ * C + torch.matmul(Ys.T, Ys)
        u = lambda_ * u + torch.matmul(Ys.T, yt)
        h_g = torch.matmul(C, h) - u
    return (h_g + nu_t * b).flatten()
 
def update_function_h_lms(h_flat, dh, h_stepsize, Ys, yt, lambda_, C, u, instant_h, nu_t, epsilon):
    new_param = h_flat + h_stepsize * dh
    # new_param[0] = 0
    # new_param[1] = 1
    return new_param
