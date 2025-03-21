import torch

from tqdm import tqdm
from src.line_search import wolfe_line_search
from src.utils import get_each_graph_filter, pack_graph_filters

class AdaCGP:
    """Adaptive Causal Graph Process model
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
        self.W = torch.zeros((N, N), dtype=torch.float32, device=device)
        self.W_pos = torch.zeros((N, N), dtype=torch.float32, device=device)
        self.W_neg = torch.zeros((N, N), dtype=torch.float32, device=device)
        self.M = int(P * (P + 3) / 2)
        self.h = torch.zeros((self.M, 1), dtype=torch.float32, device=device)
        # self.h[1] = 1
        self.C = torch.zeros((self.M, self.M), dtype=torch.float32, device=device)
        self.u = torch.zeros((self.M, 1), dtype=torch.float32, device=device)
        self.eye_NxN = torch.eye(N, N, device=device)
        self.ones_NxN = torch.ones_like(self.eye_NxN)
        self.eye_N = torch.eye(N, device=device)
        self.eye_P = torch.eye(P, device=device)
        self.mus = torch.tensor(self.mus, dtype=torch.float32, device=device)

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
        """Run the AdaCGP model

        Args:
            X (torch.tensor): input data (T, N, P)
            y (torch.tensor): target data (T, N)
            weight_matrix (torch.tensor, optional): true weight matrix, if known. Defaults to None.
            filter_coefficients (torch.tensor, optional): true filter coefficients if known. Defaults to None.
            graph_filter_matrix (torch.tensor, optional): true graph filter matrix if known. Defaults to None.
        """

        results = {
            'pred_error': [], 'pred_error_from_h': [], 'filter_error': [],
            'w_error': [], 'coeff_errors': [], 'first_alg_converged_status': [],
            'second_alg_converged_status': [], 'matrices': [], 'p_miss': [], 'p_false_alarm': [], 'psi_losses': [],
            'percentage_correct_elements': [], 'num_non_zero_elements': [],
            'prob_miss': [], 'prob_false_alarm': [], 'pred_error_recursive_moving_average': [1],
            'pred_error_recursive_moving_average_h': [1], 'precision': [], 'recall': [], 'f1': []
        }

        patience_left = self._patience
        switch_algorithm = False
        first_alg_converged = False
        second_alg_converged = False
        lowest_error = 1e10
        psi_loss = 0.0
        process_length = X.shape[0]
        debiasing_W = torch.ones_like(self.W)  # assume not sparse

        with torch.no_grad():
            with tqdm(range(process_length)) as pbar:
                for t in pbar:
                    ##################################
                    ######### GET DATA AT T ##########
                    ##################################

                    xPt = X[t, :, :].flatten()[:, None]  # (NP, 1)
                    yt = y[t]  # (N, 1)

                    if self._alternate:
                        switch_algorithm = (t % 2 == 0)

                    if self._alternate:
                        ma_error = results[self._monitor_debiasing][-1]
                    elif not first_alg_converged:
                        ma_error = results['pred_error_recursive_moving_average'][-1]
                    else:
                        ma_error = results[self._monitor_debiasing][-1]

                    ##################################
                    ######### CHECK CONVERGENCE ######
                    ##################################
                    if not self._alternate:
                        if not second_alg_converged:
                            if lowest_error != 0:
                                relative_improvement = (lowest_error - ma_error) / lowest_error
                            else:
                                relative_improvement = float('inf') if ma_error < lowest_error else 0

                            if relative_improvement > self._min_delta_percent:
                                lowest_error = ma_error
                                patience_left = self._patience
                            else:
                                patience_left -= 1

                            if patience_left == 0:
                                if not first_alg_converged:
                                    first_alg_converged = True
                                    if not self._alternate:
                                        switch_algorithm = True
                                    patience_left = self._burn_in_debiasing
                                    lowest_error = 1e10
                                else:
                                    second_alg_converged = True
                                    patience_left = self._patience
                        else:
                            patience_left -= 1
                            if patience_left == 0:
                                break
                    else:
                        # alternating
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
                    results['psi_losses'].append(psi_loss.item())
                    
                    if not self._alternate:
                        if not switch_algorithm:
                            psi_stepsize = self.update_psi(xPt, yt, t, ma_error, pbar, psi_loss)
                            pbar.set_postfix({'MA y error': ma_error, 'Step': psi_stepsize, 'Converged': switch_algorithm})
                
                            if self._use_path_1:
                                _ = self.update_w_path_1(xPt)
                            else:
                                self.update_w_path_2()
                            debiasing_W = (self.W.clone() != 0).float()
                            Psi = self.Psi.clone()
                            W = self.W.clone()
                
                        else:
                            Psi, W, psi_stepsize = self.perform_debiasing(xPt, yt, t, debiasing_W, psi_loss)
                            pbar.set_postfix({'MA y error': ma_error, 'Step': psi_stepsize, 'Converged': switch_algorithm})
                    else:
                        _ = self.update_psi(xPt, yt, t, ma_error, pbar, psi_loss)
                        if self._use_path_1:
                            _ = self.update_w_path_1(xPt)
                        else:
                            self.update_w_path_2()
                        debiasing_W = (self.W.clone() != 0).float()
                        Psi, W, psi_stepsize = self.perform_debiasing(xPt, yt, t, debiasing_W, psi_loss)
                        pbar.set_postfix({'MA y error': ma_error, 'Step': psi_stepsize, 'Converged': switch_algorithm})
            
                    ############################################
                    ########### COMPUTE FILTER COEFS ###########
                    ############################################

                    Xpt = xPt.view(self._P, self.N)
                    Ys = []
                    for i in range(1, self._P + 1):
                        x_t_m_i = Xpt[i-1, :]
                        for j in range(i + 1):
                            Yij = torch.matmul(torch.matrix_power(W, j), x_t_m_i)
                            Ys.append(Yij)
                    Ys = torch.stack(Ys, dim=1)  # (N, M)
            
                    b = torch.sign(self.h) / (self._epsilon + self.h)
                    nu_t = self._nu * torch.norm(torch.matmul(Ys.T, Ys), p=float('inf'))
            
                    if self._instant_h:
                        h_e = yt - torch.matmul(Ys, self.h)
                        h_g = torch.matmul(Ys.T, h_e)
                    else:
                        self.C = self._lambda * self.C + torch.matmul(Ys.T, Ys)
                        self.u = self._lambda * self.u + torch.matmul(Ys.T, yt)
                        h_g = torch.matmul(self.C, self.h) - self.u

                    ######## ARMIJO STEPSIZE #######
                    alpha = wolfe_line_search(
                        objective_function_h_lms,
                        gradient_function_h_lms,
                        update_function_h_lms,
                        self.h.flatten(),
                        step_init=self._h_stepsize,
                        beta=0.5,
                        args=(Ys, yt, self._lambda, self.C, self.u, self._instant_h, nu_t, self._epsilon),
                        max_iter=10
                    )
                    h_stepsize = alpha
            
                    ######### UPDATE PARAM #########
                    dh = h_g + nu_t * b
                    self.h = self.h + h_stepsize * dh
                    d_hat_h = torch.matmul(Ys, self.h)

                    ###################################
                    ######### COMPUTE RESULTS #########
                    ###################################

                    # Compute squared error of signal forecast from graph filters
                    d_hat_psi = torch.matmul(Psi, xPt)
                    e = yt - d_hat_psi
                    norm_error = torch.norm(e)**2 / torch.norm(yt)**2
                    results['pred_error'].append(norm_error.item())
                    ma_error = self._ma_alpha * norm_error + (1 - self._ma_alpha) * results['pred_error_recursive_moving_average'][-1]
                    results['pred_error_recursive_moving_average'].append(ma_error.item())
            
                    # Compute the prediction error from h
                    # if not switch_algorithm:
                    #     results['pred_error_from_h'].append(0)
                    #     results['pred_error_recursive_moving_average_h'].append(0)
                    # else:
                    h_e = yt - d_hat_h
                    norm_h_error = torch.norm(h_e)**2 / torch.norm(yt)**2
                    results['pred_error_from_h'].append(norm_h_error.item())
                    ma_error_h = self._ma_alpha * norm_h_error + (1 - self._ma_alpha) * results['pred_error_recursive_moving_average_h'][-1]
                    results['pred_error_recursive_moving_average_h'].append(ma_error_h.item())
        
                    # Compute squared error of graph filter estimation
                    if graph_filter_matrix is not None:
                        m_error = graph_filter_matrix - Psi
                        norm_m_error = torch.norm(m_error)**2 / torch.norm(graph_filter_matrix)**2
                        results['filter_error'].append(norm_m_error.item())
            
                    # Compute squared error of W estimation
                    if weight_matrix is not None:
                        weight_matrix_error = weight_matrix - W
                        norm_w_error = torch.norm(weight_matrix_error)**2 / torch.norm(weight_matrix)**2
                        results['w_error'].append(norm_w_error.item())
                        results['num_non_zero_elements'].append((W != 0).sum().item())

                        # compute the percentage of elements correctly identified in W
                        # recall
                        total = (weight_matrix != 0).sum()
                        frac = ((W != 0) * (weight_matrix != 0)).sum() / total
                        results['percentage_correct_elements'].append(frac.item())

                        true_positives = ((W != 0) * (weight_matrix != 0)).sum().item()
                        false_positives = ((W != 0) * (weight_matrix == 0)).sum().item()
                        false_negatives = ((W == 0) * (weight_matrix != 0)).sum().item()

                        precision = 0 if (true_positives + false_positives) == 0 else true_positives / (true_positives + false_positives)
                        recall = 0 if (true_positives + false_negatives) == 0 else true_positives / (true_positives + false_negatives)
                        f1 = 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)
                        results['precision'].append(precision)
                        results['recall'].append(recall)
                        results['f1'].append(f1)

                        # save results for p_miss: probability of missing a non-zero element in W
                        results['p_miss'].append(((W == 0) * (weight_matrix != 0)).sum().item() / (weight_matrix != 0).sum().item())
                        results['p_false_alarm'].append(((W != 0) * (weight_matrix == 0)).sum().item() / (weight_matrix == 0).sum().item())
                
                    # Compute the error for filter coefficient estimation
                    if filter_coefficients is not None:
                        coeff_error = filter_coefficients - self.h.flatten()
                        norm_coeff_error = torch.norm(coeff_error)**2 / torch.norm(filter_coefficients)**2
                        results['coeff_errors'].append(norm_coeff_error.item())
                
                    # Store the convergence status
                    results['first_alg_converged_status'].append(first_alg_converged)
                    results['second_alg_converged_status'].append(second_alg_converged)
        results['matrices'].append(W.cpu().numpy())
        return results
    
    def perform_debiasing(self, xPt, yt, t, debiasing_W, psi_loss):
        # Update R0 and P0
        self.R0 = self._lambda * self.R0 + torch.matmul(xPt, xPt.T)
        self.P0 = self._lambda * self.P0 + torch.matmul(yt, xPt.T)

        # Compute mus
        mu_scales = []
        Q_unpacked = get_each_graph_filter(self.Q, self.N, self._P)
        P0_unpacked = get_each_graph_filter(self.P0, self.N, self._P)
        for p in range(self._P):
            Qp = Q_unpacked[:, p, :]
            P0p = P0_unpacked[:, p, :]
            infty_norm = torch.norm(P0p - self._gamma * Qp, p=float('inf'))
            mu_scales.append(infty_norm)

        mus_pt = self.mus * torch.stack(mu_scales)

        # Compute G
        G = torch.matmul(self.Psi, self.R0) - self.P0

        # Apply mask
        masks = [torch.matrix_power(debiasing_W, i+1) for i in range(self._P)]
        mask = pack_graph_filters(masks, self.N, self._P)
        G[mask == 0] = 0

        # Compute stepsize
        eigs = torch.lobpcg(self.R0, largest=True)
        psi_stepsize = 2 / (eigs[0].item())
        A = self.eye_P * psi_stepsize
        for p in range(self._P):
            A[p, p] /= (torch.linalg.norm(xPt[p*self.N:(p+1)*self.N], ord=2)**2 + self._epsilon)

        # Armijo line search
        if self._use_armijo and (t > self._warm_up_steps):
            Psi_unpacked = get_each_graph_filter(self.Psi, self.N, self._P)
            for p in range(self._P):
                alpha = wolfe_line_search(
                    objective_function_psi_debias,
                    gradient_function_psi_debias,
                    update_function_psi_debias,
                    Psi_unpacked[:, p, :].flatten(),
                    step_init=A[p, p],
                    beta=0.5,
                    args=(self.Psi.clone(), mus_pt, self.R0, self.P0, 0, p, False, self.N, self._P, self._lambda, xPt, yt, psi_loss, self.W),
                    max_iter=10
                )
                A[p, p] = alpha

        # Update parameters
        Psi = self.Psi - torch.matmul(G, torch.kron(A, self.eye_N))
        Psi_unpacked = get_each_graph_filter(self.Psi, self.N, self._P)
        W = Psi_unpacked[:, 0, :] * debiasing_W
        return Psi, W, psi_stepsize

    def update_w_path_2(self):
        Psi_unpacked = get_each_graph_filter(self.Psi, self.N, self._P)
        self.W = Psi_unpacked[:, 0, :]

    def update_w_path_1(self, xPt):
        # Compute S
        Psi_unpacked = get_each_graph_filter(self.Psi, self.N, self._P)
        S = second_comm_term_mnlms(Psi_unpacked, self.W, self._P)

        # Compute gradient
        Psi_1 = Psi_unpacked[:, 0, :]
        V = self.W - (Psi_1 - self._gamma * S)
        M_1 = self.ones_NxN * self.mus_pt[0] / (torch.linalg.norm(xPt[:self.N], ord=2)**2 + self._epsilon)

        # Armijo stepsize
        alpha = wolfe_line_search(
            objective_function_wstep2_lms,
            gradient_function_wstep2_lms,
            update_function_wstep2_lms,
            self.W.flatten(),
            step_init=self._w_stepsize,
            beta=0.5,
            args=(self.Psi.clone(), self.mus_pt, self._gamma, self.N, self._P),
            max_iter=10
        )
        w_stepsize = alpha

        # Update param
        self.W_pos = self.W_pos - w_stepsize * (M_1 + V)
        self.W_neg = self.W_neg - w_stepsize * (M_1 - V)
        self.W_pos[self.W_pos < 0] = 0
        self.W_neg[self.W_neg < 0] = 0
        self.W = self.W_pos - self.W_neg
        return w_stepsize

    def update_psi(self, xPt, yt, t, ma_error, pbar, psi_loss):
        # Update R0 and P0
        self.R0 = self._lambda * self.R0 + torch.matmul(xPt, xPt.T)
        self.P0 = self._lambda * self.P0 + torch.matmul(yt, xPt.T)

        # Compute mus
        mu_scales = []
        Q_unpacked = get_each_graph_filter(self.Q, self.N, self._P)
        P0_unpacked = get_each_graph_filter(self.P0, self.N, self._P)
        for p in range(self._P):
            Qp = Q_unpacked[:, p, :]
            P0p = P0_unpacked[:, p, :]
            infty_norm = torch.norm(P0p - self._gamma * Qp, p=float('inf'))
            mu_scales.append(infty_norm)

        self.mus_pt = self.mus * torch.stack(mu_scales)
        M = torch.vstack([self.ones_NxN * self.mus_pt[p] for p in range(self._P)]).T  # (N, N*P)

        # Compute G
        include_comm_term = not self._use_path_1
        if include_comm_term:
            Qs = []
            Psi_unpacked = get_each_graph_filter(self.Psi, self.N, self._P)
            for p in range(self._P):
                Qp = comm_term_mnlms(Psi_unpacked, p, self._P)
                Qs.append(Qp)
            self.Q = pack_graph_filters(Qs, self.N, self._P)
            G = torch.matmul(self.Psi, self.R0) - (self.P0 - self._gamma * self.Q)
        else:
            G = torch.matmul(self.Psi, self.R0) - self.P0

        # Compute stepsize
        eigs = torch.lobpcg(self.R0, largest=True)
        psi_stepsize = 2 / (eigs[0].item())
        A = self.eye_P * psi_stepsize
        for p in range(self._P):
            A[p, p] /= (torch.linalg.norm(xPt[p*self.N:(p+1)*self.N], ord=2)**2 + self._epsilon)

        # Line search
        if self._use_armijo and (t > self._warm_up_steps):
            Psi_unpacked = get_each_graph_filter(self.Psi, self.N, self._P)
            direction_unpacked = get_each_graph_filter(G, self.N, self._P)
            for p in range(self._P):
                alpha = wolfe_line_search(
                    objective_function_psi_mlms,
                    gradient_function_psi_mlms,
                    update_function_psi_mlms,
                    Psi_unpacked[:, p, :].flatten(),
                    step_init=A[p, p],
                    beta=0.5,
                    args=(self.Psi.clone(), self.mus_pt, self.R0, self.P0, self._gamma, p, include_comm_term, self.N, self._P, self._lambda, xPt, yt, psi_loss),
                    max_iter=10
                )
                A[p, p] = alpha

        # Update Psi
        dPsi_pos = - (M + G) @ torch.kron(A, self.eye_N)
        dPsi_neg = - (M - G) @ torch.kron(A, self.eye_N)
        self.Psi_pos = self.Psi_pos + dPsi_pos
        self.Psi_neg = self.Psi_neg + dPsi_neg

        # Projection onto non-negative space
        self.Psi_pos[self.Psi_pos < 0] = 0
        self.Psi_neg[self.Psi_neg < 0] = 0
        self.Psi = self.Psi_pos - self.Psi_neg
        return psi_stepsize

def comm(A, B):
    return torch.matmul(A, B) - torch.matmul(B, A)
 
def comm_term_mnlms(W, p, P):
    comm_terms = []
    for k in range(P):
        if k == p:
            continue
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
