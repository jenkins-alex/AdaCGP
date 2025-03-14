import numpy as np
from tqdm import tqdm
from sklearn.covariance import empirical_covariance, graphical_lasso
from scipy import optimize
import scipy.linalg as la

class GraphSmooth:
    """
    Implementation of graph Laplacian learning using smoothness prior:

    Dong, X., Thanou, D., Frossard, P., & Vandergheynst, P. (2014).
    Learning graphs from signal observations under smoothness prior.
    arXiv preprint arXiv:1406.7842.
    """
    def __init__(self, N, hyperparams, device):
        self.N = N
        self.set_hyperparameters(hyperparams)

    def set_hyperparameters(self, hyperparams):
        for param, value in hyperparams.items():
            setattr(self, f"_{param}", value)
    
    def predict_topology(self, data, t):
        N = data.shape[1]
        X_window = data[t-self._P_cov:t, :].T
        L_opt, _, obj_values = self._alternating_optimization(X_window)
        W = np.diag(np.diag(L_opt)) - L_opt  # L -> W assuming no self loops
        latest_obj_fn = obj_values[-1]
        return W, latest_obj_fn
        
    def run(self, y, weight_matrix=None, **kwargs):
        # This function computes an estimate via TISO

        results = {
            'pred_error': [], 'w_error': [], 'matrices': [],
            'percentage_correct_elements': [], 'num_non_zero_elements': [],
            'p_miss': [], 'p_false_alarm': [], 'pred_error_recursive_moving_average': [1]
        }

        # init params
        y = np.array(y)
        weight_matrix = np.array(weight_matrix) if weight_matrix is not None else None
        m_y = y[:, :, 0]
        T, N = m_y.shape

        with tqdm(range(self._P_cov, T)) as pbar:
            for t in pbar:  # in paper, t=P,...
                # receive data y[t]
                ma_error = results['pred_error_recursive_moving_average'][-1]
                if self._patience is not None:
                    if (t-self._P_cov) > self._patience:
                        break

                ##################################
                ######### COMPUTE W ##############
                ##################################
                W, e = self.predict_topology(m_y, t)

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

    def _optimize_L(self, Y, alpha, beta, n, max_iters=100, tol=1e-6):
        """
        Solve the optimization problem for L given Y using CVXPY:
        min_L α tr(Y^T LY) + β||L||_F^2
        s.t. tr(L) = n
            L_ij = L_ji ≤ 0, i ≠ j
            L · 1 = 0
        """
        try:
            # Create duplication matrix
            M_dup = self._create_duplication_matrix(n)
            M_dup_T = M_dup.T
            
            # Compute YY^T
            YYT = Y @ Y.T
            vec_YYT = YYT.flatten()
            
            # the objective function: α vec(YY^T)^T M_dup vech(L) + β vech(L)^T M_dup^T M_dup vech(L)
            def objective(vech_L):
                term1 = alpha * vec_YYT.T @ M_dup @ vech_L
                term2 = beta * vech_L.T @ M_dup_T @ M_dup @ vech_L
                return term1 + term2
            
            # gradient of the objective function
            def gradient(vech_L):
                return alpha * M_dup_T @ vec_YYT + 2 * beta * M_dup_T @ M_dup @ vech_L
            
            # valid initial Laplacian matrix
            L_init = self._create_laplacian(n)
            vech_L_init = self._matrix_to_vech(L_init)
            vech_size = len(vech_L_init)
            
            # get indices for diagonal and off-diagonal elements
            diag_indices = self._get_diagonal_indices(n)
            offdiag_indices = self._get_off_diagonal_indices(n)

            # define bounds for matrix elements
            # zero or positive for diagonal elements, negative or zero for off-diagonal
            bounds = [(None, None)] * vech_size
            for idx in range(vech_size):
                if idx in offdiag_indices:
                    bounds[idx] = (None, 0)  # off-diagonal elements must be <= 0
                else:
                    bounds[idx] = (0, None)  # diagonal elements must be >= 0

            # Equality constraints using LinearConstraint
            # 1. trace constraint: sum of diagonal elements = n
            A_trace = np.zeros(vech_size)
            for idx in diag_indices:
                A_trace[idx] = 1

            # 2. row sum constraints: sum of each row = 0
            A_rows = np.zeros((n, vech_size))

            # Create a mapping from matrix positions to vech indices
            pos_to_vech = {}
            idx = 0
            for i in range(n):
                for j in range(i, n):
                    pos_to_vech[(i, j)] = idx
                    pos_to_vech[(j, i)] = idx  # Symmetric
                    idx += 1

            # fill in row sum constraint matrices
            for row in range(n):
                # for each row, we need to add all elements in that row
                for col in range(n):
                    vech_idx = pos_to_vech[(row, col)]
                    A_rows[row, vech_idx] = 1

            # Combine equality constraints
            A_eq = np.vstack([A_trace, A_rows])
            b_eq = np.zeros(n+1)
            b_eq[0] = n  # Trace = n
            
            # using trust-constr (interior point) method for constrained optimization
            result = optimize.minimize(
                objective,
                vech_L_init,
                method='trust-constr',
                jac=gradient,
                bounds=bounds,
                constraints=[
                    optimize.LinearConstraint(A_eq, b_eq, b_eq)
                ],
                options={'maxiter': max_iters, 'gtol': tol, 'verbose': 0}
            )
            
            if not result.success:
                print(f"Warning: L optimization not fully successful. Message: {result.message}")
            
            # Convert vech(L) to matrix form
            vech_L_opt = result.x
            L_opt = self._vech_to_matrix(vech_L_opt, n)
            
            return L_opt, result.success, result.message
                
        except Exception as e:
            print(f"Error in optimize_L: {str(e)}")
            # Return a fallback valid Laplacian matrix
            L_fallback = self._create_laplacian(n)
            return L_fallback, False, str(e)

    def _optimize_Y(self, X, L, alpha, n):
        """
        Solve the optimization problem for Y given L:
        min_Y ||X - Y||_F^2 + α tr(Y^T LY)
        
        The closed-form solution is:
        Y = (I_n + αL)^(-1)X
        """
        try:
            # The matrix I_n + αL is symmetric and should be positive-definite
            I_plus_alphaL = np.eye(n) + alpha * L
            
            # Ensure the matrix is numerically symmetric to avoid Cholesky issues
            I_plus_alphaL = (I_plus_alphaL + I_plus_alphaL.T) / 2
            
            # Add a small regularization to ensure positive definiteness
            min_eig = np.min(np.linalg.eigvalsh(I_plus_alphaL))
            if min_eig < 1e-10:
                I_plus_alphaL += (1e-10 - min_eig + 1e-6) * np.eye(n)
            
            # Try Cholesky factorization
            try:
                chol_factor = la.cholesky(I_plus_alphaL, lower=True)
                Y = la.cho_solve((chol_factor, True), X)
            except np.linalg.LinAlgError:
                # Fallback to direct solve if Cholesky fails
                print("Warning: Cholesky factorization failed. Using direct solve.")
                Y = np.linalg.solve(I_plus_alphaL, X)
            return Y
        
        except Exception as e:
            print(f"Error in optimize_Y: {str(e)}")
            # Return X as fallback
            return X.copy()

    def _alternating_optimization(self, X):
        """
        Solve the optimization problem:
        min_{L,Y} ||X - Y||_F^2 + α tr(Y^T LY) + β||L||_F^2
        using alternating optimization
        """
        n, d = X.shape
        
        # init Y with X
        Y = X.copy()
        
        # init L as a valid Laplacian matrix
        L = self._create_laplacian(n)
        
        # track the objective values
        obj_values = []
        
        for iter in range(self._max_iters):
            Y_prev = Y.copy()
            L_prev = L.copy()
            
            # step 1: Optimize L given Y
            L, success_L, message_L = self._optimize_L(Y, self._alpha, self._beta, n, max_iters=self._max_iters, tol=self._tol)
            if not success_L:
                print(f"Warning at iteration {iter+1}: {message_L}")

            # step 2: Optimize Y given L
            Y = self._optimize_Y(X, L, self._alpha, n)
            
            # compute the objective value
            obj_val = np.sum((X - Y)**2) + self._alpha * np.trace(Y.T @ L @ Y) + self._beta * np.sum(L**2)
            obj_values.append(obj_val)
            
            # Print progress
            if iter % 5 == 0 or iter == self._max_iters - 1:
                pass
                # print(f"Iteration {iter+1}/{max_iters}, Objective: {obj_val:.6f}")
            
            # Check convergence - using relative change in both Y and L
            Y_diff = np.linalg.norm(Y - Y_prev) / (np.linalg.norm(Y_prev) + 1e-10)
            L_diff = np.linalg.norm(L - L_prev) / (np.linalg.norm(L_prev) + 1e-10)
            if Y_diff < self._tol and L_diff < self._tol:
                # print(f"Converged after {iter+1} iterations")
                break
    
        if iter == self._max_iters - 1:
            pass
            # print(f"Maximum iterations reached ({max_iters})")
        return L, Y, obj_values

    def _create_duplication_matrix(self, n):
        """
        Create the duplication matrix M_dup that converts vech(L) to vec(L)
        """
        # size of vech(L)
        vech_size = n * (n + 1) // 2
        
        # nitialize the duplication matrix
        M_dup = np.zeros((n*n, vech_size))
        
        # Fill the duplication matrix
        vech_idx = 0
        for i in range(n):
            for j in range(i, n):  # Lower triangular part including diagonal
                vec_idx_1 = i * n + j
                M_dup[vec_idx_1, vech_idx] = 1
                
                # If not on diagonal, also set the symmetric entry
                if i != j:
                    vec_idx_2 = j * n + i
                    M_dup[vec_idx_2, vech_idx] = 1
                    
                vech_idx += 1

        return M_dup

    def _matrix_to_vech(self, mat):
        """Convert a symmetric matrix to its half-vectorization form"""
        n = mat.shape[0]
        vech_size = n * (n + 1) // 2
        vech = np.zeros(vech_size)
        
        idx = 0
        for i in range(n):
            for j in range(i, n):
                vech[idx] = mat[i, j]
                idx += 1
        
        return vech

    def _vech_to_matrix(self, vech, n):
        """Convert a half-vectorization back to a symmetric matrix"""
        mat = np.zeros((n, n))
        
        idx = 0
        for i in range(n):
            for j in range(i, n):
                mat[i, j] = vech[idx]
                if i != j:
                    mat[j, i] = vech[idx]  # Ensure symmetry
                idx += 1
        
        return mat

    def _create_laplacian(self, n):
        """
        Create a valid Laplacian matrix for a fully connected graph.
        This ensures: 
        - diagonal entries are positive
        - off-diagonal entries are non-positive
        - row sums are zero
        - trace is n
        """
        # Start with off-diagonal entries as -1/(n-1)
        # This ensures row sums will be zero when diagonal is 1
        L = np.ones((n, n)) * (-1.0 / (n-1))
        
        # Set diagonal to 1
        np.fill_diagonal(L, 1.0)
        
        return L

    def _get_diagonal_indices(self, n):
        """Get the indices of diagonal elements in vech representation"""
        diag_indices = []
        idx = 0
        for i in range(n):
            diag_indices.append(idx)
            idx += (n - i)
        return diag_indices

    def _get_off_diagonal_indices(self, n):
        """Get the indices of off-diagonal elements in vech representation"""
        offdiag_indices = []
        idx = 0
        for i in range(n):
            for j in range(i, n):
                if i != j:  # Off-diagonal element
                    offdiag_indices.append(idx)
                idx += 1
        return offdiag_indices