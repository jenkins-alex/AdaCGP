import gc
import time
import tracemalloc
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.api import VAR
import numpy as np
from scipy.special import psi
from sklearn.neighbors import KDTree
from math import sqrt


class PMIME:
    """
    Python wrapped for PMIME, Kugiumtzis, D. (2013).
    Direct-coupling information measure from nonuniform embedding.
    Physical Review E—Statistical, Nonlinear, and Soft Matter Physics, 87(6), 062918.
    """
    def __init__(self, N, hyperparams, device, **kwargs):
        self.N = N
        self.set_hyperparameters(hyperparams)

    def set_hyperparameters(self, hyperparams):
        for param, value in hyperparams.items():
            setattr(self, f"_{param}", value)
        if not hasattr(self, '_train_steps_list'):
            self._train_steps_list = None
        if not hasattr(self, '_record_complexity'):
            self._record_complexity = False

    def predict_topology(self, data, t):
        N = data.shape[1]
        X = data[:t, :]
        W, ecC = _PMIME(X, Lmax=self._Lmax, T=self._T, nnei=self._nnei, A=self._A, showtxt=self._showtxt)
        mean_cmi = self.calculate_average_cmi(ecC)
        return W, mean_cmi

    def calculate_average_cmi(self, ecC):
        """
        Calculate the average CMI ratio across all variables.
        
        Parameters:
        ecC (list): List of embedding cycle arrays for each variable
        
        Returns:
        float: Average CMI ratio across all variables
        """
        all_valid = []
        
        # loop over embedding cycles of variables
        for var_cycles in ecC:
            if var_cycles is not None and len(var_cycles) > 0:
                # extract cmi (column 2) and filter out NaNs
                ratios = [row[2] for row in var_cycles if len(row) > 4 and not np.isnan(row[2])]
                all_valid.extend(ratios)
        
        # calculate mean if we have any valid values
        if all_valid:
            return np.abs(np.mean(all_valid))
        else:
            return np.nan  # no valid values found

    def run(self, y, weight_matrix=None, **kwargs):
        # This function computes an estimate via TISO

        results = {
            'pred_error': [], 'w_error': [], 'matrices': [],
            'percentage_correct_elements': [], 'num_non_zero_elements': [],
            'p_miss': [], 'p_false_alarm': [], 'pred_error_recursive_moving_average': []
        }
        if self._record_complexity:
            results['iteration_time'] = []
            results['iteration_memory'] = []

        # init params
        lowest_error = 1e10
        patience_left = self._patience
        y = np.array(y)
        weight_matrix = np.array(weight_matrix) if weight_matrix is not None else None
        m_y = y[:, :, 0]
        T, N = m_y.shape

        # training loop
        iter_range = range(self._min_samples+1, T) if self._train_steps_list is None else self._train_steps_list
        with tqdm(iter_range) as pbar:
            for t in pbar:

                # start measuring iteration memory and time complexity
                if self._record_complexity:
                    gc.collect()
                    tracemalloc.start()
                    start_time = time.process_time()

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

                # compute squared error of the objective
                norm_error = e
                results['pred_error'].append(norm_error)
                ma_error = self._ma_alpha * norm_error + (1 - self._ma_alpha) * results['pred_error_recursive_moving_average'][-1]
                results['pred_error_recursive_moving_average'].append(ma_error)
        
                # compute squared error of W estimation
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


def _PMIME(allM, Lmax=None, T=None, nnei=None, A=None, showtxt=None):
    """
    Code adapted from https://github.com/dkugiu/Matlab/tree/master/PMIME

    function [RM,ecC] = PMIME(allM,Lmax,T,nnei,A,showtxt)
    PMIME (Partial Mutual Information on Mixed Embedding)
    computes the measure R_{X->Y|Z} for all combinations of X and Y time
    series from the multivariate time series given in matrix 'allM', of size
    N x K, where Z contains the rest K-2 time series. 
    The components of X,Y, and Z, are found from a mixed embedding aiming at
    explaining Y. The mixed embedding is formed by using the progressive 
    embedding algorithm based on conditional mutual information (CMI). 
    CMI is estimated by the method of nearest neighbors (Kraskov's method). 
    The function is the same as PMIMEsig.m but defines the stopping criterion
    differently, using a fixed rather than adjusted threshold. Specifically,
    the algorithm terminates if the contribution of the selected lagged
    variable in explaining the future response state is small enough, as
    compared to a threshold 'A'. Concretely, the algorithm terminates if 
           I(x^F; w| wemb) / I(x^F; w,wemb) <= A
    where I(x^F; w| wemb) is the CMI of the selected lagged variable w and 
    the future response state x^F given the current mixed embedding vector, 
    and I(x^F; w,wemb) is the MI between x^F and the augmented mixed
    embedding vector [wemb w].
    We experienced that in rare cases the termination condition is not 
    satisfied and the algorithm does not terminate. Therefore we included a 
    second condition for termination of the algorithm when the ratio 
    I(x^F; w| wemb) / I(x^F; w,wemb) increases in the last two embedding
    cycles. 
    The derived R measure indicates the information flow of time series X to
    time series Y conditioned on the rest time series in Z. The measure
    values are stored in a K x K matrix 'RM' and given to the output, where
    the value at position (i,j) indicates the effect from i to j (row to
    col), and the (i,i) components are zero.
    INPUTS
    - allM : the N x K matrix of the K time series of length N.
    - Lmax : the maximum delay to search for X and Y components for the mixed 
             embedding vector [default is 5].
    - T    : T steps ahead that the mixed embedding vector has to explain.
             Note that if T>1 the future vector is of length T and contains
             the samples at times t+1,..,t+T [dafault is 1]. 
    - nnei : number of nearest neighbors for density estimation [default is 5]
    - A    : the threshold for the ratio of CMI over MI of the lagged variables
             for the termination criterion.
    - showtxt : if 0 or negative do not print out anything, 
                if 1 print out the response variable index at each run, 
                if 2 or larger print also info for each embedding cycle [default is 1].
    OUTPUTS
    - RM   : A K x K matrix containing the R values computed by PMIME using
             the surrogates for setting the stopping criterion. 
    - ecC  : cell array of K components, where each component is a matrix of 
             size E x 5, and E is the number of embedding cycles. For each 
             embedding cycle the following 5 results are stored:
             1. variable index, 2. lag index, 3. CMI of the selected lagged
             variable w and the future response state x^F given the current 
             mixed embedding vector, I(x^F; w|wemb). 4. MI between x^F and 
             the augmented mixed embedding vector [wemb w], I(x^F; w,wemb).
             5. The ration of 3. and 4.: I(x^F; w|wemb)/I(x^F; w,wemb)  
    
        Copyright (C) 2015 Dimitris Kugiumtzis
     
        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.
     
        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.
     
        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.
     
    =========================================================================
    Reference : D. Kugiumtzis, "Direct coupling information measure from 
                non-uniform embedding", Physical Review E, Vol 87, 062918, 
                2013
                I. Vlachos, D. Kugiumtzis, "Non-uniform state space 
                reconstruction and coupling detection", Physical Review E, 
                Vol 82, 016207, 2010
    Link      : http://users.auth.gr/dkugiu/
    =========================================================================
    """
    # A safeguard, to make sure that the algorithm does not make  
    # more than "maxcomps" embedding cycles. 
    maxcomps = 20
    # Setting default parameter values as in MATLAB code
    if showtxt is None and A is not None and nnei is not None and T is not None and Lmax is not None:
        # nargin==5 case: PMIME(allM,Lmax,T,nnei,A)
        showtxt = 1
    elif showtxt is None and A is None and nnei is not None and T is not None and Lmax is not None:
        # nargin == 4
        showtxt = 1
        A = 0.03
    elif showtxt is None and A is None and nnei is None and T is not None and Lmax is not None:
        # nargin == 3
        showtxt = 1
        A = 0.03
        nnei = 5
    elif showtxt is None and A is None and nnei is None and T is None and Lmax is not None:
        # nargin == 2
        showtxt = 1
        A = 0.03
        nnei = 5
        T = 1
    elif showtxt is None and A is None and nnei is None and T is None and Lmax is None:
        # nargin == 1
        showtxt = 1
        A = 0.03
        nnei = 5
        T = 1
        Lmax = 5

    if A is None or (hasattr(A, '__len__') and len(A)==0):
        A = 0.03
    if nnei is None or (hasattr(nnei, '__len__') and len(nnei)==0):
        nnei = 5
    if T is None or (hasattr(T, '__len__') and len(T)==0):
        T = 1
    if Lmax is None or (hasattr(Lmax, '__len__') and len(Lmax)==0):
        Lmax = 5

    (N, K) = allM.shape
    # wV: maximum lag for each variable (K x 1 vector)
    wV = Lmax * np.ones((K, 1), dtype=int)
    ## Standardization of the input matrix columnwise in [0,1].
    minallV = np.min(allM, axis=0)
    range_all = np.ptp(allM, axis=0)  # ptp = max - min
    # Use broadcasting to scale each column.
    allM = (allM - minallV) * (1.0 / range_all)
    ## Build up the lag matrix from all variables
    total_lags = int(np.sum(wV))
    alllagM = np.full((N, total_lags), np.nan)  # lag matrix of all variables
    indlagM = np.full((K, 2), np.nan)             # Start and end of columns of each variable in lag matrix
    count = 0
    for iK in range(K):
        # In MATLAB indices are 1-indexed; here we use 0-indexing.
        # Start index for variable iK in alllagM:
        start_idx = count
        # End index is start_idx + Lmax - 1 since we want Lmax columns per variable.
        end_idx = count + int(wV[iK]) - 1
        indlagM[iK, :] = [start_idx, end_idx]
        # lag=0
        alllagM[:, start_idx] = allM[:, iK]
        # lag=1,...,Lmax-1
        for ilag in range(1, int(wV[iK])):
            # MATLAB: alllagM((ilag+1):end, start_idx+ilag)=allM(1:(end-ilag), iK);
            alllagM[ilag: , start_idx+ilag] = allM[:N-ilag, iK]
        count = count + int(wV[iK])
    # Select rows from Lmax to (end-T) as in MATLAB: alllagM = alllagM(Lmax:end-T,:)
    alllagM = alllagM[int(Lmax)-1:N - int(T), :]
    (N1, alllags) = alllagM.shape
    ## Find mixed embedding and R measure for purpose: from (X,Y,Z) -> Y 
    RM = np.zeros((K, K))
    ecC = [None for _ in range(K)]
    psinnei_val = psi(nnei)  # Computed once here, to be called several times
    psiN1_val = psi(N1)       # Computed once here, to be called several times
    for iK in range(K):
        if showtxt == 1:
            # Print on the same line without newline (end='')        
            print(f"{iK+1}..", end='')
        elif showtxt >= 2:
            print(f"Response variable index={iK+1}.. ")
            print("EmbeddingCycle  Variable  Lag  I(x^F;w|wemb)  I(x^F;w,wemb)  I(x^F;w|wemb)/I(x^F;w,wemb)")
        Xtemp = np.full((N, int(T)), np.nan)
        for iT in range(int(T)):
            # MATLAB: Xtemp(1:(end-iT), iT) = allM((1+iT):end, iK);
            Xtemp[0:N - (iT+1), iT] = allM[iT+1: , iK]
        # xFM: future vector of response: Xtemp(Lmax:end-T, :)
        xFM = Xtemp[int(Lmax)-1:N - int(T), :]
        # First embedding cycle: max I(y^T, w), over all candidates w
        miV = np.full((alllags,), np.nan)
        for i1 in range(alllags):
            # Compute the mutual information of future response and each one of
            # the candidate lags using the nearest neighbor estimate
            xnowM = np.hstack((xFM, alllagM[:, i1].reshape(-1, 1)))
            # annMaxquery expects data with rows as samples.
            _, distsM = annMaxquery(xnowM, xnowM, int(nnei)+1)
            # maxdistV: the maximum distance (last column) for each point
            maxdistV = distsM[:, -1]
            nyFV = nneighforgivenr(xFM, maxdistV - 1e-10)
            nwcandV = nneighforgivenr(alllagM[:, i1].reshape(-1, 1), maxdistV - 1e-10)
            psibothM = psi(np.column_stack((nyFV, nwcandV)))
            miV[i1] = psinnei_val + psiN1_val - np.mean(np.sum(psibothM, axis=1))
        # Select index with maximum miV
        iemb_index = int(np.nanargmax(miV))
        iembV = [iemb_index]
        xembM = alllagM[:, iemb_index].reshape(-1, 1)
        # add the selected lag variable in the first embedding cycle and show it
        # Compute variable index and lag index (converting from 0-indexed to MATLAB style)
        varind = int(np.ceil((iemb_index+1) / float(Lmax)))
        lagind = (iemb_index+1) % int(Lmax)
        if lagind == 0:
            lagind = int(Lmax)
        ecC[iK] = np.array([[varind, lagind, miV[iemb_index], np.nan, np.nan]])
        if showtxt >= 2:
            row = ecC[iK][-1, :]
            print(f"{ecC[iK].shape[0]} \t {int(row[0])} \t {int(row[1])} \t {row[2]:2.5f} \t {row[3]} \t {row[4]}")
        # End of first embedding cycle, the first lagged variable is found
        terminator = False
        maxcomps = min(alllagM.shape[1], maxcomps)  # To avoid large embedding
        # Run iteratively, for each embedding cycle select w from max I(y^F; w | wemb) 
        while (not terminator and xembM.shape[1] < maxcomps):
            # activeV: indices of candidates not already selected
            activeV = np.setdiff1d(np.arange(alllags), np.array(iembV))
            cmiV = np.full((alllags,), np.nan)  # I(y^F; w | wemb)
            miwV = np.full((alllags,), np.nan)  # I(y^F; w, wemb)
            for candidate in activeV:
                xallnowM = np.hstack((xFM, alllagM[:, candidate].reshape(-1, 1), xembM))
                _, distsM = annMaxquery(xallnowM, xallnowM, int(nnei)+1)
                maxdistV = distsM[:, -1]
                nwV = nneighforgivenr(xembM, maxdistV - 1e-10)
                nwcandV = nneighforgivenr(np.hstack((alllagM[:, candidate].reshape(-1, 1), xembM)), maxdistV - 1e-10)
                nyFwV = nneighforgivenr(np.hstack((xFM, xembM)), maxdistV - 1e-10)
                psinowM = np.full((N1, 3), np.nan)
                psinowM[:, 0] = psi(nyFwV)
                psinowM[:, 1] = psi(nwcandV)
                psinowM[:, 2] = -psi(nwV)
                cmiV[candidate] = psinnei_val - np.mean(np.sum(psinowM, axis=1))
                nyFV = nneighforgivenr(xFM, maxdistV - 1e-10)
                psinow_temp = np.column_stack((psi(nyFV), psinowM[:, 1]))
                miwV[candidate] = psinnei_val + psiN1_val - np.mean(np.sum(psinow_temp, axis=1))
            # Select candidate with maximum conditional mutual information
            ind = int(np.nanargmax(cmiV))
            xVnext = alllagM[:, ind].reshape(-1, 1)
            varind = int(np.ceil((ind+1) / float(Lmax)))
            lagind = (ind+1) % int(Lmax)
            if lagind == 0:
                lagind = int(Lmax)
            cmi_ratio = (cmiV[ind] / miwV[ind]) if not np.isnan(cmiV[ind]) and not np.isnan(miwV[ind]) and not miwV[ind] == 0 else np.nan
            new_row = np.array([varind, lagind, cmiV[ind], miwV[ind], cmi_ratio])
            ecC[iK] = np.vstack((ecC[iK], new_row))
            if len(iembV) == 1:
                if showtxt >= 2:
                    row = ecC[iK][-1, :]
                    print(f"{ecC[iK].shape[0]} \t {int(row[0])} \t {int(row[1])} \t {row[2]:2.5f} \t {row[3]:2.5f} \t {row[4]:2.5f}")
                if ecC[iK][-1, 4] > A:
                    xembM = np.hstack((xembM, xVnext))
                    iembV.append(ind)
                else:
                    terminator = True
            else:
                if showtxt >= 2:
                    row = ecC[iK][-1, :]
                    print(f"{ecC[iK].shape[0]} \t {int(row[0])} \t {int(row[1])} \t {row[2]:2.5f} \t {row[3]:2.5f} \t {row[4]:2.5f}")
                if len(iembV) == 1:
                    # Already handled above, but this branch is for len(iembV)==2
                    if ecC[iK][-1, 4] > A:
                        xembM = np.hstack((xembM, xVnext))
                        iembV.append(ind)
                    else:
                        terminator = True
                elif len(iembV) == 2:
                    if ecC[iK][-1, 4] > A:
                        xembM = np.hstack((xembM, xVnext))
                        iembV.append(ind)
                    else:
                        terminator = True
                else:
                    # For the fourth or larger embedding cycle 
                    if ecC[iK][-1, 4] > A and (ecC[iK][-1, 4] < ecC[iK][-2, 4] or ecC[iK][-2, 4] < ecC[iK][-3, 4]):
                        xembM = np.hstack((xembM, xVnext))
                        iembV.append(ind)
                    else:
                        terminator = True
        # Identify the lags of each variable in the embedding vector, if not empty, and compute the R measure for each driving variable.
        # Check if there exists any element in iembV that is not in the range for the response variable iK.
        if len(iembV) > 0 and np.any((np.array(iembV) < indlagM[iK, 0]) | (np.array(iembV) > indlagM[iK, 1])):
            # Find the lags of the variables
            xformM = np.full((len(iembV), 2), np.nan)
            for idx, val in enumerate(iembV):
                # Compute variable index and lag index (converting to MATLAB style indices)
                v_ind = int(np.ceil((val+1) / float(Lmax)))
                lag_val = (val+1) % int(Lmax)
                if lag_val == 0:
                    lag_val = int(Lmax)
                xformM[idx, 0] = v_ind
                xformM[idx, 1] = lag_val
            # Make computations only for the active variables, which are the variables included in the mixed embedding vector.
            activeV = np.unique(xformM[:, 0]).astype(int)
            # Store the lags of the response and remove it from the active variable list
            if np.any(activeV == (iK+1)):
                inowV = np.where(xformM[:, 0] == (iK+1))[0]
                xrespM = xembM[:, inowV]
                activeV = np.setdiff1d(activeV, np.array([iK+1]))
            else:
                xrespM = np.array([])  # This is the case where the response is not represented in the mixed embedding vector 
            KK = len(activeV)
            indKKM = np.full((KK, 2), np.nan)  # Start and end in xembM of the active variables
            iordembV = np.full((len(iembV),), np.nan)
            count_ord = 0
            for iKK in range(KK):
                inow_indices = np.where(xformM[:, 0] == activeV[iKK])[0]
                indKKM[iKK, :] = [count_ord, count_ord + len(inow_indices) - 1]
                iordembV[int(count_ord):int(count_ord+len(inow_indices))] = inow_indices
                count_ord = count_ord + len(inow_indices)
            iordembV = iordembV[:int(indKKM[KK-1, 1]+1)].astype(int)
            # The total embedding vector ordered with respect to the active variables and their lags, except from the response 
            xembM = xembM[:, iordembV]
            # Compute the entropy for the largest state space, containing the embedding vector and the future response vector.
            if xrespM.size == 0:
                xpastM = xembM
            else:
                xpastM = np.hstack((xrespM, xembM))
            combined = np.hstack((xFM, xpastM))
            _, dists = annMaxquery(combined, combined, int(nnei)+1)
            maxdistV = dists[:, -1]
            nyFV = nneighforgivenr(xFM, maxdistV - 1e-10)
            nwV = nneighforgivenr(xpastM, maxdistV - 1e-10)
            psi0V = np.column_stack((psi(nyFV), psi(nwV)))
            psinnei_val = psi(nnei)  # recompute if needed
            IyFw = psinnei_val + psi(N1) - np.mean(np.sum(psi0V, axis=1))  # I(y^T; w)
            # For each active (driving) variable build the arguments in I(y^T; w^X | w^Y w^Z) and then compute it.
            for iKK in range(KK):
                # indnowV: indices for the current driving variable in the ordered embedding vector
                indnowV = np.arange(int(indKKM[iKK, 0]), int(indKKM[iKK, 1]+1))
                irestV = np.setdiff1d(np.arange(xembM.shape[1]), indnowV)
                # Construct the conditioning embedding vector [w^Y w^Z]
                if irestV.size == 0 and (xrespM.size == 0):
                    xcondM = np.array([])
                elif irestV.size == 0 and (xrespM.size != 0):
                    xcondM = xrespM
                elif irestV.size != 0 and (xrespM.size == 0):
                    xcondM = xembM[:, irestV]
                else:
                    xcondM = np.hstack((xrespM, xembM[:, irestV]))
                # Compute I(y^T; w^X | w^Y w^Z) 
                if xcondM.size == 0:
                    IyFwXcond = IyFw
                else:
                    nxFcond = nneighforgivenr(np.hstack((xFM, xcondM)), maxdistV - 1e-10)
                    ncond = nneighforgivenr(xcondM, maxdistV - 1e-10)
                    psinowV = np.column_stack((psi(nxFcond), psi0V[:, 1] - psi(ncond)))
                    IyFwXcond = psinnei_val - np.mean(np.sum(psinowV, axis=1))
                # Adjust index for RM (activeV are 1-indexed)
                RM[activeV[iKK]-1, iK] = IyFwXcond / IyFw
        if ecC[iK] is not None and ecC[iK].size != 0:
            # Upon termination delete the last selected component.
            ecC[iK] = ecC[iK][:-1, :]
    if showtxt > 0:
        print("\n")
    return RM.T, ecC

def annMaxquery(query_points, data_points, k):
    # This function implements a k-nearest neighbors search.
    # query_points: array of shape (N, d)
    # data_points: array of shape (N, d)
    # k: number of neighbors to search for
    tree = KDTree(data_points, p=np.inf)  # p=inf specifies max norm
    # Query returns distances of shape (N, k) and indices of shape (N, k)
    dists, inds = tree.query(query_points, k=k)
    # Transpose the result to mimic MATLAB's output shape if needed.
    # In our code, we only use dists, and we take the last neighbor for each point.
    return inds, dists

def annMaxRvaryquery(query_points, data_points, rV, dummy, **kwargs):
    # This function implements a radius based neighbors search where each query point 
    # has its own radius (given by the vector rV).
    # query_points: array of shape (N, d)
    # data_points: array of shape (N, d)
    # rV: radius for each query point, array of shape (N,)
    N = query_points.shape[0]
    npV = np.empty(N)
    tree = KDTree(data_points, p=np.inf)  # p=inf specifies max norm
    # For each query point, count the number of neighbors within the given radius.
    for i in range(N):
        # query_radius returns a 1D array of indices within the radius rV[i]
        ind = tree.query_radius(query_points[i].reshape(1, -1), r=rV[i])
        npV[i] = len(ind[0])
    return npV

def nneighforgivenr(xM, rV):
    # xM: numpy array of shape (N, d)
    # rV: radius vector of shape (N,)
    npV = annMaxRvaryquery(xM, xM, rV, 1, search_sch='fr', radius=sqrt(1))
    npV = np.array(npV, dtype=float)
    # Replace zeros with ones as in the MATLAB code.
    npV[npV == 0] = 1
    return npV