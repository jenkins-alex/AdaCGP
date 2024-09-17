import numpy as np

class TIRSO:

    def __init__(self):
        pass

    def run(self, P, lambda_, gamma, mu, sigma, v_alpha, m_y, m_A_initial, b_trace, b_diminishing):
        # This function computes an estimate via TIRSO
        # INPUT: P, lambda, gamma, mu, sigma, v_alpha, m_y, m_A_initial
        # OUTPUT: A (a_n, n=1,...,N)
        
        N, T = m_y.shape
        assert len(v_alpha) == T, 'v_alpha should be a vector of size T'
        assert m_A_initial.shape[0] == N and m_A_initial.shape[1] == N * P, 'A_initial should have of size N X NP'
        
        Phi_prev = sigma**2 * np.eye(N * P)  # initializing Phi
        r_prev = np.zeros((N * P, N))  # r has NP X N size to avoid transpose 
        a_prev = m_A_initial
        m_r = np.zeros((N * P, N))
        t_A = np.zeros((N, N * P, T))
        v_strongcvxPar = np.zeros(T)
        
        for t in range(P + 1, T):
            # receive data y[t]
            # form g[t] via g[t]= vec([y[t-1],...,y[t-P]]^T)
            y_prev = m_y[:, t - P:t]
            aux = np.fliplr(y_prev).T
            g = aux.flatten()

            # update Phi
            Phi_t = gamma * Phi_prev + mu * np.outer(g, g)
            v_strongcvxPar[t] = np.min(np.linalg.eigvals(Phi_t))
            
            if b_trace == 1:
                if b_diminishing == 1:
                    v_alpha[t] = 1 / (np.trace(Phi_t) * np.sqrt(t))
                else:
                    v_alpha[t] = 1 / np.trace(Phi_t)

            for n in range(N):
                # update r_n
                m_r[:, n] = gamma * r_prev[:, n] + mu * m_y[n, t] * g
                grad_n = Phi_t @ a_prev[n, :].T - m_r[:, n]  # v_n in the paper

                for nprime in range(N):
                    groupindices = slice((nprime - 1) * P, nprime * P)  # n,n' group indices
                    af_nnprime = a_prev[n, groupindices] - v_alpha[t] * grad_n[groupindices]
                    
                    if n != nprime:
                        t_A[n, groupindices, t] = np.maximum(0, (1 - (v_alpha[t] * lambda_) / np.linalg.norm(af_nnprime))) * af_nnprime  # indicator rem
                    else:
                        t_A[n, groupindices, t] = af_nnprime
            
            # to store A, Phi, and r_n
            a_prev = t_A[:, :, t]
            Phi_prev = Phi_t
            r_prev = m_r
        
        return t_A, v_strongcvxPar

