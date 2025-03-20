import numpy as np
from src.models.batch.VAR import VAR

class GrangerVAR(VAR):
    """
    Implementation of Granger VAR model for time series forecasting and causal inference.
    """
    def __init__(self, N, hyperparams, device):
        super().__init__(N, hyperparams, device)
        self.N = N

    def identify_causal_W(self, var_model_out):
        # Granger causality
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