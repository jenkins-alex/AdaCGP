import numpy as np
from scipy.linalg import eigvalsh

def normalize_matrix(matrix, factor):
    """normalise the matrix by dividing it by the maximum eigenvalue * factor

    Args:
        matrix (scipy.sparse): matrix to be normalised
        factor (float): factor to multiply the maximum eigenvalue by

    Returns:
        scipy.sparse: normalised matrix
    """
    max_eigenvalue = eigvalsh(matrix.toarray()).max()
    return matrix / (factor * max_eigenvalue)

def soft_threshold(v, threshold):
    """soft threshold an array

    Args:
        v (np.ndarray): array to be thresholded
        threshold (float): threshold value

    Returns:
        np.ndarray: thresholded array
    """
    result = np.sign(v) * np.maximum(np.abs(v) - threshold, 0)
    return result
