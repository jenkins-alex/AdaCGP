import torch
import sys
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

def set_seed(seed):
    """Set seed for numpy and torch for reproducibility

    Args:
        seed (int): seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_each_graph_filter(Psi, N, P):
    """return each graph filter from the flattened graph filter

    Args:
        Psi (torch.tensor): N*NP flattened graph filter
        N (int): number of nodes
        P (int): order of the filter

    Returns:
        torch.tensor: N x P x N tensor of graph filters
    """
    return Psi.view(N, P, N)

def pack_graph_filters(filters, N, P):
    """pack graph filters into a matrix

    Args:
        filters (torch.tensor): N x P x N tensor of graph filters
        N (int): number of nodes
        P (int): order of the filter

    Returns:
        torch.tensor: N x NP flattened graph filter
    """
    return torch.cat(filters, dim=1).view(N, N * P)

def get_each_graph_filter_np(Psi, N, P):
    """return each graph filter from the flattened graph filter

    Args:
        Psi (numpy.ndarray): N*NP flattened graph filter
        N (int): number of nodes
        P (int): order of the filter

    Returns:
        numpy.ndarray: N x P x N tensor of graph filters
    """
    return Psi.reshape(N, P, N)

def pack_graph_filters_np(filters, N, P):
    """pack graph filters into a matrix

    Args:
        filters (list or numpy.ndarray): List of filters or N x P x N array of graph filters
        N (int): number of nodes
        P (int): order of the filter

    Returns:
        numpy.ndarray: N x NP flattened graph filter
    """
    import numpy as np
    
    if isinstance(filters, list):
        return np.concatenate(filters, axis=1).reshape(N, N * P)
    else:
        return filters.reshape(N, N * P)
