import numpy as np
from .graph_generation import generate_er_graph, generate_sbm_graph, \
    generate_kr_graph, generate_pl_graph, generate_random_graph, generate_kr_laplacian_graph
from scipy import sparse

def get_graph_topology(graph_type, N, max_retries=100, **kwargs):
    """Generate the graph topology with retry mechanism

    Args:
        graph_type (str): graph type
        N (int): number of nodes
        max_retries (int): maximum number of retry attempts

    Returns:
        np.ndarray: adjacency matrix of the generated graph

    Raises:
        ValueError: If unable to generate a valid graph after max_retries
    """
    graph_generators = {
        'ER': generate_er_graph,
        'SBM': generate_sbm_graph,
        'KR': generate_kr_graph,
        'PL': generate_pl_graph,
        'RANDOM': generate_random_graph,
        'RAND_LAP': generate_kr_laplacian_graph
    }

    if graph_type not in graph_generators:
        raise ValueError(f"Graph type {graph_type} not implemented")

    for attempt in range(max_retries):
        try:
            kwargs['seed'] = max_retries + attempt
            weight_matrix = graph_generators[graph_type](N, **kwargs)

            if sparse.issparse(weight_matrix):
                weight_matrix = weight_matrix.toarray()

            if weight_matrix.shape != (N, N):
                raise ValueError(f"Graph topology has incorrect shape: expected ({N}, {N}), got {weight_matrix.shape}")
            return weight_matrix

        except ValueError as e:
            if attempt == max_retries - 1:
                raise ValueError(f"Failed to generate valid graph after {max_retries} attempts: {str(e)}")
            print(f"Attempt {attempt + 1} failed, retrying...")
    raise RuntimeError("Unexpected error in get_graph_topology")

def generate_graph_filters(N, P, graph_type, **kwargs):
    """Generate graph filters for the adaptive graph AR model

    Args:
        N (int): number of nodes in the graph
        P (int): order of the filter
        graph_type ('str'): type of graph to generate

    Returns:
        list[np.ndarray]: graph filters, weight matrix, filter coefficients
    """

    # setup graph and filter coefficients
    weight_matrix = get_graph_topology(graph_type, N, **kwargs)
    filter_coefficients = generate_hs(P)

    # get graph filters
    filter_coefs = []
    start = 0
    for p in range(1, 4):
        end = start + p
        coefficients = filter_coefficients[start:end+1]
        start = end + 1
        filter_coefs.append(coefficients)

    # structure the graph filters
    power_list = [[filter_coefs[p-1][k] * np.linalg.matrix_power(weight_matrix, k) for k in range(p+1)] for p in range(1, P+1)]
    graph_filters = np.array([np.array(item).sum(axis=0) for item in power_list])
    graph_filters_flat = np.hstack([graph_filters[p, :, :] for p in range(P)])
    return graph_filters_flat, weight_matrix, filter_coefficients

def generate_hs(P):
    """Generate filter coefficients for the vertex-time filter

    Args:
        P (int): filter order

    Returns:
        np.array: filter coefficients
    """
    cs = []
    for i in range(2, P+1):
        for j in range(i+1):
            U1 = np.random.uniform(-1, -0.45)
            U2 = np.random.uniform(0.45, 1)
            cij = 0.5 * U1 + 0.5 * U2
            cij /= 2**(i+j)
            cij /= 1.5
            cs.append(cij)
    
    # randomly drop 50% of the coefficients
    cs = np.array(cs)
    mask = np.random.choice([0, 1], size=len(cs), p=[0.5, 0.5])
    cs = cs * mask
    cs = [0, 1] + list(cs)
    return np.array(cs)