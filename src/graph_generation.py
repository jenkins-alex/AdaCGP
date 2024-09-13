import numpy as np
import networkx as nx
import scipy.sparse as sp
from .utils import normalize_matrix, soft_threshold

def generate_kr_graph(n=1000, xi=3, **kwargs):
    """generate a k-regular graph topology of n nodes with xi connections per node

    Args:
        n (int, optional): number of nodes in the graph. Defaults to 1000.
        xi (int, optional): number of connections for each node. Defaults to 3.

    Returns:
        scipy.sparse: adjacency matrix of the generated graph
    """
    weights = np.random.uniform(0.5, 1.0, size=(n, xi))
    matrix = sp.lil_matrix((n, n))
    
    for i in range(n):
        matrix[i, i] = -1
        for j in range(1, xi + 1):
            matrix[i, (i - j) % n] = weights[i, j - 1]
            matrix[i, (i + j) % n] = weights[i, j - 1]
    
    matrix = normalize_matrix(matrix, 1.5)
    return matrix

def generate_sbm_graph(n=1000, num_clusters=10, **kwargs):
    """generate a stochastic block model graph with n nodes and num_clusters clusters

    Args:
        n (int, optional): number of nodes in the graph. Defaults to 1000.
        num_clusters (int, optional): number of clusters. Defaults to 10.

    Returns:
        scipy.sparse: adjacency matrix of the generated graph
    """
    sizes = [n // num_clusters] * num_clusters
    prob_matrix = 0.05 * np.eye(num_clusters)
    prob_matrix += np.random.uniform(0, 0.04, size=(num_clusters, num_clusters))
    prob_matrix[prob_matrix < 0.025] = 0
    prob_matrix = np.triu(prob_matrix) + np.triu(prob_matrix, 1).T
    
    G = nx.stochastic_block_model(sizes, prob_matrix)
    for u, v, d in G.edges(data=True):
        G[u][v]['weight'] = np.random.laplace(0, 2)
    
    matrix = nx.to_scipy_sparse_array(G)
    matrix = normalize_matrix(matrix, 1.1)
    return matrix

def generate_er_graph(n=1000, **kwargs):
    """Erdős-Renyi graph with n nodes and edge probability p

    Args:
        n (int, optional): number of nodes. Defaults to 1000.
        p (float, optional): probability of edge. Defaults to 0.04.

    Returns:
        scipy.sparse: adjacency matrix of the generated graph
    """
    W = np.random.normal(0, 1, n*n)
    W[np.abs(W) >= 1.8] = 0
    W[np.abs(W) <= 1.6] = 0
    W = W.reshape(n, n)

    G = nx.from_numpy_array(W, create_using=nx.DiGraph)
    matrix = sp.lil_matrix((n, n))
    for u, v, d in G.edges(data=True):
        weight = d['weight']
        matrix[u, v] = soft_threshold(weight, 1.5)
    matrix = normalize_matrix(matrix, 1.5)
    return matrix

def generate_pl_graph(n=1000, initial_nodes=15, p=0.8, **kwargs):
    """Power law graph with n nodes and initial_nodes nodes

    Args:
        n (int, optional): number of nodes in the graph. Defaults to 1000.
        initial_nodes (int, optional): initial nodes to begin preferential attachment scheme. Defaults to 15.
        p (float, optional): probability of attachment between initial nodes. Defaults to 0.8.

    Returns:
        scipy.sparse: adjacency matrix of the generated graph
    """
    G = nx.erdos_renyi_graph(initial_nodes, p)
    matrix = sp.lil_matrix((n, n))

    for u, v in G.edges():
        matrix[u, v] = 1.0

    # Adding new nodes with two connections each following a preferential attachment scheme
    for new_node in range(initial_nodes, n):
        existing_nodes = np.arange(new_node)
        degrees = np.array((matrix != 0).sum(axis=1)).flatten()[:new_node]

        # ensure there are no negative or zero probabilities
        if degrees.sum() == 0:
            prob = np.ones_like(degrees) / len(degrees)  # Use uniform probabilities if all degrees are zero
        else:
            prob = degrees / degrees.sum()

        cn = np.random.choice(existing_nodes, replace=False, p=prob)
        weights = np.random.normal(0, 1, size=2)
        weights += np.sign(weights) * 0.25
        matrix[new_node, cn] = weights[0]
        matrix[cn, new_node] = weights[1]

    # Set the diagonal to -0.5
    for i in range(n):
        matrix[i, i] = -0.5

    # Normalize the matrix by 1.5 times its largest eigenvalue
    matrix = normalize_matrix(matrix, 1.5)
    return matrix

def generate_random_graph(n, **kwargs):
    """create a random weight matrix for the graph

    Args:
        n (int): number of nodes in the graph

    Returns:
        sp.sparse: random weight matrix
    """

    W = np.random.normal(0, 1, n*n)
    max_weight = np.max(np.abs(W))
    W[np.abs(W) > 0.7 * max_weight] = 0
    W[np.abs(W) < 0.3 * max_weight] = 0
    W = W.reshape(n, n)

    G = nx.from_numpy_array(W, create_using=nx.DiGraph)
    matrix = sp.lil_matrix((n, n))
    for u, v, d in G.edges(data=True):
        matrix[u, v] = d['weight']
    matrix = normalize_matrix(matrix, 1.5)
    return matrix

def create_random_matrix_as_in_paper(N):
    # as in Methods of Adaptive Signal Processing on Graphs Using Vertex-Time Autoregressive Models
    W = np.random.normal(0, 1, size=(N, N))

    # threshold weight matrix to between 0.3 and 0.7 of max weight
    max_weight = np.max(np.abs(W))
    W[np.abs(W)>0.7*max_weight] = 0
    W[np.abs(W)<0.3*max_weight] = 0
    
    # calculate the eigenvalues of W and normalise by 1.5x largest eigenvalue for stable process
    w, _ = np.linalg.eig(W)
    max_eig = np.max(np.abs(w.real))
    W /= 1.5 * max_eig
    return W
    
    