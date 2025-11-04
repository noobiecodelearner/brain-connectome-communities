"""
Spectral embedding computation
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import laplacian as csgraph_laplacian


def compute_spectral_embedding(G, dim=8):
    """
    Compute spectral embedding for spatial compactness calculation
    
    Parameters
    ----------
    G : networkx.Graph
        Input graph
    dim : int
        Embedding dimension
        
    Returns
    -------
    embedding_dict : dict
        Node to embedding vector mapping
    embedding_matrix : numpy.ndarray
        Full embedding matrix
    eigenvalues : numpy.ndarray
        Computed eigenvalues
    """
    print("Computing spectral embedding...")
    
    node_list = list(G.nodes())
    node_index = {node: i for i, node in enumerate(node_list)}
    n = len(node_list)
    
    row, col, data = [], [], []
    
    for u, v, d in G.edges(data=True):
        i, j = node_index[u], node_index[v]
        weight = d.get("weight", 1.0)
        row.extend([i, j])
        col.extend([j, i])
        data.extend([weight, weight])
    
    A = csr_matrix((data, (row, col)), shape=(n, n))
    L = csgraph_laplacian(A, normed=True)
    
    try:
        eigenvalues, eigenvectors = eigsh(L, k=dim + 1, which='SM', 
                                         tol=1e-3, maxiter=10000, ncv=30)
    except Exception as e:
        print(f"[ERROR] eigsh failed: {e}")
        raise
    
    embedding_matrix = eigenvectors[:, 1:dim + 1]
    embedding_dict = {node: embedding_matrix[i] for i, node in enumerate(node_list)}
    
    return embedding_dict, embedding_matrix, eigenvalues
