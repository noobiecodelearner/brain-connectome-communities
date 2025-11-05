"""
Community detection metrics
"""

import numpy as np
import networkx as nx
from collections import defaultdict
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform


def calculate_modularity(G, partition):
    """
    Calculate modularity for a partition
    
    Parameters
    ----------
    G : networkx.Graph
        Input graph
    partition : dict
        Node to community mapping
        
    Returns
    -------
    float
        Modularity value
    """
    if isinstance(partition, dict):
        comm_map = defaultdict(set)
        for node, comm_id in partition.items():
            comm_map[comm_id].add(node)
        community_list = list(comm_map.values())
    else:
        community_list = partition
    
    return nx.community.modularity(G, community_list, weight='weight')


def spatial_compactness(embedding_dict, partition):
    """
    Calculate spatial compactness as average intra-community variance
    
    Parameters
    ----------
    embedding_dict : dict
        Node to embedding vector mapping
    partition : dict
        Node to community mapping
        
    Returns
    -------
    float
        Average spatial compactness (lower is better)
    """
    if isinstance(partition, dict):
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)
        community_list = list(communities.values())
    else:
        community_list = partition
    
    community_variances = []
    for community in community_list:
        if len(community) <= 1:
            continue
        
        community_embeddings = np.array([embedding_dict[node] for node in community])
        variance = np.mean(np.var(community_embeddings, axis=0))
        community_variances.append(variance)
    
    if not community_variances:
        return float('inf')
    
    return np.mean(community_variances)


def community_entropy(partition):
    """
    Calculate entropy of community size distribution
    
    Parameters
    ----------
    partition : dict
        Node to community mapping
        
    Returns
    -------
    float
        Entropy value
    """
    _, counts = np.unique(list(partition.values()), return_counts=True)
    return entropy(counts)


def calculate_spatial_coherence(G, partition, embedding_dict):
    """
    Calculate spatial coherence measure
    
    Parameters
    ----------
    G : networkx.Graph
        Input graph
    partition : dict
        Node to community mapping
    embedding_dict : dict
        Node to embedding vector mapping
        
    Returns
    -------
    float
        Average spatial coherence score
    """
    nodes = list(G.nodes())
    embeddings = np.array([embedding_dict[node] for node in nodes])
    communities = [partition[node] for node in nodes]
    
    distances = squareform(pdist(embeddings))
    
    k = min(10, len(nodes) - 1)
    coherence_scores = []
    
    for i in range(len(nodes)):
        nearest_indices = np.argsort(distances[i])[1:k + 1]
        same_community_count = sum(1 for j in nearest_indices 
                                   if communities[j] == communities[i])
        coherence_scores.append(same_community_count / k)
    
    return np.mean(coherence_scores)
