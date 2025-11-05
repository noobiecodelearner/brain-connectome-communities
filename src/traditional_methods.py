"""
Traditional community detection methods
"""

import time
from collections import defaultdict
import community.community_louvain as community_louvain
from networkx.algorithms.community import (
    greedy_modularity_communities,
    label_propagation_communities
)
from .metrics import calculate_modularity, spatial_compactness
from .utils import community_to_partition


def run_traditional_methods(G, embedding_dict=None):
    """
    Run traditional community detection methods for comparison
    
    Parameters
    ----------
    G : networkx.Graph
        Input graph
    embedding_dict : dict, optional
        Node to embedding vector mapping for spatial compactness
        
    Returns
    -------
    dict
        Results for each method containing partition, modularity, time, etc.
    """
    print("Running traditional community detection methods...")
    results = {}
    
    # Louvain
    print("  Running Louvain method...")
    start_time = time.time()
    louvain_partition = community_louvain.best_partition(G, weight='weight')
    
    communities = defaultdict(set)
    for node, comm_id in louvain_partition.items():
        communities[comm_id].add(node)
    louvain_communities = list(communities.values())
    
    results['louvain'] = {
        'partition': louvain_partition,
        'modularity': calculate_modularity(G, louvain_communities),
        'time': time.time() - start_time,
        'n_communities': len(louvain_communities)
    }
    
    if embedding_dict:
        results['louvain']['spatial_compactness'] = spatial_compactness(
            embedding_dict, louvain_partition
        )
    
    # Greedy Modularity
    print("  Running Greedy modularity maximization...")
    start_time = time.time()
    greedy_communities = list(greedy_modularity_communities(G, weight='weight'))
    greedy_partition = community_to_partition(greedy_communities)
    
    results['greedy'] = {
        'partition': greedy_partition,
        'modularity': calculate_modularity(G, greedy_communities),
        'time': time.time() - start_time,
        'n_communities': len(greedy_communities)
    }
    
    if embedding_dict:
        results['greedy']['spatial_compactness'] = spatial_compactness(
            embedding_dict, greedy_partition
        )
    
    # Label Propagation
    print("  Running Label propagation...")
    start_time = time.time()
    lp_communities = list(label_propagation_communities(G))
    lp_partition = community_to_partition(lp_communities)
    
    results['label_propagation'] = {
        'partition': lp_partition,
        'modularity': calculate_modularity(G, lp_communities),
        'time': time.time() - start_time,
        'n_communities': len(lp_communities)
    }
    
    if embedding_dict:
        results['label_propagation']['spatial_compactness'] = spatial_compactness(
            embedding_dict, lp_partition
        )
    
    return results
