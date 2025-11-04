"""
Utility functions for partition conversion and helper operations
"""

from collections import defaultdict
import numpy as np


def partition_to_community_list(partition, nodes):
    """Convert partition dict to community list format"""
    communities = defaultdict(list)
    for node in nodes:
        communities[partition[node]].append(node)
    return list(communities.values())


def community_to_partition(communities):
    """Convert list of communities to partition dict"""
    partition = {}
    for i, community in enumerate(communities):
        for node in community:
            partition[node] = i
    return partition


def decode_individual(individual, nodes):
    """Decode GA individual (chromosome) to partition dict"""
    return {nodes[i]: individual[i] for i in range(len(individual))}


def estimate_max_communities(G):
    """Estimate the maximum number of communities based on network size"""
    n = G.number_of_nodes()
    return min(int(np.sqrt(n)), 50)


def normalize_metrics(metrics_dict):
    """
    Normalize metrics dictionary to ensure consistent structure
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary of metrics that may contain Series, lists, or nested dicts
        
    Returns
    -------
    normalized : dict
        Flattened dictionary with scalar values
    """
    import pandas as pd
    
    normalized = {}
    
    for key, value in metrics_dict.items():
        if isinstance(value, pd.Series):
            if len(value) == 1:
                normalized[key] = value.iloc[0]
            else:
                if value.dtype in ['object', 'string']:
                    normalized[key] = value.iloc[0]
                else:
                    normalized[key] = value.mean()
        elif isinstance(value, (list, np.ndarray)):
            if len(value) == 0:
                normalized[key] = np.nan
            elif len(value) == 1:
                normalized[key] = value[0]
            else:
                normalized[key] = np.mean(value)
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                new_key = f"{key}_{sub_key}"
                if isinstance(sub_value, pd.Series):
                    if len(sub_value) == 1:
                        normalized[new_key] = sub_value.iloc[0]
                    else:
                        if sub_value.dtype in ['object', 'string']:
                            normalized[new_key] = sub_value.iloc[0]
                        else:
                            normalized[new_key] = sub_value.mean()
                elif isinstance(sub_value, (list, np.ndarray)):
                    if len(sub_value) == 0:
                        normalized[new_key] = np.nan
                    elif len(sub_value) == 1:
                        normalized[new_key] = sub_value[0]
                    else:
                        normalized[new_key] = np.mean(sub_value)
                else:
                    normalized[new_key] = sub_value
        else:
            normalized[key] = value
    
    return normalized
