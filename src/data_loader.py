"""
Data loading and preprocessing functions
"""

import pandas as pd
import networkx as nx


def load_connectome(file_path="connectome.csv"):
    """
    Load connectome data and create networkx graph
    
    Parameters
    ----------
    file_path : str
        Path to the connectome CSV file
        
    Returns
    -------
    G : networkx.Graph
        Graph with nodes and weighted edges
    df : pandas.DataFrame
        Original dataframe
    """
    print("Loading connectome data...")
    df = pd.read_csv(file_path, sep=";")
    
    G = nx.Graph()
    
    for _, row in df.iterrows():
        G.add_node(row['id node1'],
                   name=row['name node1'],
                   parent_id=row['parent id node1'],
                   parent_name=row['parent name node1'])
        
        G.add_node(row['id node2'],
                   name=row['name node2'],
                   parent_id=row['parent id node2'],
                   parent_name=row['parent name node2'])
        
        G.add_edge(row['id node1'], row['id node2'],
                   weight=row['edge weight(med nof)'],
                   confidence=row['edge confidence'])
    
    G.remove_edges_from(nx.selfloop_edges(G))
    
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G, df
