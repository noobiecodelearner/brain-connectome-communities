"""
Brain Connectome Community Detection Package
Multi-objective optimization for brain network analysis
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .data_loader import load_connectome
from .embedding import compute_spectral_embedding
from .metrics import calculate_modularity, spatial_compactness
from .traditional_methods import run_traditional_methods
from .genetic_algorithm import run_ga, select_representative_solutions
from .consensus import calculate_consensus_partition
from .analysis import (
    analyze_partitions,
    compare_community_detection_methods,
    interpret_anatomical_meaning,
    perform_statistical_testing
)

__all__ = [
    'load_connectome',
    'compute_spectral_embedding',
    'calculate_modularity',
    'spatial_compactness',
    'run_traditional_methods',
    'run_ga',
    'select_representative_solutions',
    'calculate_consensus_partition',
    'analyze_partitions',
    'compare_community_detection_methods',
    'interpret_anatomical_meaning',
    'perform_statistical_testing'
]
