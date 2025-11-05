"""
Genetic algorithm for multi-objective community detection
"""

import time
import random
import multiprocessing
from functools import partial
import numpy as np
from deap import base, creator, tools, algorithms
import community.community_louvain as community_louvain

from .metrics import calculate_modularity, spatial_compactness, community_entropy
from .utils import decode_individual, estimate_max_communities


def evaluate(individual, G, nodes, embedding_dict):
    """
    Evaluate GA individual on multiple objectives
    
    Parameters
    ----------
    individual : list
        GA chromosome (community assignments)
    G : networkx.Graph
        Input graph
    nodes : list
        List of nodes
    embedding_dict : dict
        Node to embedding vector mapping
        
    Returns
    -------
    tuple
        (negative modularity, compactness, negative entropy)
    """
    partition = decode_individual(individual, nodes)
    mod = calculate_modularity(G, partition)
    comp = spatial_compactness(embedding_dict, partition)
    ent = community_entropy(partition)
    
    # Return objectives to minimize
    norm_mod = 1 - mod  # Minimize (so we maximize modularity)
    norm_comp = comp    # Minimize
    norm_ent = -ent     # Minimize (maximize entropy)
    
    return norm_mod, norm_comp, norm_ent


def mutate_community_assignment(individual, max_communities):
    """Custom mutation that randomly changes a node's community assignment"""
    if random.random() < 0.2:
        idx = random.randint(0, len(individual) - 1)
        individual[idx] = random.randint(0, max_communities - 1)
    return individual,


def run_ga(G, embedding_dict, pop_size=100, ngen=50, cxpb=0.7, mutpb=0.2):
    """
    Run the genetic algorithm for community detection
    
    Parameters
    ----------
    G : networkx.Graph
        Input graph
    embedding_dict : dict
        Node to embedding vector mapping
    pop_size : int
        Population size
    ngen : int
        Number of generations
    cxpb : float
        Crossover probability
    mutpb : float
        Mutation probability
        
    Returns
    -------
    pareto_front : deap.tools.ParetoFront
        Pareto-optimal solutions
    pareto_partitions : list
        List of partition dicts
    logbook : deap.tools.Logbook
        Evolution statistics
    """
    print("Setting up genetic algorithm...")
    nodes = list(G.nodes())
    max_communities = estimate_max_communities(G)
    
    # Setup DEAP
    if 'FitnessMulti' in creator.__dict__:
        del creator.FitnessMulti
    if 'Individual' in creator.__dict__:
        del creator.Individual
    
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -0.1))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    toolbox = base.Toolbox()
    
    # Parallel processing
    num_processors = max(1, multiprocessing.cpu_count() - 1)
    pool = multiprocessing.Pool(processes=num_processors)
    toolbox.register("map", pool.map)
    
    # Register operators
    toolbox.register("attr_int", random.randint, 0, max_communities - 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                    toolbox.attr_int, n=len(nodes))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    evaluate_partial = partial(evaluate, G=G, nodes=nodes, embedding_dict=embedding_dict)
    toolbox.register("evaluate", evaluate_partial)
    toolbox.register("mate", tools.cxUniform, indpb=0.3)
    toolbox.register("mutate", mutate_community_assignment, max_communities=max_communities)
    toolbox.register("select", tools.selNSGA2)
    
    # Initialize population with Louvain seeds
    pop = toolbox.population(n=pop_size - 5)
    
    for _ in range(5):
        partition = community_louvain.best_partition(G)
        individual = creator.Individual([
            partition[node] if partition[node] < max_communities 
            else random.randint(0, max_communities - 1)
            for node in nodes
        ])
        individual.fitness.values = toolbox.evaluate(individual)
        pop.append(individual)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    print(f"Starting GA evolution for {ngen} generations...")
    start_time = time.time()
    
    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "max", "avg", "std"
    
    pareto_front = tools.ParetoFront()
    
    # Evolution loop
    for gen in range(ngen):
        offspring = algorithms.varAnd(pop, toolbox, cxpb, mutpb)
        
        fitnesses = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
        
        pop = toolbox.select(pop + offspring, pop_size)
        pareto_front.update(pop)
        
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(offspring), **record)
        
        if gen % 10 == 0:
            print(f"Generation {gen}/{ngen}, Time: {time.time() - start_time:.2f}s")
            print(f"  Min: {record['min']}, Max: {record['max']}")
    
    pool.close()
    pool.join()
    
    print(f"GA evolution completed in {time.time() - start_time:.2f} seconds")
    
    # Decode partitions
    pareto_partitions = [decode_individual(ind, nodes) for ind in pareto_front]
    
    return pareto_front, pareto_partitions, logbook


def calculate_crowding_distance(pareto_front):
    """Calculate crowding distance for each solution in the Pareto front"""
    n = len(pareto_front)
    if n <= 2:
        return [float('inf')] * n
    
    fitness_values = np.array([ind.fitness.values for ind in pareto_front])
    crowding_distances = np.zeros(n)
    
    for obj_idx in range(fitness_values.shape[1]):
        sorted_indices = np.argsort(fitness_values[:, obj_idx])
        
        crowding_distances[sorted_indices[0]] = float('inf')
        crowding_distances[sorted_indices[-1]] = float('inf')
        
        obj_range = fitness_values[sorted_indices[-1], obj_idx] - \
                   fitness_values[sorted_indices[0], obj_idx]
        
        if obj_range == 0:
            continue
        
        for i in range(1, n - 1):
            crowding_distances[sorted_indices[i]] += \
                (fitness_values[sorted_indices[i + 1], obj_idx] - 
                 fitness_values[sorted_indices[i - 1], obj_idx]) / obj_range
    
    return crowding_distances.tolist()


def find_knee_point(pareto_front):
    """Find knee point using maximum distance from line connecting extremes"""
    fitness_values = np.array([ind.fitness.values for ind in pareto_front])
    
    modularity = 1 - fitness_values[:, 0]
    compactness = fitness_values[:, 1]
    
    mod_norm = (modularity - modularity.min()) / (modularity.max() - modularity.min() + 1e-10)
    comp_norm = (compactness - compactness.min()) / (compactness.max() - compactness.min() + 1e-10)
    
    sorted_indices = np.argsort(comp_norm)
    mod_sorted = mod_norm[sorted_indices]
    comp_sorted = comp_norm[sorted_indices]
    
    p1 = np.array([comp_sorted[0], mod_sorted[0]])
    p2 = np.array([comp_sorted[-1], mod_sorted[-1]])
    
    distances = []
    for i in range(len(comp_sorted)):
        point = np.array([comp_sorted[i], mod_sorted[i]])
        dist = np.abs(np.cross(p2 - p1, p1 - point)) / (np.linalg.norm(p2 - p1) + 1e-10)
        distances.append(dist)
    
    knee_idx_sorted = np.argmax(distances)
    knee_idx = sorted_indices[knee_idx_sorted]
    
    return knee_idx


def select_representative_solutions(pareto_front, pareto_partitions, n_representatives=5):
    """
    Select representative solutions from Pareto front
    
    Parameters
    ----------
    pareto_front : deap.tools.ParetoFront
        Pareto-optimal solutions
    pareto_partitions : list
        List of partition dicts
    n_representatives : int
        Number of representatives to select
        
    Returns
    -------
    dict
        Selected representative solutions
    """
    if len(pareto_front) <= n_representatives:
        return {f"GA Solution {i}": pareto_partitions[i] 
                for i in range(len(pareto_partitions))}
    
    representatives = {}
    selected_indices = set()
    
    # Best modularity
    fitness_values = np.array([ind.fitness.values for ind in pareto_front])
    best_mod_idx = np.argmin(fitness_values[:, 0])
    representatives["GA Best Modularity"] = pareto_partitions[best_mod_idx]
    selected_indices.add(best_mod_idx)
    
    # Best spatial compactness
    best_comp_idx = np.argmin(fitness_values[:, 1])
    if best_comp_idx not in selected_indices:
        representatives["GA Best Compactness"] = pareto_partitions[best_comp_idx]
        selected_indices.add(best_comp_idx)
    
    # Knee point
    knee_idx = find_knee_point(pareto_front)
    if knee_idx not in selected_indices:
        representatives["GA Knee Point"] = pareto_partitions[knee_idx]
        selected_indices.add(knee_idx)
    
    # High crowding distance solutions
    crowding_distances = calculate_crowding_distance(pareto_front)
    remaining_slots = n_representatives - len(selected_indices)
    
    if remaining_slots > 0:
        cd_with_idx = [(i, cd) for i, cd in enumerate(crowding_distances) 
                       if i not in selected_indices and cd != float('inf')]
        cd_with_idx.sort(key=lambda x: x[1], reverse=True)
        
        for i in range(min(remaining_slots, len(cd_with_idx))):
            idx = cd_with_idx[i][0]
            representatives[f"GA Diverse {i+1}"] = pareto_partitions[idx]
            selected_indices.add(idx)
    
    return representatives
