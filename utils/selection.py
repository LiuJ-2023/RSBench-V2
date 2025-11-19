"""
Selection operations for evolutionary algorithms.

This module provides selection mechanisms used in various evolutionary
algorithms, including environment selection and indicator-based selection.
"""

import numpy as np
from typing import List, Tuple, Any
from .dominance import nondominated_sort, crowding_distance


def environment_selection(population: List[Any], N: int) -> Tuple[List[Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Environmental selection in NSGA-II.
    
    Args:
        population: Current population [individuals, objectives]
        N: Number of selected individuals
        
    Returns:
        Tuple of (selected_population, selected_objectives, front_numbers, crowding_distances, indices)
    """
    individuals, objectives = population
    
    # Non-dominated sorting
    front_no, max_front = nondominated_sort(objectives, N)
    
    # Select individuals from fronts
    next_label = [False for _ in range(front_no.size)]
    for i in range(front_no.size):
        if front_no[i] < max_front:
            next_label[i] = True
    
    # Calculate crowding distance
    crowd_dis = crowding_distance(objectives, front_no)
    
    # Handle last front
    last_front_indices = [i for i in range(len(front_no)) if front_no[i] == max_front]
    if last_front_indices:
        rank = np.argsort(-crowd_dis[last_front_indices])
        delta_n = rank[: (N - int(np.sum(next_label)))]
        rest = [last_front_indices[i] for i in delta_n]
        for i in rest:
            next_label[i] = True
    
    # Get selected indices
    selected_indices = np.array([i for i in range(len(next_label)) if next_label[i]])
    
    # Build selected population
    selected_individuals = [individuals[i] for i in selected_indices]
    selected_objectives = objectives[selected_indices]
    selected_front_no = front_no[selected_indices]
    selected_crowd_dis = crowd_dis[selected_indices]
    
    # Ensure correct population size
    while len(selected_individuals) != N:
        if len(selected_individuals) > N:
            # Remove random individual
            idx_ = np.random.randint(0, len(selected_individuals))
            selected_individuals.pop(idx_)
            selected_objectives = np.delete(selected_objectives, idx_, axis=0)
            selected_front_no = np.delete(selected_front_no, idx_)
            selected_crowd_dis = np.delete(selected_crowd_dis, idx_)
        elif len(selected_individuals) < N:
            # Add random individual from original population
            idx_ = np.random.randint(0, len(individuals))
            selected_individuals.append(individuals[idx_])
            selected_objectives = np.vstack([selected_objectives, objectives[idx_:idx_+1]])
            selected_front_no = np.append(selected_front_no, front_no[idx_])
            selected_crowd_dis = np.append(selected_crowd_dis, crowd_dis[idx_])
    
    return selected_individuals, selected_objectives, selected_front_no, selected_crowd_dis, selected_indices


def IBEA_selection(population: List[Any], 
                  objectives: np.ndarray, 
                  offspring: List[Any], 
                  offspring_objectives: np.ndarray, 
                  N: int, 
                  kappa: float = 0.05) -> Tuple[List[Any], np.ndarray]:
    """
    Indicator-Based Evolutionary Algorithm selection.
    
    Args:
        population: Current population
        objectives: Objective values of current population
        offspring: Offspring population
        offspring_objectives: Objective values of offspring
        N: Population size
        kappa: Scaling factor for fitness calculation
        
    Returns:
        Tuple of (selected_population, selected_objectives)
    """
    # Combine population and offspring
    combined_pop = population + offspring
    combined_obj = np.vstack([objectives, offspring_objectives])
    
    # Calculate fitness using epsilon indicator
    fitness, indicator_matrix, scaling_factor = _calculate_ibea_fitness(combined_obj, kappa)
    
    # Selection loop
    while len(combined_pop) > N:
        # Find individual with minimum fitness
        min_idx = np.argmin(fitness)
        
        # Update fitness values
        fitness = fitness + np.exp(-indicator_matrix[min_idx:min_idx+1, :] / 
                                 scaling_factor[:, min_idx:min_idx+1] / kappa)
        
        # Remove individual
        fitness = np.delete(fitness, min_idx)
        combined_pop.pop(min_idx)
        combined_obj = np.delete(combined_obj, min_idx, axis=0)
        indicator_matrix = np.delete(indicator_matrix, min_idx, axis=0)
        indicator_matrix = np.delete(indicator_matrix, min_idx, axis=1)
        scaling_factor = np.delete(scaling_factor, min_idx, axis=1)
    
    return combined_pop, combined_obj


def _calculate_ibea_fitness(objectives: np.ndarray, kappa: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate IBEA fitness using epsilon indicator.
    
    Args:
        objectives: Objective values
        kappa: Scaling factor
        
    Returns:
        Tuple of (fitness_values, indicator_matrix, scaling_factor)
    """
    N = objectives.shape[0]
    
    # Normalize objectives
    obj_max = np.max(objectives, axis=0, keepdims=True)
    obj_min = np.min(objectives, axis=0, keepdims=True)
    normalized_obj = (objectives - obj_min) / (obj_max - obj_min + 1e-10)
    
    # Calculate epsilon indicator matrix
    indicator_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                # Epsilon indicator: minimum factor by which solution i needs to be scaled
                # to weakly dominate solution j
                epsilon = np.max(normalized_obj[i, :] - normalized_obj[j, :])
                indicator_matrix[i, j] = epsilon
    
    # Calculate scaling factor
    scaling_factor = np.max(np.abs(indicator_matrix), axis=0, keepdims=True)
    
    # Calculate fitness
    fitness = np.sum(-np.exp(-indicator_matrix / scaling_factor / kappa), axis=0) + 1
    
    return fitness, indicator_matrix, scaling_factor


def tournament_selection(population: List[Any], 
                        objectives: np.ndarray, 
                        tournament_size: int = 2) -> List[Any]:
    """
    Tournament selection for single-objective optimization.
    
    Args:
        population: Current population
        objectives: Objective values (single objective)
        tournament_size: Size of tournament
        
    Returns:
        Selected individuals
    """
    selected = []
    
    for _ in range(len(population)):
        # Select tournament participants
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_objectives = objectives[tournament_indices]
        
        # Select winner (best objective value)
        winner_idx = tournament_indices[np.argmin(tournament_objectives)]
        selected.append(population[winner_idx])
    
    return selected


def roulette_wheel_selection(population: List[Any], 
                           fitness: np.ndarray) -> List[Any]:
    """
    Roulette wheel selection based on fitness values.
    
    Args:
        population: Current population
        fitness: Fitness values (higher is better)
        
    Returns:
        Selected individuals
    """
    # Ensure positive fitness values
    fitness = fitness - np.min(fitness) + 1e-10
    
    # Calculate selection probabilities
    total_fitness = np.sum(fitness)
    probabilities = fitness / total_fitness
    
    # Select individuals
    selected_indices = np.random.choice(len(population), len(population), 
                                      replace=True, p=probabilities)
    
    return [population[i] for i in selected_indices]


def rank_selection(population: List[Any], 
                  objectives: np.ndarray, 
                  pressure: float = 1.5) -> List[Any]:
    """
    Rank-based selection.
    
    Args:
        population: Current population
        objectives: Objective values
        pressure: Selection pressure (1.0 = no pressure, 2.0 = high pressure)
        
    Returns:
        Selected individuals
    """
    # Rank individuals (lower rank = better)
    ranks = np.argsort(np.argsort(objectives.flatten()))
    
    # Calculate selection probabilities based on ranks
    probabilities = (2 - pressure) / len(population) + 2 * (pressure - 1) * (len(population) - ranks) / (len(population) * (len(population) - 1))
    
    # Select individuals
    selected_indices = np.random.choice(len(population), len(population), 
                                      replace=True, p=probabilities)
    
    return [population[i] for i in selected_indices]
