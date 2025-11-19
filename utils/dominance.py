"""
Dominance and Pareto front calculations.

This module provides functions for determining dominance relationships
and calculating Pareto fronts in multi-objective optimization.
"""

import numpy as np
from typing import Tuple, List


def is_dominated(obj1: np.ndarray, obj2: np.ndarray) -> bool:
    """
    Check if obj1 is dominated by obj2.
    
    Args:
        obj1: First objective vector
        obj2: Second objective vector
        
    Returns:
        True if obj1 is dominated by obj2
    """
    return np.all(obj1 >= obj2) and np.any(obj1 > obj2)


def is_non_dominated(obj1: np.ndarray, obj2: np.ndarray) -> bool:
    """
    Check if obj1 and obj2 are non-dominated.
    
    Args:
        obj1: First objective vector
        obj2: Second objective vector
        
    Returns:
        True if obj1 and obj2 are non-dominated
    """
    return not is_dominated(obj1, obj2) and not is_dominated(obj2, obj1)


def get_pareto_front(objectives: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get Pareto front from a set of objective vectors.
    
    Args:
        objectives: Objective values (n_individuals, n_objectives)
        
    Returns:
        Tuple of (pareto_front_objectives, pareto_front_indices)
    """
    n_individuals = objectives.shape[0]
    pareto_indices = []
    
    for i in range(n_individuals):
        is_pareto = True
        for j in range(n_individuals):
            if i != j and is_dominated(objectives[i], objectives[j]):
                is_pareto = False
                break
        
        if is_pareto:
            pareto_indices.append(i)
    
    pareto_indices = np.array(pareto_indices)
    return objectives[pareto_indices], pareto_indices


def nondominated_sort(objectives: np.ndarray, n_sort: int) -> Tuple[np.ndarray, int]:
    """
    Non-dominated sorting algorithm.
    
    Args:
        objectives: Objective vectors (n_individuals, n_objectives)
        n_sort: Number of individuals to sort
        
    Returns:
        Tuple of (front_numbers, max_front_number)
    """
    n, m_obj = objectives.shape
    
    # Handle single objective case
    if m_obj == 1:
        sorted_indices = np.argsort(objectives.flatten())
        front_no = np.zeros(n)
        for i, idx in enumerate(sorted_indices):
            front_no[idx] = i + 1
        return front_no, int(np.max(front_no))
    
    # Multi-objective case
    front_no = np.inf * np.ones(n)
    max_front = 0
    
    while np.sum(front_no < np.inf) < min(n_sort, n):
        max_front += 1
        
        for i in range(n):
            if front_no[i] == np.inf:
                dominated = False
                
                # Check against solutions in current front
                for j in range(i, 0, -1):
                    if front_no[j - 1] == max_front:
                        # Check if solution i is dominated by solution j-1
                        m = 0
                        while m < m_obj and objectives[i, m] >= objectives[j - 1, m]:
                            m += 1
                        dominated = m == m_obj
                        
                        if dominated or m_obj == 2:
                            break
                
                if not dominated:
                    front_no[i] = max_front
    
    return front_no, max_front


def crowding_distance(objectives: np.ndarray, front_no: np.ndarray) -> np.ndarray:
    """
    Calculate crowding distance for each individual.
    
    Args:
        objectives: Objective vectors (n_individuals, n_objectives)
        front_no: Front numbers for each individual
        
    Returns:
        Crowding distances
    """
    n, M = objectives.shape
    crowd_dis = np.zeros(n)
    
    # Get unique front numbers
    fronts = np.unique(front_no)
    fronts = fronts[fronts != np.inf]
    
    for front in fronts:
        # Get individuals in this front
        front_indices = np.where(front_no == front)[0]
        
        if len(front_indices) <= 2:
            # If front has 2 or fewer individuals, set infinite distance
            crowd_dis[front_indices] = np.inf
            continue
        
        # Calculate crowding distance for each objective
        for m in range(M):
            # Sort by objective m
            sorted_indices = np.argsort(objectives[front_indices, m])
            sorted_front = front_indices[sorted_indices]
            
            # Set boundary points to infinite distance
            crowd_dis[sorted_front[0]] = np.inf
            crowd_dis[sorted_front[-1]] = np.inf
            
            # Calculate distance for intermediate points
            obj_range = objectives[sorted_front[-1], m] - objectives[sorted_front[0], m]
            if obj_range > 0:
                for i in range(1, len(sorted_front) - 1):
                    distance = (objectives[sorted_front[i + 1], m] - 
                              objectives[sorted_front[i - 1], m]) / obj_range
                    crowd_dis[sorted_front[i]] += distance
    
    return crowd_dis


def fast_non_dominated_sort(objectives: np.ndarray) -> List[List[int]]:
    """
    Fast non-dominated sorting algorithm.
    
    Args:
        objectives: Objective vectors (n_individuals, n_objectives)
        
    Returns:
        List of fronts, where each front is a list of individual indices
    """
    n = objectives.shape[0]
    
    # Initialize data structures
    S = [[] for _ in range(n)]  # Set of solutions dominated by solution i
    n_dominated = np.zeros(n)   # Number of solutions that dominate solution i
    rank = np.zeros(n)          # Rank of solution i
    front = [[]]                # List of fronts
    
    # Calculate dominance relationships
    for i in range(n):
        for j in range(n):
            if i != j:
                if is_dominated(objectives[i], objectives[j]):
                    S[i].append(j)
                elif is_dominated(objectives[j], objectives[i]):
                    n_dominated[i] += 1
        
        if n_dominated[i] == 0:
            rank[i] = 0
            front[0].append(i)
    
    # Build subsequent fronts
    front_idx = 0
    while front[front_idx]:
        Q = []
        for i in front[front_idx]:
            for j in S[i]:
                n_dominated[j] -= 1
                if n_dominated[j] == 0:
                    rank[j] = front_idx + 1
                    Q.append(j)
        front_idx += 1
        front.append(Q)
    
    return front[:-1]  # Remove empty last front


def calculate_hypervolume_contribution(objectives: np.ndarray, 
                                     reference_point: np.ndarray,
                                     individual_idx: int) -> float:
    """
    Calculate hypervolume contribution of a single individual.
    
    Args:
        objectives: Objective vectors (n_individuals, n_objectives)
        reference_point: Reference point for hypervolume calculation
        individual_idx: Index of the individual
        
    Returns:
        Hypervolume contribution
    """
    # Remove the individual
    remaining_objectives = np.delete(objectives, individual_idx, axis=0)
    
    # Calculate hypervolume without the individual
    hv_without = _calculate_hypervolume(remaining_objectives, reference_point)
    
    # Calculate hypervolume with all individuals
    hv_with = _calculate_hypervolume(objectives, reference_point)
    
    return hv_with - hv_without


def _calculate_hypervolume(objectives: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Calculate hypervolume (simplified version for 2D and 3D).
    
    Args:
        objectives: Objective vectors
        reference_point: Reference point
        
    Returns:
        Hypervolume value
    """
    if objectives.shape[1] == 2:
        return _hypervolume_2d(objectives, reference_point)
    elif objectives.shape[1] == 3:
        return _hypervolume_3d(objectives, reference_point)
    else:
        # For higher dimensions, use approximation
        return _hypervolume_approximation(objectives, reference_point)


def _hypervolume_2d(objectives: np.ndarray, reference_point: np.ndarray) -> float:
    """Calculate 2D hypervolume."""
    # Sort by first objective
    sorted_indices = np.argsort(objectives[:, 0])
    sorted_obj = objectives[sorted_indices]
    
    hv = 0.0
    prev_x = reference_point[0]
    
    for i in range(len(sorted_obj)):
        x = sorted_obj[i, 0]
        y = sorted_obj[i, 1]
        
        if y < reference_point[1]:
            hv += (prev_x - x) * (reference_point[1] - y)
            prev_x = x
    
    return hv


def _hypervolume_3d(objectives: np.ndarray, reference_point: np.ndarray) -> float:
    """Calculate 3D hypervolume (simplified)."""
    volume = 0.0
    
    for i in range(len(objectives)):
        if np.all(objectives[i] < reference_point):
            contribution = np.prod(reference_point - objectives[i])
            volume += contribution
    
    return volume


def _hypervolume_approximation(objectives: np.ndarray, reference_point: np.ndarray) -> float:
    """Approximate hypervolume for higher dimensions."""
    n_samples = 10000
    n_dominated = 0
    
    for _ in range(n_samples):
        random_point = np.random.uniform(
            np.min(objectives, axis=0), 
            reference_point
        )
        
        if np.any(np.all(objectives <= random_point, axis=1)):
            n_dominated += 1
    
    total_volume = np.prod(reference_point - np.min(objectives, axis=0))
    return total_volume * (n_dominated / n_samples)
