"""
Evaluation metrics for recommender systems and multi-objective optimization.

This module provides various metrics for evaluating the performance of
recommender systems and evolutionary algorithms.
"""

import numpy as np
from typing import List, Dict, Any, Optional


def hypervolume(objectives: np.ndarray, reference_point: Optional[np.ndarray] = None) -> float:
    """
    Calculate hypervolume indicator.
    
    Args:
        objectives: Objective values (n_individuals, n_objectives)
        reference_point: Reference point for hypervolume calculation
        
    Returns:
        Hypervolume value
    """
    if reference_point is None:
        reference_point = np.max(objectives, axis=0) + 1.0
    
    if objectives.shape[1] == 2:
        return _hypervolume_2d(objectives, reference_point)
    elif objectives.shape[1] == 3:
        return _hypervolume_3d(objectives, reference_point)
    else:
        return _hypervolume_approximation(objectives, reference_point)


def _hypervolume_2d(objectives: np.ndarray, reference_point: np.ndarray) -> float:
    """Calculate 2D hypervolume."""
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


def diversity_metrics(recommendations: List[str], data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate diversity metrics for recommendations.
    
    Args:
        recommendations: List of recommended items
        data: Data containing category information
        
    Returns:
        Dictionary of diversity metrics
    """
    metrics = {}
    
    # Category diversity
    categories = []
    for item in recommendations:
        try:
            idx = data["candidate_set"].index(item)
            categories.extend(data["category_list"][idx])
        except:
            pass
    
    unique_categories = len(set(categories))
    total_categories = len(categories) + 1e-10
    
    metrics['category_diversity'] = unique_categories / total_categories
    metrics['intra_list_diversity'] = 1 - metrics['category_diversity']  # Lower is better for optimization
    
    # Item diversity (based on item similarity if available)
    if 'item_features' in data:
        item_features = data['item_features']
        similarities = []
        
        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                try:
                    idx1 = data["candidate_set"].index(recommendations[i])
                    idx2 = data["candidate_set"].index(recommendations[j])
                    
                    feat1 = item_features[idx1]
                    feat2 = item_features[idx2]
                    
                    # Calculate cosine similarity
                    similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
                    similarities.append(similarity)
                except:
                    pass
        
        if similarities:
            metrics['average_similarity'] = np.mean(similarities)
            metrics['item_diversity'] = 1 - metrics['average_similarity']
        else:
            metrics['average_similarity'] = 0.0
            metrics['item_diversity'] = 1.0
    
    return metrics


def fairness_metrics(recommendations: List[str], data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate fairness metrics for recommendations.
    
    Args:
        recommendations: List of recommended items
        data: Data containing popularity and group information
        
    Returns:
        Dictionary of fairness metrics
    """
    metrics = {}
    
    # Average Popularity of Top items (APT)
    popularity_scores = []
    for item in recommendations:
        try:
            idx = data["candidate_set"].index(item)
            popularity_scores.append(data["popular_list"][idx])
        except:
            popularity_scores.append(0)
    
    if popularity_scores:
        metrics['apt'] = np.mean(popularity_scores)
        metrics['fairness'] = 1 - metrics['apt']  # Lower APT = higher fairness
    else:
        metrics['apt'] = 0.0
        metrics['fairness'] = 1.0
    
    # Group fairness (if group information is available)
    if 'group_list' in data:
        group_counts = {}
        for item in recommendations:
            try:
                idx = data["candidate_set"].index(item)
                group = data["group_list"][idx]
                group_counts[group] = group_counts.get(group, 0) + 1
            except:
                pass
        
        if group_counts:
            # Calculate Gini coefficient for group distribution
            group_proportions = list(group_counts.values())
            group_proportions = np.array(group_proportions) / sum(group_proportions)
            
            # Gini coefficient calculation
            n = len(group_proportions)
            gini = 0.0
            for i in range(n):
                for j in range(n):
                    gini += abs(group_proportions[i] - group_proportions[j])
            gini = gini / (2 * n * np.sum(group_proportions))
            
            metrics['gini_coefficient'] = gini
            metrics['group_fairness'] = 1 - gini  # Lower Gini = higher fairness
        else:
            metrics['gini_coefficient'] = 0.0
            metrics['group_fairness'] = 1.0
    
    return metrics


def accuracy_metrics(recommendations: List[str], data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate accuracy metrics for recommendations.
    
    Args:
        recommendations: List of recommended items
        data: Data containing target information
        
    Returns:
        Dictionary of accuracy metrics
    """
    metrics = {}
    
    # Position-based accuracy
    target = data.get('target', '')
    if target in recommendations:
        position = recommendations.index(target)
        metrics['position'] = position
        metrics['reciprocal_rank'] = 1.0 / (position + 1)
        metrics['accuracy'] = position / (len(recommendations) + 1e-10)
    else:
        metrics['position'] = len(recommendations)
        metrics['reciprocal_rank'] = 0.0
        metrics['accuracy'] = 1.0  # Maximum penalty for not finding target
    
    # Hit rate
    metrics['hit_rate'] = 1.0 if target in recommendations else 0.0
    
    # Precision@K
    k_values = [5, 10, 20]
    for k in k_values:
        if target in recommendations[:k]:
            metrics[f'precision@{k}'] = 1.0 / k
        else:
            metrics[f'precision@{k}'] = 0.0
    
    return metrics


def coverage_metrics(recommendations: List[str], data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate coverage metrics for recommendations.
    
    Args:
        recommendations: List of recommended items
        data: Data containing candidate set information
        
    Returns:
        Dictionary of coverage metrics
    """
    metrics = {}
    
    candidate_set = data.get('candidate_set', [])
    if candidate_set:
        # Item coverage
        unique_recommended = len(set(recommendations))
        total_candidates = len(candidate_set)
        metrics['item_coverage'] = unique_recommended / total_candidates
        
        # Category coverage
        if 'category_list' in data:
            recommended_categories = set()
            for item in recommendations:
                try:
                    idx = candidate_set.index(item)
                    recommended_categories.update(data['category_list'][idx])
                except:
                    pass
            
            all_categories = set()
            for categories in data['category_list']:
                all_categories.update(categories)
            
            if all_categories:
                metrics['category_coverage'] = len(recommended_categories) / len(all_categories)
            else:
                metrics['category_coverage'] = 0.0
    
    return metrics


def novelty_metrics(recommendations: List[str], data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate novelty metrics for recommendations.
    
    Args:
        recommendations: List of recommended items
        data: Data containing popularity information
        
    Returns:
        Dictionary of novelty metrics
    """
    metrics = {}
    
    if 'popular_list' in data:
        popularity_scores = []
        for item in recommendations:
            try:
                idx = data["candidate_set"].index(item)
                popularity_scores.append(data["popular_list"][idx])
            except:
                popularity_scores.append(0)
        
        if popularity_scores:
            # Average popularity (lower is more novel)
            metrics['average_popularity'] = np.mean(popularity_scores)
            metrics['novelty'] = 1 - metrics['average_popularity']
            
            # Popularity distribution
            metrics['popularity_std'] = np.std(popularity_scores)
        else:
            metrics['average_popularity'] = 0.0
            metrics['novelty'] = 1.0
            metrics['popularity_std'] = 0.0
    
    return metrics


def comprehensive_evaluation(recommendations: List[str], data: Dict[str, Any]) -> Dict[str, float]:
    """
    Perform comprehensive evaluation of recommendations.
    
    Args:
        recommendations: List of recommended items
        data: Data containing all necessary information
        
    Returns:
        Dictionary of all evaluation metrics
    """
    metrics = {}
    
    # Combine all metric types
    metrics.update(accuracy_metrics(recommendations, data))
    metrics.update(diversity_metrics(recommendations, data))
    metrics.update(fairness_metrics(recommendations, data))
    metrics.update(coverage_metrics(recommendations, data))
    metrics.update(novelty_metrics(recommendations, data))
    
    return metrics


def calculate_aggregate_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate aggregate metrics across multiple evaluations.
    
    Args:
        all_metrics: List of metric dictionaries from multiple evaluations
        
    Returns:
        Dictionary of aggregate metrics
    """
    if not all_metrics:
        return {}
    
    aggregate = {}
    
    # Get all metric names
    all_keys = set()
    for metrics in all_metrics:
        all_keys.update(metrics.keys())
    
    # Calculate statistics for each metric
    for key in all_keys:
        values = [metrics.get(key, 0.0) for metrics in all_metrics]
        values = [v for v in values if not np.isnan(v)]  # Remove NaN values
        
        if values:
            aggregate[f'{key}_mean'] = np.mean(values)
            aggregate[f'{key}_std'] = np.std(values)
            aggregate[f'{key}_min'] = np.min(values)
            aggregate[f'{key}_max'] = np.max(values)
            aggregate[f'{key}_median'] = np.median(values)
    
    return aggregate
