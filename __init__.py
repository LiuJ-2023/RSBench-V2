"""
RSBench-V3: A Library for Evolutionary Computation with LLM-based Recommender Systems

This library provides a comprehensive framework for applying evolutionary algorithms
to optimize Large Language Model (LLM) based recommender systems. It includes:

- Multiple evolutionary algorithms (NSGA-II, MOEA/D, IBEA) with LLM integration
- Benchmark problems for recommender systems (accuracy, diversity, fairness)
- Async LLM integration with token tracking and cost estimation
- Modular design for easy extension and customization

Author: Based on RSBench_v2 by Jiao Liu et al.
"""

__version__ = "1.0.0"
__author__ = "RSBench-V3 Team"

# Core imports
from .core.problems import BaseProblem, AccDivProblem, AccFairProblem, AccDivFairProblem
from .core.algorithms import BaseAlgorithm, NSGA2LLM, MOEADLLM, IBEALLM
from .core.operators import LLMOperator, TokenCounterCallback
from .core.evaluation import Evaluator, Metrics

# Utility imports
from utils.selection import environment_selection, IBEA_selection
from utils.dominance import nondominated_sort, crowding_distance
from utils.metrics import hypervolume, diversity_metrics, fairness_metrics

# Dataset imports
from datasets import DatasetLoader, MovieLensDataset, GameDataset, BundleDataset

__all__ = [
    # Core classes
    'BaseProblem', 'AccDivProblem', 'AccFairProblem', 'AccDivFairProblem',
    'BaseAlgorithm', 'NSGA2LLM', 'MOEADLLM', 'IBEALLM',
    'LLMOperator', 'TokenCounterCallback',
    'Evaluator', 'Metrics',
    
    # Utilities
    'environment_selection', 'IBEA_selection',
    'nondominated_sort', 'crowding_distance',
    'hypervolume', 'diversity_metrics', 'fairness_metrics',
    
    # Datasets
    'DatasetLoader', 'MovieLensDataset', 'GameDataset', 'BundleDataset',
]
