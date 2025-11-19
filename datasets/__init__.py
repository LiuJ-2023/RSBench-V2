"""
Dataset handling for RSBench-V3 library.

This module provides utilities for loading and managing datasets
used in recommender system benchmarks.
"""

from .loader import (
    DatasetLoader, MovieLensDataset, GameDataset, BundleDataset,
    create_dataset_loader, load_sample_data, validate_dataset_structure, preprocess_dataset
)

__all__ = [
    'DatasetLoader', 'MovieLensDataset', 'GameDataset', 'BundleDataset',
    'create_dataset_loader', 'load_sample_data', 'validate_dataset_structure', 'preprocess_dataset'
]
