"""
Dataset loader for recommender system benchmarks.

This module provides utilities for loading and preprocessing datasets
used in LLM-based recommender system optimization.
"""

import json
import os
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod


class DatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.
    
    This class defines the interface for loading and preprocessing
    datasets for recommender system benchmarks.
    """
    
    def __init__(self, data_path: str, seed: int = 42):
        """
        Initialize the dataset loader.
        
        Args:
            data_path: Path to the dataset directory
            seed: Random seed for reproducibility
        """
        self.data_path = data_path
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Dataset attributes
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        
        # Load datasets
        self._load_datasets()
    
    @abstractmethod
    def _load_datasets(self):
        """Load and preprocess the datasets."""
        pass
    
    @abstractmethod
    def get_train_data(self, size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get training data.
        
        Args:
            size: Optional size limit for the data
            
        Returns:
            List of training samples
        """
        pass
    
    @abstractmethod
    def get_valid_data(self, size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get validation data.
        
        Args:
            size: Optional size limit for the data
            
        Returns:
            List of validation samples
        """
        pass
    
    @abstractmethod
    def get_test_data(self, size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get test data.
        
        Args:
            size: Optional size limit for the data
            
        Returns:
            List of test samples
        """
        pass
    
    def sample_data(self, data: List[Dict[str, Any]], size: int) -> List[Dict[str, Any]]:
        """
        Sample data with specified size.
        
        Args:
            data: Input data
            size: Desired sample size
            
        Returns:
            Sampled data
        """
        if size is None or size >= len(data):
            return data
        return random.sample(data, size)


class MovieLensDataset(DatasetLoader):
    """
    MovieLens dataset loader.
    
    This class handles loading and preprocessing of MovieLens datasets
    for recommender system benchmarks.
    """
    
    def __init__(self, data_path: str, seed: int = 42):
        """
        Initialize MovieLens dataset loader.
        
        Args:
            data_path: Path to MovieLens dataset directory
            seed: Random seed for reproducibility
        """
        super().__init__(data_path, seed)
    
    def _load_datasets(self):
        """Load MovieLens datasets."""
        # Load training data
        train_files = [f for f in os.listdir(self.data_path) if f.startswith('train_') and f.endswith('.json')]
        if train_files:
            # Use the first available training file
            train_file = train_files[0]
            with open(os.path.join(self.data_path, train_file), 'r') as f:
                self.train_data = json.load(f)
        
        # Load validation data
        valid_files = [f for f in os.listdir(self.data_path) if f.startswith('valid') and f.endswith('.json')]
        if valid_files:
            valid_file = valid_files[0]
            with open(os.path.join(self.data_path, valid_file), 'r') as f:
                self.valid_data = json.load(f)
        
        # Load test data
        test_files = [f for f in os.listdir(self.data_path) if f.startswith('test_') and f.endswith('.json')]
        if test_files:
            test_file = test_files[0]
            with open(os.path.join(self.data_path, test_file), 'r') as f:
                self.test_data = json.load(f)
    
    def get_train_data(self, size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get MovieLens training data."""
        if self.train_data is None:
            raise ValueError("Training data not loaded")
        return self.sample_data(self.train_data, size)
    
    def get_valid_data(self, size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get MovieLens validation data."""
        if self.valid_data is None:
            raise ValueError("Validation data not loaded")
        return self.sample_data(self.valid_data, size)
    
    def get_test_data(self, size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get MovieLens test data."""
        if self.test_data is None:
            raise ValueError("Test data not loaded")
        return self.sample_data(self.test_data, size)


class GameDataset(DatasetLoader):
    """
    Game dataset loader.
    
    This class handles loading and preprocessing of game datasets
    for recommender system benchmarks.
    """
    
    def __init__(self, data_path: str, seed: int = 42):
        """
        Initialize Game dataset loader.
        
        Args:
            data_path: Path to Game dataset directory
            seed: Random seed for reproducibility
        """
        super().__init__(data_path, seed)
    
    def _load_datasets(self):
        """Load Game datasets."""
        # Load training data
        train_files = [f for f in os.listdir(self.data_path) if f.startswith('train_') and f.endswith('.json')]
        if train_files:
            train_file = train_files[0]
            with open(os.path.join(self.data_path, train_file), 'r') as f:
                self.train_data = json.load(f)
        
        # Load validation data
        valid_files = [f for f in os.listdir(self.data_path) if f.startswith('valid') and f.endswith('.json')]
        if valid_files:
            valid_file = valid_files[0]
            with open(os.path.join(self.data_path, valid_file), 'r') as f:
                self.valid_data = json.load(f)
        
        # Load test data
        test_files = [f for f in os.listdir(self.data_path) if f.startswith('test_') and f.endswith('.json')]
        if test_files:
            test_file = test_files[0]
            with open(os.path.join(self.data_path, test_file), 'r') as f:
                self.test_data = json.load(f)
    
    def get_train_data(self, size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get Game training data."""
        if self.train_data is None:
            raise ValueError("Training data not loaded")
        return self.sample_data(self.train_data, size)
    
    def get_valid_data(self, size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get Game validation data."""
        if self.valid_data is None:
            raise ValueError("Validation data not loaded")
        return self.sample_data(self.valid_data, size)
    
    def get_test_data(self, size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get Game test data."""
        if self.test_data is None:
            raise ValueError("Test data not loaded")
        return self.sample_data(self.test_data, size)


class BundleDataset(DatasetLoader):
    """
    Bundle dataset loader.
    
    This class handles loading and preprocessing of bundle datasets
    for recommender system benchmarks.
    """
    
    def __init__(self, data_path: str, seed: int = 42):
        """
        Initialize Bundle dataset loader.
        
        Args:
            data_path: Path to Bundle dataset directory
            seed: Random seed for reproducibility
        """
        super().__init__(data_path, seed)
    
    def _load_datasets(self):
        """Load Bundle datasets."""
        # Load training data
        train_files = [f for f in os.listdir(self.data_path) if f.startswith('train_') and f.endswith('.json')]
        if train_files:
            train_file = train_files[0]
            with open(os.path.join(self.data_path, train_file), 'r') as f:
                self.train_data = json.load(f)
        
        # Load validation data
        valid_files = [f for f in os.listdir(self.data_path) if f.startswith('valid') and f.endswith('.json')]
        if valid_files:
            valid_file = valid_files[0]
            with open(os.path.join(self.data_path, valid_file), 'r') as f:
                self.valid_data = json.load(f)
        
        # Load test data
        test_files = [f for f in os.listdir(self.data_path) if f.startswith('test_') and f.endswith('.json')]
        if test_files:
            test_file = test_files[0]
            with open(os.path.join(self.data_path, test_file), 'r') as f:
                self.test_data = json.load(f)
    
    def get_train_data(self, size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get Bundle training data."""
        if self.train_data is None:
            raise ValueError("Training data not loaded")
        return self.sample_data(self.train_data, size)
    
    def get_valid_data(self, size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get Bundle validation data."""
        if self.valid_data is None:
            raise ValueError("Validation data not loaded")
        return self.sample_data(self.valid_data, size)
    
    def get_test_data(self, size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get Bundle test data."""
        if self.test_data is None:
            raise ValueError("Test data not loaded")
        return self.sample_data(self.test_data, size)


def create_dataset_loader(dataset_name: str, data_path: str, seed: int = 42) -> DatasetLoader:
    """
    Factory function to create dataset loaders.
    
    Args:
        dataset_name: Name of the dataset ("MovieLens", "Game", "Bundle")
        data_path: Path to the dataset directory
        seed: Random seed for reproducibility
        
    Returns:
        DatasetLoader instance
    """
    dataset_map = {
        "MovieLens": MovieLensDataset,
        "Game": GameDataset,
        "Bundle": BundleDataset
    }
    
    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset_map[dataset_name](data_path, seed)


def load_sample_data(data_path: str, dataset_name: str, data_type: str = "train", size: int = 100) -> List[Dict[str, Any]]:
    """
    Quick function to load sample data from a dataset.
    
    Args:
        data_path: Path to the dataset directory
        dataset_name: Name of the dataset
        data_type: Type of data to load ("train", "valid", "test")
        size: Number of samples to load
        
    Returns:
        List of data samples
    """
    loader = create_dataset_loader(dataset_name, data_path)
    
    if data_type == "train":
        return loader.get_train_data(size)
    elif data_type == "valid":
        return loader.get_valid_data(size)
    elif data_type == "test":
        return loader.get_test_data(size)
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def validate_dataset_structure(data: List[Dict[str, Any]]) -> bool:
    """
    Validate that dataset has the required structure.
    
    Args:
        data: Dataset to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not data:
        return False
    
    required_keys = ['input', 'target', 'candidate_set']
    sample = data[0]
    
    for key in required_keys:
        if key not in sample:
            print(f"Missing required key: {key}")
            return False
    
    # Check for optional keys that are commonly used
    optional_keys = ['category_list', 'popular_list', 'group_list']
    for key in optional_keys:
        if key not in sample:
            print(f"Warning: Optional key '{key}' not found in dataset")
    
    return True


def preprocess_dataset(data: List[Dict[str, Any]], 
                      max_candidates: int = 20,
                      max_categories: int = 10) -> List[Dict[str, Any]]:
    """
    Preprocess dataset to ensure consistent structure.
    
    Args:
        data: Raw dataset
        max_candidates: Maximum number of candidates to keep
        max_categories: Maximum number of categories per item
        
    Returns:
        Preprocessed dataset
    """
    processed_data = []
    
    for sample in data:
        processed_sample = sample.copy()
        
        # Limit candidate set size
        if len(processed_sample['candidate_set']) > max_candidates:
            processed_sample['candidate_set'] = processed_sample['candidate_set'][:max_candidates]
        
        # Limit category list size
        if 'category_list' in processed_sample:
            for i, categories in enumerate(processed_sample['category_list']):
                if len(categories) > max_categories:
                    processed_sample['category_list'][i] = categories[:max_categories]
        
        processed_data.append(processed_sample)
    
    return processed_data
