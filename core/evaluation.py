"""
Evaluation and Metrics for LLM-based Recommender Systems.

This module provides evaluation utilities and metrics for assessing
the performance of evolutionary algorithms on recommender system problems.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod
import pandas as pd


class Metrics:
    """
    Collection of evaluation metrics for multi-objective optimization.
    
    This class provides various metrics for evaluating the performance
    of evolutionary algorithms on multi-objective problems.
    """
    
    @staticmethod
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
        
        # Simple hypervolume calculation for 2D and 3D
        if objectives.shape[1] == 2:
            return Metrics._hypervolume_2d(objectives, reference_point)
        elif objectives.shape[1] == 3:
            return Metrics._hypervolume_3d(objectives, reference_point)
        else:
            # For higher dimensions, use approximation
            return Metrics._hypervolume_approximation(objectives, reference_point)
    
    @staticmethod
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
    
    @staticmethod
    def _hypervolume_3d(objectives: np.ndarray, reference_point: np.ndarray) -> float:
        """Calculate 3D hypervolume (simplified)."""
        # This is a simplified 3D hypervolume calculation
        # For production use, consider using specialized libraries like pymoo
        volume = 0.0
        
        for i in range(len(objectives)):
            if np.all(objectives[i] < reference_point):
                # Calculate volume contribution
                contribution = np.prod(reference_point - objectives[i])
                volume += contribution
        
        return volume
    
    @staticmethod
    def _hypervolume_approximation(objectives: np.ndarray, reference_point: np.ndarray) -> float:
        """Approximate hypervolume for higher dimensions."""
        # Monte Carlo approximation
        n_samples = 10000
        n_dominated = 0
        
        for _ in range(n_samples):
            # Generate random point in hypercube
            random_point = np.random.uniform(
                np.min(objectives, axis=0), 
                reference_point
            )
            
            # Check if dominated by any solution
            if np.any(np.all(objectives <= random_point, axis=1)):
                n_dominated += 1
        
        # Calculate hypervolume
        total_volume = np.prod(reference_point - np.min(objectives, axis=0))
        return total_volume * (n_dominated / n_samples)
    
    @staticmethod
    def igd(objectives: np.ndarray, reference_front: np.ndarray) -> float:
        """
        Calculate Inverted Generational Distance (IGD).
        
        Args:
            objectives: Objective values of current population
            reference_front: Reference Pareto front
            
        Returns:
            IGD value
        """
        if len(objectives) == 0 or len(reference_front) == 0:
            return float('inf')
        
        distances = []
        for ref_point in reference_front:
            min_dist = np.min(np.linalg.norm(objectives - ref_point, axis=1))
            distances.append(min_dist)
        
        return np.mean(distances)
    
    @staticmethod
    def gd(objectives: np.ndarray, reference_front: np.ndarray) -> float:
        """
        Calculate Generational Distance (GD).
        
        Args:
            objectives: Objective values of current population
            reference_front: Reference Pareto front
            
        Returns:
            GD value
        """
        if len(objectives) == 0 or len(reference_front) == 0:
            return float('inf')
        
        distances = []
        for obj_point in objectives:
            min_dist = np.min(np.linalg.norm(reference_front - obj_point, axis=1))
            distances.append(min_dist)
        
        return np.mean(distances)
    
    @staticmethod
    def spread(objectives: np.ndarray) -> float:
        """
        Calculate spread (diversity) metric.
        
        Args:
            objectives: Objective values
            
        Returns:
            Spread value
        """
        if len(objectives) <= 1:
            return 0.0
        
        # Calculate distances between consecutive points
        sorted_indices = np.argsort(objectives[:, 0])
        sorted_obj = objectives[sorted_indices]
        
        distances = []
        for i in range(len(sorted_obj) - 1):
            dist = np.linalg.norm(sorted_obj[i+1] - sorted_obj[i])
            distances.append(dist)
        
        if len(distances) == 0:
            return 0.0
        
        mean_distance = np.mean(distances)
        if mean_distance == 0:
            return 0.0
        
        # Calculate spread
        spread_value = np.sqrt(np.sum((np.array(distances) - mean_distance) ** 2)) / mean_distance
        return spread_value
    
    @staticmethod
    def convergence_metric(objectives: np.ndarray, reference_point: np.ndarray) -> float:
        """
        Calculate convergence metric (distance to reference point).
        
        Args:
            objectives: Objective values
            reference_point: Reference point (usually ideal point)
            
        Returns:
            Average distance to reference point
        """
        if len(objectives) == 0:
            return float('inf')
        
        distances = np.linalg.norm(objectives - reference_point, axis=1)
        return np.mean(distances)


class Evaluator:
    """
    Comprehensive evaluator for evolutionary algorithms.
    
    This class provides methods for evaluating algorithm performance
    across multiple runs and comparing different algorithms.
    """
    
    def __init__(self, reference_front: Optional[np.ndarray] = None):
        """
        Initialize the evaluator.
        
        Args:
            reference_front: Reference Pareto front for comparison
        """
        self.reference_front = reference_front
        self.results = {}
    
    def evaluate_run(self, 
                    objectives: np.ndarray, 
                    run_id: str = "run_0",
                    algorithm_name: str = "algorithm") -> Dict[str, float]:
        """
        Evaluate a single run of an algorithm.
        
        Args:
            objectives: Objective values from the run
            run_id: Identifier for this run
            algorithm_name: Name of the algorithm
            
        Returns:
            Dictionary of metric values
        """
        metrics = {}
        
        # Basic metrics
        metrics['hypervolume'] = Metrics.hypervolume(objectives)
        metrics['spread'] = Metrics.spread(objectives)
        
        # Reference point for convergence
        reference_point = np.zeros(objectives.shape[1])
        metrics['convergence'] = Metrics.convergence_metric(objectives, reference_point)
        
        # Reference front metrics if available
        if self.reference_front is not None:
            metrics['igd'] = Metrics.igd(objectives, self.reference_front)
            metrics['gd'] = Metrics.gd(objectives, self.reference_front)
        
        # Store results
        if algorithm_name not in self.results:
            self.results[algorithm_name] = {}
        self.results[algorithm_name][run_id] = metrics
        
        return metrics
    
    def compare_algorithms(self, 
                          algorithm_results: Dict[str, List[np.ndarray]],
                          metric_names: List[str] = None) -> pd.DataFrame:
        """
        Compare multiple algorithms across multiple runs.
        
        Args:
            algorithm_results: Dictionary mapping algorithm names to lists of objective arrays
            metric_names: List of metrics to compute
            
        Returns:
            DataFrame with comparison results
        """
        if metric_names is None:
            metric_names = ['hypervolume', 'spread', 'convergence']
        
        comparison_data = []
        
        for alg_name, runs in algorithm_results.items():
            for run_idx, objectives in enumerate(runs):
                metrics = self.evaluate_run(objectives, f"run_{run_idx}", alg_name)
                
                row = {'algorithm': alg_name, 'run': run_idx}
                for metric in metric_names:
                    row[metric] = metrics.get(metric, np.nan)
                
                comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def plot_comparison(self, 
                       comparison_df: pd.DataFrame,
                       metric: str = 'hypervolume',
                       save_path: Optional[str] = None):
        """
        Plot comparison of algorithms for a specific metric.
        
        Args:
            comparison_df: DataFrame from compare_algorithms
            metric: Metric to plot
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Box plot
        sns.boxplot(data=comparison_df, x='algorithm', y=metric)
        plt.title(f'Comparison of Algorithms - {metric.upper()}')
        plt.xlabel('Algorithm')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_pareto_fronts(self, 
                          algorithm_results: Dict[str, List[np.ndarray]],
                          run_idx: int = 0,
                          save_path: Optional[str] = None):
        """
        Plot Pareto fronts for different algorithms.
        
        Args:
            algorithm_results: Dictionary mapping algorithm names to lists of objective arrays
            run_idx: Index of the run to plot
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(algorithm_results)))
        
        for i, (alg_name, runs) in enumerate(algorithm_results.items()):
            if run_idx < len(runs):
                objectives = runs[run_idx]
                
                if objectives.shape[1] == 2:
                    plt.scatter(objectives[:, 0], objectives[:, 1], 
                              label=alg_name, alpha=0.7, color=colors[i])
                elif objectives.shape[1] == 3:
                    ax = plt.axes(projection='3d')
                    ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2],
                              label=alg_name, alpha=0.7, color=colors[i])
        
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        if objectives.shape[1] == 3:
            ax.set_zlabel('Objective 3')
        plt.title('Pareto Fronts Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self, 
                       comparison_df: pd.DataFrame,
                       save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            comparison_df: DataFrame from compare_algorithms
            save_path: Optional path to save the report
            
        Returns:
            Report string
        """
        report = []
        report.append("=" * 60)
        report.append("EVOLUTIONARY ALGORITHM EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS:")
        report.append("-" * 30)
        
        for metric in ['hypervolume', 'spread', 'convergence']:
            if metric in comparison_df.columns:
                report.append(f"\n{metric.upper()}:")
                summary = comparison_df.groupby('algorithm')[metric].agg(['mean', 'std', 'min', 'max'])
                for alg in summary.index:
                    report.append(f"  {alg}:")
                    report.append(f"    Mean: {summary.loc[alg, 'mean']:.4f}")
                    report.append(f"    Std:  {summary.loc[alg, 'std']:.4f}")
                    report.append(f"    Min:  {summary.loc[alg, 'min']:.4f}")
                    report.append(f"    Max:  {summary.loc[alg, 'max']:.4f}")
        
        # Best algorithm for each metric
        report.append("\nBEST ALGORITHMS:")
        report.append("-" * 20)
        
        for metric in ['hypervolume', 'spread', 'convergence']:
            if metric in comparison_df.columns:
                if metric == 'hypervolume':
                    # Higher is better
                    best_alg = comparison_df.groupby('algorithm')[metric].mean().idxmax()
                else:
                    # Lower is better
                    best_alg = comparison_df.groupby('algorithm')[metric].mean().idxmin()
                
                best_value = comparison_df.groupby('algorithm')[metric].mean().loc[best_alg]
                report.append(f"{metric.upper()}: {best_alg} ({best_value:.4f})")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text
