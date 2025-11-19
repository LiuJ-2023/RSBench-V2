"""
Plotting utilities for RSBench evolution process visualization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Optional, Tuple
import time
from datetime import datetime


class EvolutionPlotter:
    """Plotter for tracking evolution process in objective space."""
    
    def __init__(self, 
                 save_dir: str = "Results",
                 objectives_names: Optional[List[str]] = None,
                 algorithm_name: str = "NSGA-II"):
        """
        Initialize the evolution plotter.
        
        Args:
            save_dir: Directory to save plots
            objectives_names: Names of objectives (e.g., ['Accuracy', 'Diversity'])
            algorithm_name: Name of the algorithm for plot titles
        """
        self.save_dir = save_dir
        self.algorithm_name = algorithm_name
        self.objectives_names = objectives_names or ['Objective 1', 'Objective 2']
        self.generation_data = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('default')
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """Set up matplotlib plotting style."""
        plt.rcParams['figure.figsize'] = (10, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.dpi'] = 100
    
    def record_generation(self, 
                         generation: int,
                         population: List[str],
                         objectives: np.ndarray,
                         pareto_front: Optional[np.ndarray] = None,
                         metrics: Optional[Dict] = None):
        """
        Record data for a generation.
        
        Args:
            generation: Generation number
            population: Population of solutions
            objectives: Objective values for the population
            pareto_front: Pareto front solutions (optional)
            metrics: Additional metrics (optional)
        """
        data = {
            'generation': generation,
            'population': population,
            'objectives': objectives.copy(),
            'pareto_front': pareto_front.copy() if pareto_front is not None else None,
            'metrics': metrics or {},
            'timestamp': time.time()
        }
        self.generation_data.append(data)
    
    def plot_objective_space(self, 
                           generation: Optional[int] = None,
                           show_pareto: bool = True,
                           show_evolution: bool = True,
                           save_plot: bool = True) -> str:
        """
        Plot the objective space for a specific generation or all generations.
        
        Args:
            generation: Specific generation to plot (None for all)
            show_pareto: Whether to highlight Pareto front
            show_evolution: Whether to show evolution trajectory
            save_plot: Whether to save the plot
            
        Returns:
            Path to saved plot file
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if generation is not None:
            # Plot specific generation
            self._plot_single_generation(ax, generation, show_pareto)
            title = f"{self.algorithm_name} - Generation {generation}"
        else:
            # Plot all generations with evolution
            self._plot_evolution(ax, show_pareto, show_evolution)
            title = f"{self.algorithm_name} - Evolution Process"
        
        ax.set_xlabel(self.objectives_names[0])
        ax.set_ylabel(self.objectives_names[1])
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Save plot
        if save_plot:
            filename = self._generate_filename(generation)
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved: {filepath}")
        
        plt.tight_layout()
        return filepath if save_plot else ""
    
    def _plot_single_generation(self, ax, generation: int, show_pareto: bool):
        """Plot a single generation."""
        if generation >= len(self.generation_data):
            return
        
        data = self.generation_data[generation]
        objectives = data['objectives']
        
        # Plot all solutions
        ax.scatter(objectives[:, 0], objectives[:, 1], 
                  c='lightblue', s=50, alpha=0.7, 
                  label=f'Generation {generation} Population')
        
        # Highlight Pareto front if available
        if show_pareto and data['pareto_front'] is not None:
            pareto_obj = data['pareto_front']
            ax.scatter(pareto_obj[:, 0], pareto_obj[:, 1], 
                      c='red', s=100, alpha=0.8, marker='*',
                      label=f'Generation {generation} Pareto Front')
    
    def _plot_evolution(self, ax, show_pareto: bool, show_evolution: bool):
        """Plot evolution across all generations."""
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.generation_data)))
        
        for i, data in enumerate(self.generation_data):
            objectives = data['objectives']
            gen = data['generation']
            
            # Plot population
            ax.scatter(objectives[:, 0], objectives[:, 1], 
                      c=[colors[i]], s=30, alpha=0.6,
                      label=f'Gen {gen}' if i % max(1, len(self.generation_data)//5) == 0 else "")
            
            # Plot Pareto front
            if show_pareto and data['pareto_front'] is not None:
                pareto_obj = data['pareto_front']
                ax.scatter(pareto_obj[:, 0], pareto_obj[:, 1], 
                          c=[colors[i]], s=80, alpha=0.9, marker='*')
        
        # Show evolution trajectory for best solutions (disabled by user request)
        # if show_evolution and len(self.generation_data) > 1:
        #     self._plot_evolution_trajectory(ax)
    
    def _plot_evolution_trajectory(self, ax):
        """Plot evolution trajectory of best solutions."""
        # Find best solution in each generation (assuming minimization)
        best_solutions = []
        for data in self.generation_data:
            objectives = data['objectives']
            # Find solution with minimum sum of objectives
            best_idx = np.argmin(np.sum(objectives, axis=1))
            best_solutions.append(objectives[best_idx])
        
        best_solutions = np.array(best_solutions)
        ax.plot(best_solutions[:, 0], best_solutions[:, 1], 
               'k--', linewidth=2, alpha=0.8, label='Best Solution Trajectory')
    
    def _generate_filename(self, generation: Optional[int]) -> str:
        """Generate filename for saved plot."""
        if generation is not None:
            return f"{self.algorithm_name}_Evolution_Iteration_{generation}_{self.timestamp}.png"
        else:
            return f"{self.algorithm_name}_Evolution_{self.timestamp}.png"
    
    def plot_metrics_evolution(self, 
                              metrics: List[str] = None,
                              save_plot: bool = True) -> str:
        """
        Plot evolution of metrics over generations.
        
        Args:
            metrics: List of metric names to plot
            save_plot: Whether to save the plot
            
        Returns:
            Path to saved plot file
        """
        if not self.generation_data:
            return ""
        
        # Default metrics to plot
        if metrics is None:
            metrics = ['hypervolume', 'spread', 'convergence']
        
        # Extract metric data
        generations = [data['generation'] for data in self.generation_data]
        metric_data = {metric: [] for metric in metrics}
        
        for data in self.generation_data:
            for metric in metrics:
                value = data['metrics'].get(metric, 0)
                metric_data[metric].append(value)
        
        # Create subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            axes[i].plot(generations, metric_data[metric], 'o-', linewidth=2, markersize=6)
            axes[i].set_xlabel('Generation')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_title(f'{metric.capitalize()} Evolution')
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.algorithm_name} - Metrics Evolution')
        plt.tight_layout()
        
        # Save plot
        if save_plot:
            filename = f"{self.algorithm_name}_Metrics_{self.timestamp}.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"📊 Metrics plot saved: {filepath}")
        
        return filepath if save_plot else ""
    
    def create_summary_report(self) -> str:
        """Create a summary report of the evolution process."""
        if not self.generation_data:
            return ""
        
        report_path = os.path.join(self.save_dir, f"{self.algorithm_name}_Summary_{self.timestamp}.txt")
        
        with open(report_path, 'w') as f:
            f.write(f"{self.algorithm_name} Evolution Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Generations: {len(self.generation_data)}\n")
            f.write(f"Algorithm: {self.algorithm_name}\n")
            f.write(f"Objectives: {', '.join(self.objectives_names)}\n\n")
            
            # Generation details
            for data in self.generation_data:
                gen = data['generation']
                pop_size = len(data['population'])
                f.write(f"Generation {gen}:\n")
                f.write(f"  Population Size: {pop_size}\n")
                
                if data['metrics']:
                    f.write("  Metrics:\n")
                    for metric, value in data['metrics'].items():
                        f.write(f"    {metric}: {value:.4f}\n")
                f.write("\n")
        
        print(f"📋 Summary report saved: {report_path}")
        return report_path
    
    def save_final_results(self, 
                          final_population: List[str],
                          final_objectives: np.ndarray,
                          final_metrics: Dict) -> str:
        """Save final results to files."""
        results_dir = os.path.join(self.save_dir, f"Final_Results_{self.timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save population
        pop_file = os.path.join(results_dir, "final_population.txt")
        with open(pop_file, 'w') as f:
            for i, prompt in enumerate(final_population):
                f.write(f"Solution {i+1}:\n")
                f.write(f"Objectives: {final_objectives[i]}\n")
                f.write(f"Prompt: {prompt}\n\n")
        
        # Save objectives
        obj_file = os.path.join(results_dir, "final_objectives.npy")
        np.save(obj_file, final_objectives)
        
        # Save metrics
        metrics_file = os.path.join(results_dir, "final_metrics.txt")
        with open(metrics_file, 'w') as f:
            for metric, value in final_metrics.items():
                f.write(f"{metric}: {value}\n")
        
        print(f"💾 Final results saved to: {results_dir}")
        return results_dir


def create_objective_space_plot(objectives: np.ndarray,
                               pareto_front: Optional[np.ndarray] = None,
                               title: str = "Objective Space",
                               objectives_names: List[str] = None,
                               save_path: Optional[str] = None) -> str:
    """
    Create a simple objective space plot.
    
    Args:
        objectives: Objective values
        pareto_front: Pareto front solutions
        title: Plot title
        objectives_names: Names of objectives
        save_path: Path to save the plot
        
    Returns:
        Path to saved plot
    """
    if objectives_names is None:
        objectives_names = ['Objective 1', 'Objective 2']
    
    plt.figure(figsize=(10, 8))
    
    # Plot all solutions
    plt.scatter(objectives[:, 0], objectives[:, 1], 
               c='lightblue', s=50, alpha=0.7, label='Population')
    
    # Plot Pareto front if provided
    if pareto_front is not None:
        plt.scatter(pareto_front[:, 0], pareto_front[:, 1], 
                   c='red', s=100, alpha=0.8, marker='*', label='Pareto Front')
    
    plt.xlabel(objectives_names[0])
    plt.ylabel(objectives_names[1])
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved: {save_path}")
    
    plt.tight_layout()
    return save_path or ""
