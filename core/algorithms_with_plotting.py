"""
Evolutionary Algorithms with LLM Integration and Plotting.
"""

import numpy as np
import asyncio
import time
import copy
import pickle
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from .operators import LLMOperator, get_llm_operator
from utils.selection import environment_selection, IBEA_selection
from utils.dominance import nondominated_sort, crowding_distance
from utils.plotting import EvolutionPlotter


class BaseAlgorithm(ABC):
    """
    Abstract base class for evolutionary algorithms.
    
    This class defines the interface that all evolutionary algorithms
    must implement, including initialization, evolution, and selection.
    """
    
    def __init__(self, 
                 problem,
                 pop_size: int,
                 max_iter: int,
                 api_key: str,
                 llm_model: str = 'gpt',
                 operator_type: str = "standard",
                 **kwargs):
        """
        Initialize the base algorithm.
        
        Args:
            problem: Problem instance to optimize
            pop_size: Population size
            max_iter: Maximum number of iterations
            api_key: API key for LLM services
            llm_model: LLM model type ('gpt' or 'glm')
            operator_type: Type of LLM operator to use
            **kwargs: Additional algorithm-specific parameters
        """
        self.problem = problem
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.api_key = api_key
        self.llm_model = llm_model
        
        # Initialize LLM operator
        self.operator = get_llm_operator(
            operator_type=operator_type,
            llm_model=llm_model,
            api_key=api_key,
            token_stats=problem.token_stats
        )
        
        # Initialize history for tracking
        self.history = {}
        self.population = []
        self.objectives = np.array([])
    
    @abstractmethod
    async def run(self, save_path: Optional[str] = None) -> Tuple[List[str], np.ndarray]:
        """
        Run the evolutionary algorithm.
        
        Args:
            save_path: Optional path to save results
            
        Returns:
            Tuple of (final_population, final_objectives)
        """
        pass
    
    def save_results(self, save_path: str):
        """Save algorithm results to file."""
        results = {
            'population': self.population,
            'objectives': self.objectives,
            'history': self.history,
            'algorithm_params': {
                'pop_size': self.pop_size,
                'max_iter': self.max_iter,
                'llm_model': self.llm_model
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
    
    def load_results(self, load_path: str):
        """Load algorithm results from file."""
        with open(load_path, 'rb') as f:
            results = pickle.load(f)
        
        self.population = results['population']
        self.objectives = results['objectives']
        self.history = results['history']
    
    def _print_summary(self):
        """Print algorithm summary."""
        print(f"\n📊 ALGORITHM SUMMARY:")
        print(f"Algorithm: {getattr(self, 'algorithm_name', 'Unknown')}")
        print(f"Population Size: {self.pop_size}")
        print(f"Iterations: {self.max_iter}")
        print(f"Final Population Size: {len(self.population)}")
        print(f"Objectives: {self.objectives.shape[1] if len(self.objectives) > 0 else 0}")
        
        # Token usage summary
        if hasattr(self.problem, 'token_stats'):
            token_stats = self.problem.token_stats
            print(f"\n=== Token Usage Summary ===")
            print(f"Total tokens: {token_stats['total_tokens']}")
            print(f"Prompt tokens: {token_stats['prompt_tokens']}")
            print(f"Completion tokens: {token_stats['completion_tokens']}")
            print(f"Estimated cost: ${token_stats.get('estimated_cost', 0.0):.4f}")


class NSGA2LLM(BaseAlgorithm):
    """
    NSGA-II algorithm with LLM integration and plotting.
    
    This implementation combines the NSGA-II multi-objective evolutionary
    algorithm with LLM-based initialization and crossover operators,
    plus comprehensive plotting and visualization capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.algorithm_name = "NSGA2-LLM"
        
        # Initialize plotter
        self.plotter = None
        self.enable_plotting = kwargs.get('enable_plotting', True)
        if self.enable_plotting:
            self._setup_plotter()
    
    def _setup_plotter(self):
        """Setup the evolution plotter."""
        # Get objective names from problem
        if hasattr(self.problem, 'objectives_names'):
            objectives_names = self.problem.objectives_names
        else:
            # Default names based on problem type
            if hasattr(self.problem, '__class__'):
                class_name = self.problem.__class__.__name__
                if 'AccDiv' in class_name and 'Fair' in class_name:
                    objectives_names = ['Accuracy', 'Diversity', 'Fairness']
                elif 'AccDiv' in class_name:
                    objectives_names = ['Accuracy', 'Diversity']
                elif 'AccFair' in class_name:
                    objectives_names = ['Accuracy', 'Fairness']
                else:
                    objectives_names = ['Objective 1', 'Objective 2']
            else:
                objectives_names = ['Objective 1', 'Objective 2']
        
        self.plotter = EvolutionPlotter(
            save_dir="Results",
            objectives_names=objectives_names,
            algorithm_name=self.algorithm_name
        )
    
    async def run(self, save_path: Optional[str] = None) -> Tuple[List[str], np.ndarray]:
        """
        Run NSGA-II with LLM integration and plotting.
        
        Args:
            save_path: Optional path to save results
            
        Returns:
            Tuple of (final_population, final_objectives)
        """
        print(f'🚀 {self.algorithm_name} Algorithm Starting!')
        
        # Initialize population
        print('Initializing the Population...')
        start_time = time.time()
        
        example_prompt = self._get_example_prompt()
        self.population = self.operator.initialize(example_prompt, self.pop_size)
        
        # Evaluate initial population
        self.problem.sample_test_data()
        print('🔄 Evaluating initial population...')
        self.objectives = await self.problem.evaluate(self.population)
        
        init_time = time.time() - start_time
        print(f"✅ Initial evaluation completed in: {init_time/60:.2f} minutes")
        
        # Record initial state
        self.history[0] = {
            'population': copy.deepcopy(self.population),
            'objectives': copy.deepcopy(self.objectives),
            'init_time': init_time
        }
        
        # Record initial generation for plotting
        if self.plotter:
            pareto_front = self._get_pareto_front(self.objectives)
            self.plotter.record_generation(
                generation=0,
                population=self.population,
                objectives=self.objectives,
                pareto_front=pareto_front
            )
        
        # Evolution loop
        print('Evolution is starting!')
        for iteration in range(1, self.max_iter + 1):
            print(f'🔄 Generation {iteration}/{self.max_iter}')
            
            iter_start = time.time()
            
            # Generate offspring
            crossover_start = time.time()
            offspring = self.operator.crossover(self.population)
            crossover_time = time.time() - crossover_start
            print(f'⚡ Crossover completed in: {crossover_time:.2f} seconds')
            
            # Evaluate offspring
            self.problem.sample_test_data()
            eval_start = time.time()
            print(f'🔄 Evaluating offspring generation {iteration}...')
            offspring_objectives = await self.problem.evaluate(offspring)
            eval_time = time.time() - eval_start
            print(f'✅ Generation {iteration} evaluation completed in: {eval_time:.2f} seconds')
            
            # Environment selection
            selection_start = time.time()
            self.population, self.objectives = self._environment_selection(
                self.population, self.objectives, offspring, offspring_objectives
            )
            selection_time = time.time() - selection_start
            print(f'⚡ Environment selection completed in: {selection_time:.2f} seconds')
            
            # Record iteration
            total_iter_time = time.time() - iter_start
            self.history[iteration] = {
                'population': copy.deepcopy(self.population),
                'objectives': copy.deepcopy(self.objectives),
                'crossover_time': crossover_time,
                'eval_time': eval_time,
                'selection_time': selection_time,
                'total_time': total_iter_time
            }
            
            # Record generation for plotting
            if self.plotter:
                pareto_front = self._get_pareto_front(self.objectives)
                metrics = self._calculate_metrics(self.objectives)
                self.plotter.record_generation(
                    generation=iteration,
                    population=self.population,
                    objectives=self.objectives,
                    pareto_front=pareto_front,
                    metrics=metrics
                )
            
            print(f'✅ Accomplished iteration {iteration}')
            
            # Save intermediate results
            if save_path:
                self.save_results(save_path)
        
        print('🎯 Evolution has been finished!')
        
        # Generate final plots and save results
        if self.plotter:
            self._generate_final_plots()
            self._save_final_results()
        
        self._print_summary()
        
        return self.population, self.objectives
    
    def _environment_selection(self, 
                             population: List[str], 
                             objectives: np.ndarray,
                             offspring: List[str], 
                             offspring_objectives: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """Perform environment selection using NSGA-II."""
        # Combine population and offspring
        combined_pop = population + offspring
        combined_obj = np.vstack([objectives, offspring_objectives])
        
        # Apply NSGA-II selection
        selected_pop, selected_obj, _, _, _ = environment_selection(
            [combined_pop, combined_obj], self.pop_size
        )
        
        return selected_pop, selected_obj
    
    def _get_pareto_front(self, objectives: np.ndarray) -> np.ndarray:
        """Extract Pareto front from objectives."""
        try:
            from utils.dominance import nondominated_sort
            fronts = nondominated_sort(objectives)
            if fronts and len(fronts[0]) > 0:
                return objectives[fronts[0]]
            return objectives
        except:
            return objectives
    
    def _calculate_metrics(self, objectives: np.ndarray) -> Dict:
        """Calculate performance metrics."""
        try:
            from utils.metrics import hypervolume, spread, convergence
            metrics = {}
            
            # Calculate hypervolume
            try:
                metrics['hypervolume'] = hypervolume(objectives)
            except:
                metrics['hypervolume'] = 0.0
            
            # Calculate spread
            try:
                metrics['spread'] = spread(objectives)
            except:
                metrics['spread'] = 0.0
            
            # Calculate convergence (if we have reference point)
            try:
                metrics['convergence'] = convergence(objectives)
            except:
                metrics['convergence'] = 0.0
            
            return metrics
        except:
            return {}
    
    def _generate_final_plots(self):
        """Generate final plots."""
        if not self.plotter:
            return
        
        print("📊 Generating evolution plots...")
        
        # Plot objective space evolution
        self.plotter.plot_objective_space(
            generation=None,  # All generations
            show_pareto=True,
            show_evolution=True,
            save_plot=True
        )
        
        # Plot metrics evolution
        self.plotter.plot_metrics_evolution(save_plot=True)
        
        # Create summary report
        self.plotter.create_summary_report()
    
    def _save_final_results(self):
        """Save final results."""
        if not self.plotter:
            return
        
        # Calculate final metrics
        final_metrics = self._calculate_metrics(self.objectives)
        
        # Save final results
        self.plotter.save_final_results(
            final_population=self.population,
            final_objectives=self.objectives,
            final_metrics=final_metrics
        )
    
    def _get_example_prompt(self) -> str:
        """Get example prompt for initialization."""
        return """Based on the user's current session interactions, you need to answer the following subtasks step by step:
1. Discover combinations of items within the session, where the number of combinations can be one or more.
2. For each combination, you need to recommend the most suitable items from the candidate set.
3. The recommendation should be based on the user's preferences and the context of the session.
4. You should provide a ranked list of recommendations for each combination.
5. The recommendations should be diverse and cover different aspects of the user's interests.

Please provide your recommendations in a clear and structured format."""
