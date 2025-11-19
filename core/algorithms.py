"""
Evolutionary Algorithms with LLM Integration.

This module provides implementations of various evolutionary algorithms
enhanced with LLM-based operators for prompt optimization.
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
        
        # Algorithm state
        self.population = None
        self.objectives = None
        self.iteration = 0
        self.history = {}
    
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
            'token_stats': self.problem.token_stats,
            'algorithm_params': {
                'pop_size': self.pop_size,
                'max_iter': self.max_iter,
                'llm_model': self.llm_model
            }
        }
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)


class NSGA2LLM(BaseAlgorithm):
    """
    NSGA-II algorithm with LLM integration.
    
    This implementation combines the NSGA-II multi-objective evolutionary
    algorithm with LLM-based initialization and crossover operators.
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
        Run NSGA-II with LLM integration.
        
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
            
            print(f'✅ Accomplished iteration {iteration}')
            
            # Save intermediate results
            if save_path:
                self.save_results(save_path)
        
        print('🎯 Evolution has been finished!')
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
2. Based on the items within each combination, infer the user's interactive intent within each combination.
3. Select the intent from the inferred ones that best represents the user's current preferences.
4. Based on the selected intent, please rerank the 20 items in the candidate set according to the possibility of potential user interactions and show me your ranking results with item index.
Note that the order of all items in the candidate set must be given, and the items for ranking must be within the candidate set."""
    
    def _print_summary(self):
        """Print algorithm summary."""
        print('\n📊 ALGORITHM SUMMARY:')
        print(f'Algorithm: {self.algorithm_name}')
        print(f'Population Size: {self.pop_size}')
        print(f'Iterations: {self.max_iter}')
        print(f'Final Population Size: {len(self.population)}')
        print(f'Objectives: {self.objectives.shape[1]}')
        
        # Token usage summary
        token_stats = self.problem.token_stats
        print(f"\n=== Token Usage Summary ===")
        print(f"Total tokens: {token_stats['total_tokens']}")
        print(f"Prompt tokens: {token_stats['prompt_tokens']}")
        print(f"Completion tokens: {token_stats['completion_tokens']}")
        print(f"Estimated cost: ${token_stats['total_cost']:.4f}")


class MOEADLLM(BaseAlgorithm):
    """
    MOEA/D algorithm with LLM integration.
    
    This implementation combines the MOEA/D decomposition-based
    multi-objective evolutionary algorithm with LLM operators.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.algorithm_name = "MOEA/D-LLM"
        
        # MOEA/D specific parameters
        self.num_neighbors = kwargs.get('num_neighbors', 10)
        self.decomposition_method = kwargs.get('decomposition_method', 'tchebycheff')
        
        # Initialize weight vectors
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weight vectors for MOEA/D."""
        obj_num = self.problem.get_objective_count()
        
        # Generate uniform weight vectors
        if obj_num == 2:
            self.weights = np.array([[i/self.pop_size, 1-i/self.pop_size] 
                                   for i in range(self.pop_size)])
        elif obj_num == 3:
            # Generate 3D weights using simplex lattice
            self.weights = self._generate_3d_weights()
        else:
            # Generate random weights for higher dimensions
            self.weights = np.random.dirichlet([1] * obj_num, self.pop_size)
        
        # Calculate neighborhood structure
        self._calculate_neighborhoods()
    
    def _generate_3d_weights(self) -> np.ndarray:
        """Generate 3D weight vectors using simplex lattice."""
        # Simple approach for 3D weights
        weights = []
        for i in range(self.pop_size):
            w1 = i / self.pop_size
            w2 = (self.pop_size - i) / self.pop_size
            w3 = 1 - w1 - w2
            if w3 >= 0:
                weights.append([w1, w2, w3])
        
        # Fill remaining weights randomly if needed
        while len(weights) < self.pop_size:
            w = np.random.dirichlet([1, 1, 1])
            weights.append(w)
        
        return np.array(weights[:self.pop_size])
    
    def _calculate_neighborhoods(self):
        """Calculate neighborhood structure for MOEA/D."""
        # Calculate distances between weight vectors
        distances = np.zeros((self.pop_size, self.pop_size))
        for i in range(self.pop_size):
            for j in range(self.pop_size):
                distances[i, j] = np.linalg.norm(self.weights[i] - self.weights[j])
        
        # Find neighbors for each subproblem
        self.neighbors = []
        for i in range(self.pop_size):
            neighbor_indices = np.argsort(distances[i])[:self.num_neighbors]
            self.neighbors.append(neighbor_indices)
    
    async def run(self, save_path: Optional[str] = None) -> Tuple[List[str], np.ndarray]:
        """Run MOEA/D with LLM integration."""
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
        
        # Evolution loop
        print('Evolution is starting!')
        for iteration in range(1, self.max_iter + 1):
            print(f'🔄 Generation {iteration}/{self.max_iter}')
            
            iter_start = time.time()
            
            # MOEA/D evolution
            self.population, self.objectives = await self._moead_evolution()
            
            # Record iteration
            total_iter_time = time.time() - iter_start
            self.history[iteration] = {
                'population': copy.deepcopy(self.population),
                'objectives': copy.deepcopy(self.objectives),
                'total_time': total_iter_time
            }
            
            print(f'✅ Accomplished iteration {iteration}')
            
            # Save intermediate results
            if save_path:
                self.save_results(save_path)
        
        print('🎯 Evolution has been finished!')
        self._print_summary()
        
        return self.population, self.objectives
    
    async def _moead_evolution(self) -> Tuple[List[str], np.ndarray]:
        """Perform MOEA/D evolution step."""
        new_population = self.population.copy()
        new_objectives = self.objectives.copy()
        
        for i in range(self.pop_size):
            # Select parents from neighborhood
            neighbor_indices = self.neighbors[i]
            parent_indices = np.random.choice(neighbor_indices, 2, replace=False)
            
            parent1 = self.population[parent_indices[0]]
            parent2 = self.population[parent_indices[1]]
            
            # Generate offspring
            offspring = self.operator.crossover([parent1, parent2])
            if not offspring:
                continue
            
            # Evaluate offspring
            offspring_objectives = await self.problem.evaluate(offspring)
            
            # Update neighborhood
            for j in neighbor_indices:
                if self._is_better(offspring_objectives[0], new_objectives[j], self.weights[j]):
                    new_population[j] = offspring[0]
                    new_objectives[j] = offspring_objectives[0]
        
        return new_population, new_objectives
    
    def _is_better(self, obj1: np.ndarray, obj2: np.ndarray, weight: np.ndarray) -> bool:
        """Check if obj1 is better than obj2 for the given weight vector."""
        if self.decomposition_method == 'tchebycheff':
            return self._tchebycheff_value(obj1, weight) < self._tchebycheff_value(obj2, weight)
        else:
            # Default to weighted sum
            return np.sum(obj1 * weight) < np.sum(obj2 * weight)
    
    def _tchebycheff_value(self, obj: np.ndarray, weight: np.ndarray) -> float:
        """Calculate Tchebycheff scalarization value."""
        return np.max(obj * weight) + 0.05 * np.sum(obj * weight)
    
    def _get_example_prompt(self) -> str:
        """Get example prompt for initialization."""
        return """Based on the user's current session interactions, you need to answer the following subtasks step by step:
1. Discover combinations of items within the session, where the number of combinations can be one or more.
2. Based on the items within each combination, infer the user's interactive intent within each combination.
3. Select the intent from the inferred ones that best represents the user's current preferences.
4. Based on the selected intent, please rerank the 20 items in the candidate set according to the possibility of potential user interactions and show me your ranking results with item index.
Note that the order of all items in the candidate set must be given, and the items for ranking must be within the candidate set."""
    
    def _print_summary(self):
        """Print algorithm summary."""
        print('\n📊 ALGORITHM SUMMARY:')
        print(f'Algorithm: {self.algorithm_name}')
        print(f'Population Size: {self.pop_size}')
        print(f'Iterations: {self.max_iter}')
        print(f'Neighbors: {self.num_neighbors}')
        print(f'Decomposition: {self.decomposition_method}')
        
        # Token usage summary
        token_stats = self.problem.token_stats
        print(f"\n=== Token Usage Summary ===")
        print(f"Total tokens: {token_stats['total_tokens']}")
        print(f"Prompt tokens: {token_stats['prompt_tokens']}")
        print(f"Completion tokens: {token_stats['completion_tokens']}")
        print(f"Estimated cost: ${token_stats['total_cost']:.4f}")


class IBEALLM(BaseAlgorithm):
    """
    IBEA algorithm with LLM integration.
    
    This implementation combines the Indicator-Based Evolutionary Algorithm
    with LLM-based operators for prompt optimization.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.algorithm_name = "IBEA-LLM"
        
        # IBEA specific parameters
        self.kappa = kwargs.get('kappa', 0.05)
        self.indicator = kwargs.get('indicator', 'epsilon')
    
    async def run(self, save_path: Optional[str] = None) -> Tuple[List[str], np.ndarray]:
        """Run IBEA with LLM integration."""
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
            
            # IBEA selection
            selection_start = time.time()
            self.population, self.objectives = IBEA_selection(
                self.population, self.objectives, offspring, offspring_objectives, 
                self.pop_size, self.kappa
            )
            selection_time = time.time() - selection_start
            print(f'⚡ IBEA selection completed in: {selection_time:.2f} seconds')
            
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
            
            print(f'✅ Accomplished iteration {iteration}')
            
            # Save intermediate results
            if save_path:
                self.save_results(save_path)
        
        print('🎯 Evolution has been finished!')
        self._print_summary()
        
        return self.population, self.objectives
    
    def _get_example_prompt(self) -> str:
        """Get example prompt for initialization."""
        return """Based on the user's current session interactions, you need to answer the following subtasks step by step:
1. Discover combinations of items within the session, where the number of combinations can be one or more.
2. Based on the items within each combination, infer the user's interactive intent within each combination.
3. Select the intent from the inferred ones that best represents the user's current preferences.
4. Based on the selected intent, please rerank the 20 items in the candidate set according to the possibility of potential user interactions and show me your ranking results with item index.
Note that the order of all items in the candidate set must be given, and the items for ranking must be within the candidate set."""
    
    def _print_summary(self):
        """Print algorithm summary."""
        print('\n📊 ALGORITHM SUMMARY:')
        print(f'Algorithm: {self.algorithm_name}')
        print(f'Population Size: {self.pop_size}')
        print(f'Iterations: {self.max_iter}')
        print(f'Kappa: {self.kappa}')
        print(f'Indicator: {self.indicator}')
        
        # Token usage summary
        token_stats = self.problem.token_stats
        print(f"\n=== Token Usage Summary ===")
        print(f"Total tokens: {token_stats['total_tokens']}")
        print(f"Prompt tokens: {token_stats['prompt_tokens']}")
        print(f"Completion tokens: {token_stats['completion_tokens']}")
        print(f"Estimated cost: ${token_stats['total_cost']:.4f}")


def get_algorithm(algorithm_name: str, **kwargs) -> BaseAlgorithm:
    """
    Factory function to create evolutionary algorithms.
    
    Args:
        algorithm_name: Name of the algorithm ("NSGA2", "MOEAD", "IBEA")
        **kwargs: Additional parameters for the algorithm
        
    Returns:
        BaseAlgorithm instance
    """
    algorithm_map = {
        "NSGA2": NSGA2LLM,
        "MOEAD": MOEADLLM,
        "IBEA": IBEALLM
    }
    
    if algorithm_name.upper() not in algorithm_map:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    return algorithm_map[algorithm_name.upper()](**kwargs)
