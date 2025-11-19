# NSGA-II Demo Examples

This directory contains key demo examples for running NSGA-II with LLM integration on RSBench problems.

## Available Examples

### 1. Basic Example (`basic_example.py`)
- **Purpose**: Simple demonstration of NSGA-II with LLM integration
- **Features**: 
  - Basic NSGA-II algorithm
  - MovieLens dataset
  - Accuracy vs Diversity optimization
  - Token usage tracking
- **Usage**: `python examples/basic_example.py`

### 2. NSGA-II with Plotting (`nsga2_with_plotting.py`)
- **Purpose**: Enhanced demo with comprehensive plotting and visualization
- **Features**:
  - NSGA-II algorithm with plotting integration
  - Evolution process visualization
  - Objective space plots
  - Performance metrics tracking
  - Automatic plot generation and saving
- **Usage**: `python examples/nsga2_with_plotting.py`

## Prerequisites

1. **API Keys**: Set up your API keys in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

2. **Datasets**: Ensure the dataset files are available in the `Dataset/` directory

3. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

1. **Set up environment**:
   ```bash
   # Create .env file with your API key
   echo "OPENAI_API_KEY=your_key_here" > .env
   ```

2. **Run basic example**:
   ```bash
   python examples/basic_example.py
   ```

3. **Run plotting demo**:
   ```bash
   python examples/nsga2_with_plotting.py
   ```

## Expected Output

### Basic Example
- Dataset loading progress
- Algorithm initialization
- Evolution progress with timing
- Final results and metrics
- Token usage summary

### Plotting Demo
- All features from basic example
- Evolution plots saved to `Results/` folder
- Performance metrics visualization
- Summary reports and final results

## Generated Files

The plotting demo creates the following files in the `Results/` directory:
- `NSGA2-LLM_Evolution_*.png` - Objective space evolution plots
- `NSGA2-LLM_Evolution_Iteration_*.png` - Individual generation plots
- `NSGA2-LLM_Metrics_*.png` - Performance metrics over time
- `NSGA2-LLM_Summary_*.txt` - Algorithm summary report
- `Final_Results_*/` - Complete results directory

## RSBench Problems

The library supports three main problem types:

### 1. Accuracy vs Diversity (AccDiv)
- **Objective 1**: Accuracy - How well recommendations match user preferences
- **Objective 2**: Diversity - How diverse the recommended items are
- **Use case**: Balancing recommendation quality with item variety

### 2. Accuracy vs Fairness (AccFair)
- **Objective 1**: Accuracy - How well recommendations match user preferences
- **Objective 2**: Fairness - How fair recommendations are across different groups
- **Use case**: Ensuring equitable recommendations

### 3. Accuracy vs Diversity vs Fairness (AccDivFair)
- **Objective 1**: Accuracy - How well recommendations match user preferences
- **Objective 2**: Diversity - How diverse the recommended items are
- **Objective 3**: Fairness - How fair recommendations are across different groups
- **Use case**: Multi-objective optimization for comprehensive recommendation quality

## NSGA-II Algorithm

The Non-dominated Sorting Genetic Algorithm II (NSGA-II) is used for multi-objective optimization:

- **Population-based**: Maintains a population of candidate solutions
- **Non-dominated sorting**: Ranks solutions based on Pareto dominance
- **Crowding distance**: Maintains diversity in the solution set
- **LLM integration**: Uses LLM-based crossover for prompt evolution

## Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```
   ❌ Please set OPENAI_API_KEY in your .env file
   ```
   **Solution**: Create a `.env` file with your API key

2. **Dataset Not Found**
   ```
   ⚠️ Dataset loading failed: [Errno 2] No such file or directory: 'Dataset/Movie'
   ```
   **Solution**: The demo will create sample data automatically, or adjust the dataset path

3. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'core'
   ```
   **Solution**: Run from the project root directory, not from the examples folder

4. **Memory Issues**
   ```
   MemoryError: Unable to allocate array
   ```
   **Solution**: Reduce population size, iterations, or batch size

### Performance Tips

1. **Start Small**: Begin with small parameters (POP_SIZE=8, MAX_ITER=3)
2. **Monitor Costs**: Check token usage and costs in the output
3. **Use Sample Data**: The demo creates sample data if datasets aren't available
4. **Adjust Batch Size**: Smaller batch sizes reduce memory usage

## Advanced Usage

### Custom Problems

You can create custom problems by extending the `BaseProblem` class:

```python
from core.problems import BaseProblem

class CustomProblem(BaseProblem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_num = 2  # Number of objectives
    
    def _setup_llm_components(self):
        # Setup your LLM components
        pass
    
    def sample_test_data(self):
        # Sample your test data
        pass
    
    async def evaluate(self, population):
        # Implement your evaluation logic
        pass
```

### Custom Algorithms

You can create custom algorithms by extending the `BaseAlgorithm` class:

```python
from core.algorithms import BaseAlgorithm

class CustomAlgorithm(BaseAlgorithm):
    async def run(self, save_path=None):
        # Implement your algorithm logic
        pass
```

## References

- [NSGA-II Paper](https://ieeexplore.ieee.org/document/996017)
- [Multi-objective Optimization](https://en.wikipedia.org/wiki/Multi-objective_optimization)
- [Pareto Front](https://en.wikipedia.org/wiki/Pareto_front)

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the main library documentation
3. Check the example files for usage patterns
4. Ensure all dependencies are properly installed