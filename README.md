# RSBench-V2: Multi-Objective Recommender System Benchmark

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/status-active-green.svg)](https://github.com/your-repo/RSBench-V2)

## 🎯 Overview

RSBench-V2 is a comprehensive library for multi-objective optimization of recommender systems using Large Language Models (LLMs) and evolutionary algorithms. It provides a standardized benchmark for evaluating recommendation systems across multiple objectives including accuracy, diversity, and fairness.

> **⚠️ Note: This is a new version** - Compared to the original paper, this implementation features **asynchronous evaluation** that significantly improves execution efficiency. All LLM calls are executed in parallel using asyncio, making the evaluation process much faster than the sequential approach described in the original work.

## 🚀 Why RSBench-V2 is a Library

### 1. **Modular Architecture**
- **Core Components**: Separate modules for algorithms, problems, datasets, and utilities
- **Extensible Design**: Easy to add new algorithms, problems, or datasets
- **Plugin System**: LLM operators can be swapped and customized

### 2. **Standardized Interface**
- **Unified API**: Consistent interface across all components
- **Type Hints**: Full type annotation for better IDE support
- **Documentation**: Comprehensive docstrings and examples

### 3. **Production-Ready Features**
- **Async Support**: High-performance asynchronous evaluation
- **Error Handling**: Robust error handling and recovery
- **Logging**: Comprehensive logging and progress tracking
- **Configuration**: Flexible configuration system

### 4. **Research & Development Tools**
- **Visualization**: Built-in plotting and analysis tools
- **Metrics**: Comprehensive performance metrics
- **Benchmarking**: Standardized evaluation protocols
- **Reproducibility**: Deterministic results with seed control

### 5. **Multi-Objective Optimization**
- **NSGA-II**: State-of-the-art multi-objective evolutionary algorithm
- **Pareto Front**: Automatic Pareto front identification
- **Diversity Metrics**: Crowding distance and spread metrics
- **Convergence Analysis**: Hypervolume and convergence tracking

## 🏗️ Architecture

```
RSBench-V2/
├── core/                    # Core library components
│   ├── algorithms.py       # Evolutionary algorithms (NSGA-II, MOEA/D, IBEA)
│   ├── problems.py         # Multi-objective problems (AccDiv, AccFair, AccDivFair)
│   └── operators.py        # LLM-based genetic operators
├── datasets/               # Dataset handling
│   ├── loader.py          # Dataset loaders (MovieLens, Game, Bundle)
│   └── __init__.py        # Dataset factory functions
├── utils/                  # Utility modules
│   ├── plotting.py        # Visualization and plotting
│   ├── metrics.py         # Performance metrics
│   ├── selection.py       # Selection operators
│   └── dominance.py       # Pareto dominance utilities
├── examples/               # Example scripts
│   ├── basic_example.py   # Basic usage example
│   └── nsga2_with_plotting.py  # Advanced example with plotting
└── Results/               # Generated results and plots
```

## 🎯 Key Features

### **Multi-Objective Problems**
- **Accuracy vs Diversity**: Balance recommendation quality with item variety
- **Accuracy vs Fairness**: Ensure equitable recommendations across groups
- **Accuracy vs Diversity vs Fairness**: Comprehensive three-objective optimization

### **Evolutionary Algorithms**
- **NSGA-II**: Non-dominated Sorting Genetic Algorithm II
- **MOEA/D**: Multi-Objective Evolutionary Algorithm based on Decomposition
- **IBEA**: Indicator-Based Evolutionary Algorithm

### **LLM Integration**
- **GPT Models**: OpenAI GPT-3.5/4 integration
- **GLM Models**: ZhipuAI GLM integration
- **Custom Models**: Extensible for other LLM providers
- **Token Tracking**: Cost monitoring and optimization

### **High-Performance Evaluation**
- **Async Processing**: Parallel evaluation of populations - **ALL prompts and ALL samples run in parallel**
- **True Asyncio**: Maximum parallelism with asyncio for optimal performance
- **Batch Processing**: Efficient batch evaluation
- **Rate Limiting**: API rate limit management
- **Error Recovery**: Robust error handling and fallbacks
- **Performance Improvement**: Significantly faster than sequential evaluation in the original paper

### **Visualization & Analysis**
- **Evolution Plots**: Objective space evolution visualization
- **Pareto Fronts**: Non-dominated solution highlighting
- **Metrics Tracking**: Performance metrics over generations
- **Export Options**: Multiple output formats (PNG, TXT, NPY)

## 📦 Installation

### Prerequisites
- Python 3.8+
- OpenAI API key (for GPT models)
- ZhipuAI API key (for GLM models, optional)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Setup Environment
```bash
# Create .env file with your API keys
echo "OPENAI_API_KEY=your_openai_key_here" > .env
echo "ZHIPUAI_API_KEY=your_zhipuai_key_here" >> .env  # Optional
```

## 🚀 Quick Start

### Basic Usage
```python
import asyncio
from core.algorithms import NSGA2LLM
from core.problems import AccDivProblem
from datasets import create_dataset_loader

async def main():
    # Load dataset
    dataset_loader = create_dataset_loader("MovieLens", "Dataset/Movie")
    train_data = dataset_loader.get_train_data()
    
    # Create problem
    problem = AccDivProblem(
        train_data=train_data,
        batch_size=5,
        llm_model='gpt',
        api_key=os.getenv('OPENAI_API_KEY')
    )
    
    # Create algorithm
    algorithm = NSGA2LLM(
        problem=problem,
        pop_size=10,
        max_iter=3,
        api_key=os.getenv('OPENAI_API_KEY'),
        llm_model='gpt'
    )
    
    # Run optimization
    final_population, final_objectives = await algorithm.run()
    
    print(f"Optimization completed!")
    print(f"Final population size: {len(final_population)}")
    print(f"Objectives shape: {final_objectives.shape}")

# Run the example
asyncio.run(main())
```

### With Plotting
```python
from core.algorithms_with_plotting import NSGA2LLM

# Enable plotting
algorithm = NSGA2LLM(
    problem=problem,
    pop_size=10,
    max_iter=3,
    enable_plotting=True  # Enable visualization
)

# Run optimization with automatic plot generation
final_population, final_objectives = await algorithm.run()
```

## 📊 Example Output

```
🚀 NSGA2-LLM Algorithm Starting!
Initializing the Population...
🔄 Evaluating initial population...
🚀 TRULY ASYNC EVALUATION
⚡ ALL prompts and ALL samples run in parallel with asyncio!
🔧 Processing 10 prompts with 5 samples each
🚀 Creating 50 async LLM tasks...
⚡ Executing 50 LLM calls in parallel...
📊 Evaluation Summary: 10 prompts, 50 samples, 19.09s total
✅ Initial evaluation completed in: 0.59 minutes

> **Performance Note**: The asynchronous evaluation shown above demonstrates the efficiency improvement over sequential evaluation. All LLM calls are executed concurrently, resulting in significantly reduced total execution time compared to the original paper's approach.

Evolution is starting!
🔄 Generation 1/3
⚡ Crossover completed in: 17.54 seconds
🔄 Evaluating offspring generation 1...
📊 Evaluation Summary: 10 prompts, 50 samples, 20.22s total
✅ Generation 1 evaluation completed in: 20.22 seconds
⚡ Environment selection completed in: 0.00 seconds
✅ Accomplished iteration 1

🎯 Evolution has been finished!
📊 Generating evolution plots...
📊 Plot saved: Results/NSGA2-LLM_Evolution_20250913_205950.png
📊 Metrics plot saved: Results/NSGA2-LLM_Metrics_20250913_205950.png
📋 Summary report saved: Results/NSGA2-LLM_Summary_20250913_205950.txt

📊 ALGORITHM SUMMARY:
Algorithm: NSGA2-LLM
Population Size: 10
Iterations: 3
Final Population Size: 10
Objectives: 2

=== Token Usage Summary ===
Total tokens: 379724
Prompt tokens: 237084
Completion tokens: 142640
Estimated cost: $0.3325
```

## 🔧 Configuration

### Algorithm Parameters
```python
algorithm = NSGA2LLM(
    problem=problem,
    pop_size=20,           # Population size
    max_iter=10,           # Maximum iterations
    api_key=api_key,
    llm_model='gpt',       # 'gpt' or 'glm'
    enable_plotting=True   # Enable visualization
)
```

### Problem Parameters
```python
problem = AccDivProblem(
    train_data=train_data,
    batch_size=10,         # Evaluation batch size
    llm_model='gpt',
    api_key=api_key
)
```

## 📈 Performance Metrics

### Multi-Objective Metrics
- **Hypervolume**: Volume of objective space dominated by solutions
- **Spread**: Diversity of solutions in objective space
- **Convergence**: Distance to reference Pareto front
- **Pareto Front Size**: Number of non-dominated solutions

### Efficiency Metrics
- **Evaluation Time**: Time per generation
- **Token Usage**: LLM API usage and costs
- **Memory Usage**: Memory consumption tracking
- **Parallel Efficiency**: Async processing performance

## 🎨 Visualization

The library automatically generates:
- **Evolution Plots**: Objective space evolution over generations
- **Pareto Fronts**: Non-dominated solution visualization
- **Metrics Plots**: Performance metrics over time
- **Summary Reports**: Comprehensive algorithm analysis

## 🔬 Research Applications

### Academic Research
- **Multi-objective Optimization**: Benchmark for MOO algorithms
- **Recommender Systems**: Evaluation of recommendation strategies
- **LLM Integration**: Study of LLM-based optimization
- **Fairness in AI**: Bias and fairness analysis

### Industry Applications
- **Recommendation Systems**: Production recommendation optimization
- **Content Curation**: Multi-objective content selection
- **Personalization**: Balanced personalization strategies
- **A/B Testing**: Automated optimization testing

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/your-repo/RSBench-V2.git
cd RSBench-V2
pip install -e .
pip install -r requirements-dev.txt
```

### Running Tests
```bash
python -m pytest tests/
```
---

**RSBench-V2** - Advancing Multi-Objective Recommender System Research 🚀
