"""
Basic example of using RSBench-V3 library.

This example demonstrates how to:
1. Load a dataset
2. Create a problem instance
3. Run an evolutionary algorithm
4. Evaluate results
"""

import os
import sys
import asyncio
import numpy as np
from dotenv import load_dotenv

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
try:
    load_dotenv()
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")
    print("Please create a .env file with your API key or set it directly in the code")

# Import RSBench-V3 components
from core.problems import AccDivProblem
from core.algorithms import get_algorithm
from datasets import create_dataset_loader
from core.evaluation import Evaluator


async def main():
    """Main example function."""
    print("🚀 RSBench-V3 Basic Example")
    print("=" * 50)
    
    # Configuration
    dataset_name = "MovieLens"
    dataset_path = "Dataset/Movie"  # Adjust path as needed
    algorithm_name = "NSGA2"
    pop_size = 10
    max_iter = 3
    batch_size = 20
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY in your .env file")
        return
    
    try:
        # 1. Load dataset
        print("📁 Loading dataset...")
        loader = create_dataset_loader(dataset_name, dataset_path)
        train_data = loader.get_train_data(size=100)  # Load 100 samples
        print(f"✅ Loaded {len(train_data)} training samples")
        
        # 2. Create problem instance
        print("🔧 Creating problem instance...")
        problem = AccDivProblem(
            train_data=train_data,
            batch_size=batch_size,
            api_key=api_key,
            llm_model='gpt'
        )
        print("✅ Problem instance created")
        
        # 3. Create algorithm
        print("🧬 Creating evolutionary algorithm...")
        algorithm = get_algorithm(
            algorithm_name=algorithm_name,
            problem=problem,
            pop_size=pop_size,
            max_iter=max_iter,
            api_key=api_key,
            llm_model='gpt'
        )
        print("✅ Algorithm created")
        
        # 4. Run optimization
        print("🏃 Running optimization...")
        final_population, final_objectives = await algorithm.run()
        print("✅ Optimization completed")
        
        # 5. Display results
        print("\n📊 RESULTS:")
        print("-" * 30)
        print(f"Final population size: {len(final_population)}")
        print(f"Objective values shape: {final_objectives.shape}")
        print(f"Number of objectives: {final_objectives.shape[1]}")
        
        # Show best solutions
        print("\n🏆 Best Solutions:")
        for i in range(min(3, len(final_population))):
            print(f"\nSolution {i+1}:")
            print(f"  Objectives: {final_objectives[i]}")
            print(f"  Prompt: {final_population[i][:100]}...")
        
        # 6. Evaluate results
        print("\n📈 Evaluation:")
        evaluator = Evaluator()
        metrics = evaluator.evaluate_run(final_objectives, "example_run", algorithm_name)
        
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        # 7. Token usage summary
        print("\n💰 Token Usage Summary:")
        token_stats = problem.token_stats
        print(f"  Total tokens: {token_stats['total_tokens']}")
        print(f"  Prompt tokens: {token_stats['prompt_tokens']}")
        print(f"  Completion tokens: {token_stats['completion_tokens']}")
        print(f"  Estimated cost: ${token_stats['total_cost']:.4f}")
        
        print("\n🎉 Example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
