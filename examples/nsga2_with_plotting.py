"""
NSGA-II with Plotting Demo - Enhanced version with evolution visualization.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from dotenv import load_dotenv

# Load environment variables
try:
    load_dotenv()
except:
    print("⚠️  No .env file found, using system environment variables")

from core.algorithms_with_plotting import NSGA2LLM
from core.problems import AccDivProblem
from datasets import create_dataset_loader
from utils.plotting import EvolutionPlotter

async def main():
    print("🚀 RSBench-V3 NSGA-II with Plotting Demo")
    print("=" * 50)
    
    # Load dataset
    print("📁 Loading dataset...")
    dataset_loader = create_dataset_loader("MovieLens", "Dataset/Movie")
    train_data = dataset_loader.get_train_data()
    print(f"✅ Loaded {len(train_data)} training samples")
    
    # Create problem instance
    print("🔧 Creating problem instance...")
    problem = AccDivProblem(
        train_data=train_data,
        batch_size=20,  # Small batch for demo
        llm_model='gpt',
        api_key=os.getenv('OPENAI_API_KEY')
    )
    print("✅ Problem instance created")
    
    # Create algorithm with plotting enabled
    print("🧬 Creating evolutionary algorithm with plotting...")
    algorithm = NSGA2LLM(
        problem=problem,
        pop_size=10,  # Small population for demo
        max_iter=5,  # Few iterations for demo
        api_key=os.getenv('OPENAI_API_KEY'),
        llm_model='gpt',
        enable_plotting=True  # Enable plotting
    )
    print("✅ Algorithm created with plotting enabled")
    
    # Run optimization
    print("🏃 Running optimization with plotting...")
    final_population, final_objectives = await algorithm.run()
    
    print("✅ Optimization completed")
    print(f"📊 Final population size: {len(final_population)}")
    print(f"📊 Final objectives shape: {final_objectives.shape}")
    
    # Display results
    print("\n📊 RESULTS:")
    print("-" * 30)
    print(f"Final population size: {len(final_population)}")
    print(f"Objective values shape: {final_objectives.shape}")
    print(f"Number of objectives: {final_objectives.shape[1]}")
    
    print("\n🏆 Best Solutions:")
    for i, (prompt, obj) in enumerate(zip(final_population[:3], final_objectives[:3])):
        print(f"\nSolution {i+1}:")
        print(f"  Objectives: {obj}")
        print(f"  Prompt: {prompt[:100]}...")
    
    # Show final metrics
    if hasattr(algorithm, 'plotter') and algorithm.plotter:
        final_metrics = algorithm._calculate_metrics(final_objectives)
        print(f"\n📈 Final Evaluation:")
        for metric, value in final_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Token usage summary
    token_stats = problem.token_stats
    print(f"\n💰 Token Usage Summary:")
    print(f"  Total tokens: {token_stats['total_tokens']}")
    print(f"  Prompt tokens: {token_stats['prompt_tokens']}")
    print(f"  Completion tokens: {token_stats['completion_tokens']}")
    print(f"  Estimated cost: ${token_stats.get('estimated_cost', 0.0):.4f}")
    
    print("\n🎉 Demo completed successfully!")
    print("📁 Check the 'Results' folder for generated plots and reports!")

if __name__ == "__main__":
    asyncio.run(main())
