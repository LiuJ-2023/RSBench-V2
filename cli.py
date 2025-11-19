"""
Command-line interface for RSBench-V3 library.
"""

import argparse
import asyncio
import os
from dotenv import load_dotenv

from core.problems import AccDivProblem, AccFairProblem, AccDivFairProblem
from core.algorithms import get_algorithm
from datasets import create_dataset_loader


async def run_experiment(args):
    """Run a single experiment."""
    # Load environment variables
    load_dotenv()
    
    # Get API key
    if args.llm_model == 'gpt':
        api_key = os.getenv("OPENAI_API_KEY")
    elif args.llm_model == 'glm':
        api_key = os.getenv("ZHIPUAI_API_KEY")
    else:
        raise ValueError(f"Unsupported LLM model: {args.llm_model}")
    
    if not api_key:
        print(f"❌ Please set {args.llm_model.upper()}_API_KEY in your .env file")
        return
    
    # Load dataset
    print(f"📁 Loading {args.dataset} dataset...")
    loader = create_dataset_loader(args.dataset, args.data_path)
    train_data = loader.get_train_data(size=args.data_size)
    print(f"✅ Loaded {len(train_data)} training samples")
    
    # Create problem
    print(f"🔧 Creating {args.problem} problem...")
    problem_map = {
        "AccDiv": AccDivProblem,
        "AccFair": AccFairProblem,
        "AccDivFair": AccDivFairProblem
    }
    
    if args.problem not in problem_map:
        raise ValueError(f"Unknown problem: {args.problem}")
    
    problem = problem_map[args.problem](
        train_data=train_data,
        batch_size=args.batch_size,
        api_key=api_key,
        llm_model=args.llm_model
    )
    print("✅ Problem created")
    
    # Create algorithm
    print(f"🧬 Creating {args.algorithm} algorithm...")
    algorithm = get_algorithm(
        algorithm_name=args.algorithm,
        problem=problem,
        pop_size=args.pop_size,
        max_iter=args.max_iter,
        api_key=api_key,
        llm_model=args.llm_model
    )
    print("✅ Algorithm created")
    
    # Run optimization
    print("🏃 Running optimization...")
    population, objectives = await algorithm.run(save_path=args.save_path)
    print("✅ Optimization completed")
    
    # Display results
    print(f"\n📊 RESULTS:")
    print(f"Final population size: {len(population)}")
    print(f"Objective values shape: {objectives.shape}")
    
    # Token usage
    print(f"\n💰 Token Usage:")
    token_stats = problem.token_stats
    print(f"Total tokens: {token_stats['total_tokens']}")
    print(f"Estimated cost: ${token_stats['total_cost']:.4f}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="RSBench-V3: Evolutionary Computation for LLM-based Recommender Systems")
    
    # Experiment parameters
    parser.add_argument("-a", "--algorithm", default="NSGA2", 
                       choices=["NSGA2", "MOEAD", "IBEA"],
                       help="Evolutionary algorithm to use")
    parser.add_argument("-p", "--problem", default="AccDiv",
                       choices=["AccDiv", "AccFair", "AccDivFair"],
                       help="Problem type to optimize")
    parser.add_argument("-d", "--dataset", default="MovieLens",
                       choices=["MovieLens", "Game", "Bundle"],
                       help="Dataset to use")
    parser.add_argument("-l", "--llm", default="gpt",
                       choices=["gpt", "glm"],
                       help="LLM model to use")
    
    # Algorithm parameters
    parser.add_argument("--pop-size", type=int, default=20,
                       help="Population size")
    parser.add_argument("--max-iter", type=int, default=10,
                       help="Maximum number of iterations")
    parser.add_argument("--batch-size", type=int, default=20,
                       help="Evaluation batch size")
    
    # Data parameters
    parser.add_argument("--data-path", default="Dataset/Movie",
                       help="Path to dataset directory")
    parser.add_argument("--data-size", type=int, default=100,
                       help="Number of training samples to use")
    
    # Output parameters
    parser.add_argument("--save-path", default=None,
                       help="Path to save results")
    
    args = parser.parse_args()
    
    # Run experiment
    asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()
