"""
LLM Operators for Evolutionary Computation.

This module provides LLM-based operators for evolutionary algorithms,
including initialization, crossover, mutation, and token tracking.
"""

import asyncio
import time
import re
import numpy as np
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.callbacks.manager import AsyncCallbackManager


class TokenCounterCallback(AsyncCallbackHandler):
    """
    Callback handler for tracking token usage and costs.
    
    This class tracks token consumption across LLM calls and estimates
    the associated costs based on the model being used.
    """
    
    def __init__(self, token_stats: Dict[str, Any], model: str):
        """
        Initialize the token counter.
        
        Args:
            token_stats: Dictionary to store token statistics
            model: Model type ('gpt' or 'glm')
        """
        super().__init__()
        self.token_stats = token_stats
        self.model = model
    
    async def on_chat_model_start(self, serialized, inputs, **kwargs):
        """Called when a chat model starts (required by interface)."""
        pass
    
    async def on_llm_end(self, response, **kwargs):
        """Collect token usage and calculate costs."""
        token_usage = response.llm_output.get('token_usage', {})
        
        # Accumulate token statistics
        self.token_stats['total_tokens'] += token_usage.get('total_tokens', 0)
        self.token_stats['prompt_tokens'] += token_usage.get('prompt_tokens', 0)
        self.token_stats['completion_tokens'] += token_usage.get('completion_tokens', 0)
        
        # Calculate cost based on model type
        if self.model == 'gpt':
            # GPT-3.5-turbo pricing (adjust as needed)
            cost = (token_usage.get('prompt_tokens', 0) * 0.0005/1000 +
                    token_usage.get('completion_tokens', 0) * 0.0015/1000)
        elif self.model == 'glm':
            # GLM pricing (adjust as needed)
            cost = (token_usage.get('prompt_tokens', 0) + 
                    token_usage.get('completion_tokens', 0)) * 0.01/1000
        else:
            cost = 0
        
        self.token_stats['total_cost'] += cost


class LLMOperator(ABC):
    """
    Abstract base class for LLM-based evolutionary operators.
    
    This class provides the interface for LLM operators used in
    evolutionary algorithms, including initialization and crossover.
    """
    
    def __init__(self, 
                 llm_model: str,
                 api_key: str,
                 token_stats: Dict[str, Any],
                 **kwargs):
        """
        Initialize the LLM operator.
        
        Args:
            llm_model: LLM model type ('gpt' or 'glm')
            api_key: API key for LLM services
            token_stats: Dictionary for tracking token usage
            **kwargs: Additional parameters
        """
        self.llm_model = llm_model
        self.api_key = api_key
        self.token_stats = token_stats
        
        # Initialize LLM components
        self._setup_llm_components()
    
    @abstractmethod
    def _setup_llm_components(self):
        """Setup LLM components specific to the operator."""
        pass
    
    @abstractmethod
    def initialize(self, example: str, pop_size: int) -> List[str]:
        """
        Initialize a population using LLM.
        
        Args:
            example: Example prompt for initialization
            pop_size: Population size
            
        Returns:
            List of initialized prompts
        """
        pass
    
    @abstractmethod
    def crossover(self, population: List[str]) -> List[str]:
        """
        Perform crossover operation using LLM.
        
        Args:
            population: Current population
            
        Returns:
            List of offspring prompts
        """
        pass


class StandardLLMOperator(LLMOperator):
    """
    Standard LLM operator for evolutionary algorithms.
    
    This operator provides basic initialization and crossover operations
    using LLM-based prompt generation and modification.
    """
    
    def __init__(self, *args, **kwargs):
        # Default prompts
        self.initial_prompt = """Now, I have a prompt for my task. I want to modify this prompt to better achieve my task.
        I will give an example of my current prompt. Please randomly generate a prompt based on my example.
        My example is as follows:
        {example}
        Note that the final prompt should be bracketed with <START> and <END>."""
        
        self.crossover_prompt = """Now, I will give you two existing prompts.
        Prompt 1: {prompt1}
        Prompt 2: {prompt2}
        Please follow the following crossover and mutation instructions step-by-step to generate a better prompt.
        Please keep the prompt as a chain-of-thought.
        1. Thought Level Crossover: Randomly swap the execution steps of two chain-of-thought.
        2. Step Level Crossover: For each of the two prompts, select a suitable step. Then, crossover the selected steps to generate a new prompt.
        3. Mutate the prompt generated in Step 2. The mutation operator can be add a step, delete a step, or modify an existing step.
        Note that, the final prompt should be bracketed with <START> and <END>."""
        
        super().__init__(*args, **kwargs)
    
    def _setup_llm_components(self):
        """Setup LLM components for standard operations."""
        # Initialize LLM
        if self.llm_model == 'glm':
            import os
            os.environ["ZHIPUAI_API_KEY"] = self.api_key
            self.llm = ChatZhipuAI(model="glm-4")
        elif self.llm_model == 'gpt':
            self.llm = ChatOpenAI(api_key=self.api_key)
        
        self.output_parser = StrOutputParser()
        
        # Setup initialization chain
        init_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an initializer to provide a set of initial prompts according to user's requirement"),
            ("user", self.initial_prompt)
        ])
        self.chain_initialize = init_prompt | self.llm | self.output_parser
        
        # Setup crossover chain
        crossover_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an evolutionary operator for prompt optimization."),
            ("user", self.crossover_prompt)
        ])
        self.chain_crossover = crossover_prompt | self.llm | self.output_parser
    
    def initialize(self, example: str, pop_size: int) -> List[str]:
        """
        Initialize population using LLM.
        
        Args:
            example: Example prompt for initialization
            pop_size: Population size
            
        Returns:
            List of initialized prompts
        """
        population = []
        
        for i in range(pop_size):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    token_counter = TokenCounterCallback(self.token_stats, self.llm_model)
                    output = self.chain_initialize.invoke(
                        {"example": example},
                        config={"callbacks": [token_counter]}
                    )
                    break
                except Exception as e:
                    print(f'Initialization failed, attempt {attempt + 1}/{max_retries}: {e}')
                    if attempt == max_retries - 1:
                        # Use example as fallback
                        output = f"<START>{example}<END>"
                    time.sleep(2)
            
            # Extract prompts from output
            prompts = self._extract_prompts(output)
            population.extend(prompts)
        
        return population[:pop_size]  # Ensure correct population size
    
    def crossover(self, population: List[str]) -> List[str]:
        """
        Perform crossover operation using LLM.
        
        Args:
            population: Current population
            
        Returns:
            List of offspring prompts
        """
        offspring = []
        
        for i in range(len(population)):
            # Select two parents
            parent_indices = np.random.choice(len(population), 2, replace=False)
            parent1 = population[parent_indices[0]]
            parent2 = population[parent_indices[1]]
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    token_counter = TokenCounterCallback(self.token_stats, self.llm_model)
                    output = self.chain_crossover.invoke(
                        {"prompt1": parent1, "prompt2": parent2},
                        config={"callbacks": [token_counter]}
                    )
                    break
                except Exception as e:
                    print(f'Crossover failed, attempt {attempt + 1}/{max_retries}: {e}')
                    if attempt == max_retries - 1:
                        # Use parent1 as fallback
                        output = f"<START>{parent1}<END>"
                    time.sleep(2)
            
            # Extract prompts from output
            prompts = self._extract_prompts(output)
            offspring.extend(prompts)
        
        return offspring[:len(population)]  # Ensure correct offspring size
    
    def _extract_prompts(self, response: str) -> List[str]:
        """
        Extract prompts from LLM response.
        
        Args:
            response: LLM response string
            
        Returns:
            List of extracted prompts
        """
        # Try different patterns to extract prompts
        patterns = [
            r'<START>\s*(.*?)\s*<END>',
            r'<START>(.*?)<END>',
            r'```(.*?)```',
            r'"(.*?)"'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return [match.strip() for match in matches if match.strip()]
        
        # If no pattern matches, return the whole response
        return [response.strip()]


class AdvancedLLMOperator(LLMOperator):
    """
    Advanced LLM operator with enhanced capabilities.
    
    This operator provides more sophisticated initialization and crossover
    operations with better prompt engineering and error handling.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Enhanced prompts
        self.initial_prompt = """You are an expert prompt engineer. I need you to create diverse and effective prompts for a recommender system task.

Given this example prompt:
{example}

Please generate a new prompt that:
1. Maintains the core functionality of the example
2. Uses different wording and structure
3. Incorporates best practices for prompt engineering
4. Is optimized for the specific task

The prompt should be bracketed with <START> and <END>."""
        
        self.crossover_prompt = """You are an expert in prompt optimization. I will give you two prompts and you need to create a better one by combining their strengths.

Prompt 1: {prompt1}
Prompt 2: {prompt2}

Please create a new prompt that:
1. Combines the best elements from both prompts
2. Maintains logical flow and coherence
3. Incorporates innovative approaches
4. Is optimized for performance

The new prompt should be bracketed with <START> and <END>."""
    
    def _setup_llm_components(self):
        """Setup LLM components for advanced operations."""
        if self.llm_model == 'glm':
            import os
            os.environ["ZHIPUAI_API_KEY"] = self.api_key
            self.llm = ChatZhipuAI(model="glm-4")
        elif self.llm_model == 'gpt':
            self.llm = ChatOpenAI(api_key=self.api_key)
        
        self.output_parser = StrOutputParser()
        
        # Setup initialization chain
        init_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert prompt engineer specializing in recommender systems."),
            ("user", self.initial_prompt)
        ])
        self.chain_initialize = init_prompt | self.llm | self.output_parser
        
        # Setup crossover chain
        crossover_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in prompt optimization and evolutionary computation."),
            ("user", self.crossover_prompt)
        ])
        self.chain_crossover = crossover_prompt | self.llm | self.output_parser
    
    def initialize(self, example: str, pop_size: int) -> List[str]:
        """Initialize population with enhanced prompt engineering."""
        population = []
        
        for i in range(pop_size):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    token_counter = TokenCounterCallback(self.token_stats, self.llm_model)
                    output = self.chain_initialize.invoke(
                        {"example": example},
                        config={"callbacks": [token_counter]}
                    )
                    break
                except Exception as e:
                    print(f'Advanced initialization failed, attempt {attempt + 1}/{max_retries}: {e}')
                    if attempt == max_retries - 1:
                        output = f"<START>{example}<END>"
                    time.sleep(2)
            
            prompts = self._extract_prompts(output)
            population.extend(prompts)
        
        return population[:pop_size]
    
    def crossover(self, population: List[str]) -> List[str]:
        """Perform advanced crossover with better prompt engineering."""
        offspring = []
        
        for i in range(len(population)):
            parent_indices = np.random.choice(len(population), 2, replace=False)
            parent1 = population[parent_indices[0]]
            parent2 = population[parent_indices[1]]
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    token_counter = TokenCounterCallback(self.token_stats, self.llm_model)
                    output = self.chain_crossover.invoke(
                        {"prompt1": parent1, "prompt2": parent2},
                        config={"callbacks": [token_counter]}
                    )
                    break
                except Exception as e:
                    print(f'Advanced crossover failed, attempt {attempt + 1}/{max_retries}: {e}')
                    if attempt == max_retries - 1:
                        output = f"<START>{parent1}<END>"
                    time.sleep(2)
            
            prompts = self._extract_prompts(output)
            offspring.extend(prompts)
        
        return offspring[:len(population)]
    
    def _extract_prompts(self, response: str) -> List[str]:
        """Extract prompts with enhanced pattern matching."""
        patterns = [
            r'<START>\s*(.*?)\s*<END>',
            r'<START>(.*?)<END>',
            r'```python\s*(.*?)\s*```',
            r'```(.*?)```',
            r'"(.*?)"',
            r"'(.*?)'"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return [match.strip() for match in matches if match.strip()]
        
        return [response.strip()]


def get_llm_operator(operator_type: str = "standard", **kwargs) -> LLMOperator:
    """
    Factory function to create LLM operators.
    
    Args:
        operator_type: Type of operator ("standard" or "advanced")
        **kwargs: Additional parameters for the operator
        
    Returns:
        LLMOperator instance
    """
    if operator_type == "standard":
        return StandardLLMOperator(**kwargs)
    elif operator_type == "advanced":
        return AdvancedLLMOperator(**kwargs)
    else:
        raise ValueError(f"Unknown operator type: {operator_type}")
