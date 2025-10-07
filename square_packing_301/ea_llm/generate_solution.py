
"""
Main script to generate square packing solutions using LLM prompts
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from logger import LOGGER
from util import generate_solution
from config_prompt import PROMPT_SOLUTION_GENERATOR, PROMPT_SOLUTION_GENERATOR_SYSTEM

def main():
    """Main function to generate and validate a square packing solution."""
    # Generate solution using LLM
    LOGGER.info("Generating solution using LLM...")
    
    # Find next generation directory (gen_0, gen_1, etc.)
    base_dir = os.path.join(os.path.dirname(__file__), "generated_solutions")
    gen_num = 0
    
    # Find the current generation
    while os.path.exists(os.path.join(base_dir, f'gen_{gen_num}')):
        gen_num += 1
    
    # If gen_num > 0, check if the previous generation has files
    if gen_num > 0:
        prev_gen_dir = os.path.join(base_dir, f'gen_{gen_num - 1}')
        if os.path.exists(prev_gen_dir):
            # Check if previous generation directory has any files
            files_in_prev = [f for f in os.listdir(prev_gen_dir) if os.path.isfile(os.path.join(prev_gen_dir, f))]
            if not files_in_prev:
                # Use the empty previous generation directory instead of creating new one
                gen_num -= 1
    
    gen_dir = os.path.join(base_dir, f'gen_{gen_num}')
    os.makedirs(gen_dir, exist_ok=True)
    response_path = os.path.join(gen_dir, 'llm_response.txt')
    

    generate_solution(provider="claude",
                      model="sonnet-4",
                      response_path=response_path,
                      prompt=PROMPT_SOLUTION_GENERATOR,
                      system_prompt=PROMPT_SOLUTION_GENERATOR_SYSTEM)


if __name__ == "__main__":
    main()