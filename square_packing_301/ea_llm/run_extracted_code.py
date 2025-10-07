#!/usr/bin/env python3
"""
Script to run the extracted LLM-generated code block.
"""

import os
from logger import LOGGER

def run_python_code(code_file_path, function_name="square_301_solver"):
    """Run the extracted code block and optionally call a specific function."""
    if not os.path.exists(code_file_path):
        LOGGER.error(f"File not found: {code_file_path}")
        return None
    
    try:
        # Create namespace for execution
        namespace = {}
        
        with open(code_file_path, 'r', encoding='utf-8') as f:
            code_content = f.read()
        
        # Execute the code to load functions and classes
        exec(code_content, namespace)
        
        # Try to call the specified function if it exists
        if function_name and function_name in namespace:
            if callable(namespace[function_name]):
                result = namespace[function_name]()
                if isinstance(result, list):
                    LOGGER.info(f"Generated {len(result)} solutions")
                return result
            else:
                LOGGER.warning(f"'{function_name}' found but is not callable")
        elif function_name:
            LOGGER.error(f"Function '{function_name}' not found")
        
        return None
        
    except Exception as e:
        LOGGER.error(f"Error executing code: {e}")
        return None

def store_results(results, output_file='llm_generated_solution.json'):
    """Store the results in a JSON file."""
    import json
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        LOGGER.error(f"Error saving results: {e}")

if __name__ == "__main__":
    # Default file path
    file_path = os.path.join(os.path.dirname(__file__), 'llm_gen', 'extracted_code_block_1.py')
    function_name = "square_301_solver"
    
    result = run_python_code(file_path, function_name)
    store_results(result, output_file=os.path.join(os.path.dirname(__file__), 'llm_gen', 'llm_generated_solution.json'))