#!/usr/bin/env python3
"""
Script to run the extracted LLM-generated code block.
"""

import os

def run_python_code(code_file_path, function_name="square_301_solver"):
    """Run the extracted code block and optionally call a specific function."""
    print("=" * 60)
    print("RUNNING EXTRACTED LLM CODE BLOCK")
    print("=" * 60)
    
    # Check if extracted code exists
    if not os.path.exists(code_file_path):
        print(f"File not found: {code_file_path}")
        return None
    
    print(f"Running: {code_file_path}")
    
    try:
        # Create namespace for execution
        namespace = {}
        
        with open(code_file_path, 'r', encoding='utf-8') as f:
            code_content = f.read()
        
        # Execute the code to load functions and classes
        exec(code_content, namespace)
        print("Code executed successfully!")
        
        # Try to call the specified function if it exists
        if function_name and function_name in namespace:
            if callable(namespace[function_name]):
                print(f"Calling function: {function_name}()")
                result = namespace[function_name]()
                print(f"Function completed. Result type: {type(result)}")
                if isinstance(result, list):
                    print(f"Returned {len(result)} items")
                return result
            else:
                print(f"'{function_name}' found but is not callable")
        elif function_name:
            print(f"Function '{function_name}' not found in code")
            print("Available functions:")
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith('_'):
                    print(f"  - {name}")
        
        return None
        
    except Exception as e:
        print(f"Error executing code: {e}")
        return None

def store_results(results, output_file='llm_generated_solution.json'):
    """Store the results in a JSON file."""
    import json
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    # Default file path
    file_path = os.path.join(os.path.dirname(__file__), 'llm_gen', 'extracted_code_block_1.py')
    function_name = "square_301_solver"
    
    result = run_python_code(file_path, function_name)
    store_results(result, output_file=os.path.join(os.path.dirname(__file__), 'llm_gen', 'llm_generated_solution.json'))