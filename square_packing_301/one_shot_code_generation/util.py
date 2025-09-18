import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from llm.call_llm import call_llm
from jinja2 import Template
import os
from typing import List, Tuple


def generate_prompt(number_of_squares: int) -> str:
    """Generate the prompt for the LLM using a Jinja2 template."""
    template_path = os.path.join(os.path.dirname(__file__), 'prompts/one_shot_prompt.j2')
    with open(template_path, 'r') as file:
        template_content = file.read()
    
    template = Template(template_content)
    prompt = template.render(NUMBER_OF_SQUARES=number_of_squares)
    return prompt

def parse_solution(response: str) -> List[Tuple[int, int]]:
    """
    Parse the LLM response to extract a single Python code block and save it to a file
    """
    import re
    
    # Extract Python code blocks from the response
    code_blocks = re.findall(r'```python\s*(.*?)\s*```', response, re.DOTALL)
    
    if not code_blocks:
        print("No Python code blocks found in the response")
        return []
    
    if len(code_blocks) > 1:
        print(f"Warning: Found {len(code_blocks)} code blocks, expected only 1. Using the first one.")
    
    # Save the first (and expected only) code block
    code_block = code_blocks[0]
    code_file_path = os.path.join(os.path.dirname(__file__), 'llm_gen/extracted_code_block_1.py')
    
    try:
        with open(code_file_path, 'w', encoding='utf-8') as code_file:
            code_file.write(code_block)
        print(f"Saved code block to: {code_file_path}")
    except Exception as e:
        print(f"Error saving code block: {e}")
        return []
    
    print("Code block saved successfully.")
    return [] 


def generate_solution(number_of_squares: int = 301, provider: str = "claude", model: str = "sonnet-4") -> List[Tuple[int, int]]:
    """Generate a solution for packing the given number of unit squares."""
    prompt = generate_prompt(number_of_squares)
    template_path = os.path.join(os.path.dirname(__file__), 'prompts/system_prompt.j2')
    with open(template_path, 'r') as file:
        template_content = file.read()
    
    response = call_llm(
        message=prompt,
        provider=provider,
        model=model,  
        max_tokens=8000,  # Increased token limit
        temperature=0.3,  
        system_prompt= template_content
    )
    
    print("LLM Response:")
    print(response)
    
    # Save the full LLM response to a text file
    response_file_path = os.path.join(os.path.dirname(__file__), 'llm_gen/llm_response.txt')
    try:
        with open(response_file_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LLM RESPONSE FOR SQUARE PACKING PROBLEM\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {__import__('datetime').datetime.now()}\n")
            f.write(f"Provider: {provider}\n")
            f.write(f"Model: {model}\n")
            f.write(response)
            
        print(f"Full LLM response saved to: {response_file_path}")
    except Exception as e:
        print(f"Warning: Could not save LLM response to file: {e}")

    parse_solution(response)
