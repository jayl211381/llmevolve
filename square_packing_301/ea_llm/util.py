import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from llm.call_llm import call_llm
from jinja2 import Template
import os
from typing import List, Tuple
from logger import LOGGER

def load_prompt_template(template_name: str, context: dict = None) -> str:
    """Load and render a Jinja2 template with context."""
    try:
        template_path = os.path.join(os.path.dirname(__file__), template_name)
        with open(template_path, 'r', encoding='utf-8') as file:
            template_content = file.read()
        
        # If context provided, render the template
        if context:
            template = Template(template_content)
            return template.render(context)
        else:
            return template_content
            
    except FileNotFoundError:
        LOGGER.error(f"Template not found: {template_path}")
        return ""
    except Exception as e:
        LOGGER.error(f"Error loading template {template_name}: {e}")
        return ""

def parse_solution(response: str, response_path: str) -> List[Tuple[int, int]]:
    """
    Parse the LLM response to extract a single Python code block and save it to a file
    """
    import re
    
    # Extract Python code blocks from the response
    code_blocks = re.findall(r'```python\s*(.*?)\s*```', response, re.DOTALL)
    
    if not code_blocks:
        LOGGER.error("No code blocks found")
        return []
    
    if len(code_blocks) > 1:
        LOGGER.warning(f"Found {len(code_blocks)} code blocks, using first")
    
    # Save the first (and expected only) code block
    code_block = code_blocks[0]

    try:
        with open(response_path, 'w', encoding='utf-8') as code_file:
            code_file.write(code_block)
    except Exception as e:
        LOGGER.error(f"Error saving code: {e}")
        return []
    return [] 


def generate_solution(provider: str, model: str, prompt: str, 
                      system_prompt: str, response_path: str) -> List[Tuple[int, int]]:
    """Generate a solution for packing the given number of unit squares."""

    LOGGER.info(f"Calling LLM: {provider}/{model}")
    response = call_llm(
        message=prompt,
        provider=provider,
        model=model,  
        max_tokens=8000,  # Increased token limit
        temperature=0.3,  
        system_prompt=system_prompt
    )
    
    # Save the full LLM response to a text file
    try:
        with open(response_path, 'w', encoding='utf-8') as f:
            f.write(response)
    except Exception as e:
        LOGGER.error(f"Could not save LLM response: {e}")
    # Replace the txt file in path with py file
    response_path = response_path.replace('.txt', '.py')
    parse_solution(response, response_path)
