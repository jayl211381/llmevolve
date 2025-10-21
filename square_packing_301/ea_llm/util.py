import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from llm.call_llm import call_llm
from jinja2 import Template
import os
from typing import List, Tuple
from logger import LOGGER
import fitz
import base64
import stat
import shutil

def force_remove_all_files_in_directory(directory_path: str) -> bool:
    """
    Aggressively remove all files and directories in the given path.
    Uses multiple strategies to handle Windows/OneDrive permission issues.
    
    Args:
        directory_path: Path to the directory to remove completely
        
    Returns:
        bool: True if removal was successful, False otherwise
    """
    if not os.path.exists(directory_path):
        LOGGER.info(f"Directory {directory_path} does not exist, nothing to remove")
        return True
    
    def force_remove_readonly(func, path, exc):
        """Error handler to force removal of read-only files"""
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except:
            pass  # Continue even if chmod fails
    
    # Try multiple removal strategies
    removed = False
    
    # Strategy 1: Normal removal
    try:
        shutil.rmtree(directory_path)
        LOGGER.info(f"Cleaned existing directory: {directory_path}")
        removed = True
    except:
        pass
    
    # Strategy 2: Force removal with permission changes
    if not removed and os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path, onerror=force_remove_readonly)
            LOGGER.info(f"Force cleaned existing directory: {directory_path}")
            removed = True
        except:
            pass
    
    # Strategy 3: Manual brute force removal
    if not removed and os.path.exists(directory_path):
        LOGGER.info(f"Attempting aggressive manual cleanup of: {directory_path}")
        for root, dirs, files in os.walk(directory_path, topdown=False):
            # Force remove all files
            for name in files:
                file_path = os.path.join(root, name)
                try:
                    os.chmod(file_path, stat.S_IWRITE)
                    os.remove(file_path)
                except:
                    try:
                        # Alternative: mark for deletion and try again
                        os.chmod(file_path, stat.S_IWRITE | stat.S_IREAD)
                        os.remove(file_path)
                    except:
                        pass  # Skip individual failures but keep trying
            
            # Force remove all directories
            for name in dirs:
                dir_path = os.path.join(root, name)
                try:
                    os.chmod(dir_path, stat.S_IWRITE)
                    os.rmdir(dir_path)
                except:
                    pass  # Skip individual failures
        
        # Finally try to remove the root directory
        try:
            os.chmod(directory_path, stat.S_IWRITE)
            os.rmdir(directory_path)
            removed = True
        except:
            pass
        
        LOGGER.info("Force cleanup completed")
    
    return removed


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

def load_pdf_as_base64(pdf_path: str) -> str:
    """
    Load a PDF file and convert it to a base64-encoded string.
    Extracts text only.
    """
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_data = pdf_file.read()
            encoded_pdf = base64.b64encode(pdf_data).decode('utf-8')
            return encoded_pdf
    except FileNotFoundError:
        LOGGER.error(f"PDF file not found: {pdf_path}")
        return ""
    except Exception as e:
        LOGGER.error(f"Error loading PDF {pdf_path}: {e}")
        return ""

def load_img_as_base64(img_path: str) -> str:
    """
    Load an image file and convert it to a base64-encoded string.
    """
    try:
        with open(img_path, 'rb') as img_file:
            img_data = img_file.read()
            encoded_img = base64.b64encode(img_data).decode('utf-8')
            return encoded_img
    except FileNotFoundError:
        LOGGER.error(f"Image file not found: {img_path}")
        return ""
    except Exception as e:
        LOGGER.error(f"Error loading image {img_path}: {e}")
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
