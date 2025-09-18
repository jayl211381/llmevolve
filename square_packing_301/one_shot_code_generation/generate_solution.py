#!/usr/bin/env python3
"""
Main script to generate square packing solutions using LLM one-shot code generation.
This script uses the util.py functions to generate prompts, call the LLM, 
and parse the Python code from the response.
"""

import sys
import os

# Add the parent directories to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from util import generate_solution


def main():
    """Main function to generate and validate a square packing solution."""
    # Generate solution using LLM
    print("Generating solution using LLM...")
    
    generate_solution(number_of_squares=301, provider="claude", model="sonnet-4")
        

if __name__ == "__main__":
    main()