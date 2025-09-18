# LLM Configuration
import os
LLM_ENSEMBLE_MODELS = {'claude' : 'sonnet-4', 
                       'gemini': 'gemini-1.5-flash', 
                       'openai': 'gpt-4o-mini'}

INITIAL_PROGRAM = os.path.join(os.path.dirname(__file__), 'base_program.py')

IMPROVE_PROMPT = ""

# Evolutionary Algorithm Configuration
