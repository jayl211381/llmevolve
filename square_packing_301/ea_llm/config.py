# LLM Configuration
import os

## Basic Configuration
#--------------------------------------------------------------------------------------------------
NUM_SQUARES = 301

## LLM ensemble Configuration
#--------------------------------------------------------------------------------------------------
LLM_ENSEMBLE_MODELS = {'claude' : 'sonnet-4', 
                       'gemini': 'gemini-1.5-flash', 
                       'openai': 'gpt-4o-mini'}

INITIAL_PROGRAM = os.path.join(os.path.dirname(__file__), 'base_program.py')

## Evolutionary Algorithm Configuration
#----------------------------------------------------------------------------------------------------

# Island model parameters
ISLAND_COUNT = 5
MIGRATION_INTERVAL = 10  # Number of generations between migrations

# Population parameters
POPULATION_SIZE = 20
MUTATION_RATE = 0.3
TOURNAMENT_SIZE = 5
GENERATIONS = 1000

# map elite model parameters
# MUTATION_RATE = 0.1
# CROSSOVER_RATE = 0.7
# ITERATIONS = 2000
# INITIAL_SAMPLES = 100