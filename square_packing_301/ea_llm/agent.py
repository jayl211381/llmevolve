# The main agent program for the EA+LLM approach

import os
import random
from config import ISLAND_COUNT, MIGRATION_INTERVAL, POPULATION_SIZE, TOURNAMENT_SIZE, GENERATIONS
from logger import LOGGER
from util import load_pdf_as_base64, load_img_as_base64, force_remove_all_files_in_directory
from config_prompt import SAMPLER_PROMPT, SAMPLER_PROMPT_SYSTEM
from config import NUM_SQUARES, LLM_ENSEMBLE_MODELS
from llm.call_llm import call_llm
from data_classes import Program
import json
import re
from typing import List, Tuple

class Optimization_agent:
    def __init__(self, island):
        # The island the agent is running on
        self.island = island
    
    def load_inspirations(self):
        # Load inspirations from the inspirations folder
        inspirations_dir = os.path.join(os.path.dirname(__file__), 'inspirations')
        inspirations = []
        if os.path.exists(inspirations_dir):
            for file_name in os.listdir(inspirations_dir):
                file_path = os.path.join(inspirations_dir, file_name)
                if not os.path.isfile(file_path):
                    continue
                    
                file_ext = file_name.lower()
                
                if file_ext.endswith('.pdf'):
                    inspirations.append(load_pdf_as_base64(file_path))
                elif file_ext.endswith(('.png', '.jpg', '.jpeg')):
                    inspirations.append(load_img_as_base64(file_path))
                elif file_ext.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        inspirations.append(f.read())
                elif file_ext.endswith(('md')):
                    continue
                else:
                    raise Exception(f"Unsupported file type in inspirations: {file_name}")
                
        return inspirations

    def prompt_sampler(self):
        # combines parent programs with inspirations to create a new prompt
        inspirations = self.load_inspirations()

        # Prepare prior programs data if we have prior programs
        prior_programs_data = []
        if hasattr(self, 'prior_programs') and self.prior_programs:
            for program in self.prior_programs:
                prior_programs_data.append({
                    'fitness_score': getattr(program, 'fitness_score', 'N/A'),
                    'side_length': getattr(program, 'side_length', 'N/A'),
                    'efficiency': getattr(program, 'efficiency', 'N/A'),
                    'overlap_penalty': getattr(program, 'overlaps', 'N/A'),
                    'llm_evaluation_response': getattr(program, 'llm_evaluation', 'N/A'),
                    'code': getattr(program, 'code', '')
                })

        # Assert that the current program must have code
        assert getattr(self.current_program, 'code', None), "Current program must have code"

        # Load and render the template with the data
        rendered_prompt = SAMPLER_PROMPT.render(
            NUM_SQUARES=NUM_SQUARES,
            PRIOR_PROGRAMS=prior_programs_data,
            
            # Current program data
            CURRENT_PROGRAM_FITNESS_SCORE=getattr(self.current_program, 'fitness_score', 'N/A'),
            CURRENT_PROGRAM_SIDE_LENGTH=getattr(self.current_program, 'side_length', 'N/A'),
            CURRENT_PROGRAM_EFFICIENCY=getattr(self.current_program, 'efficiency', 'N/A'),
            CURRENT_PROGRAM_OVERLAP_PENALTY=getattr(self.current_program, 'overlap_penalty', 'N/A'),
            CURRENT_PROGRAM_LLM_EVALUATION_RESPONSE=getattr(self.current_program, 'llm_evaluation_response', 'N/A'),
            CURRENT_PROGRAM_CODE=getattr(self.current_program, 'code', 'N/A'),
            
            INSPIRATIONS=inspirations
        )
        
        return rendered_prompt
    
    def extract_code_from_response(self, response: str, model: str) -> str:
        """Extract code snippet from LLM response."""
        # Save the full LLM response to a text file 
        response_file_path = os.path.join(os.path.dirname(__file__), 'llm_gen/llm_response.txt')
        try:
            with open(response_file_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("LLM RESPONSE FOR SQUARE PACKING PROBLEM\n")
                f.write("=" * 80 + "\n")
                f.write(f"Timestamp: {__import__('datetime').datetime.now()}\n")
                f.write(f"Model: {model}\n")
                f.write(response)
                
            print(f"Full LLM response saved to: {response_file_path}")
        except Exception as e:
            print(f"Warning: Could not save LLM response to file: {e}")
            
        self.parse_solution(response)
        return
    
    def parse_solution(response: str) -> List[Tuple[int, int]]:
        """
        Parse the LLM response to extract a single Python code block and save it to a file
        """
        
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
    
    def ensemble_generate_diff(self):
        # Generate diffs using an ensemble of LLM models
        self.model_responses = {model: [] for model in LLM_ENSEMBLE_MODELS.values()}
        for provider, model in LLM_ENSEMBLE_MODELS.items():
            response = call_llm(
                message=self.prompt_sampler(),
                provider=provider,
                model=model,
                system_prompt=SAMPLER_PROMPT_SYSTEM,
                temperature=0.7
            )
            self.model_responses[model].append(response)
            # Process the response to extract the diff
            diff = self.extract_code_from_response(response, model)

            LOGGER.info(f"Model {provider}/{model} generated diff of length {len(diff) if diff else 0}")
        return self.model_responses
    
    def evolve(self):
        # Evolve the current program using the optimization agent
        LOGGER.info(f"Evolving program... island {self.island.id}")

        # Get the current and prior best programs
        self.current_program, self.prior_programs = self.island.evolve()

        # Load prompts with program data and inspirations and generate solution
        LOGGER.info("Generating diffs using ensemble LLM models...")
        self.ensemble_generate_diff()
        return 

    def apply_diff(self):
        # Apply the generated diff to the current program to create a new program
        for diff in self.model_responses:
            self.current_program = self.apply_single_diff(self.current_program, diff)
        return self.current_program
    
    def evaluate(self):
        # Evaluate the new program by running it and measuring fitness
        pass
    
class ea_controller:
    '''
    Controller for the evolutionary algorithm with multiple islands
    '''
    def __init__(self, num_islands: int, migration_interval: int):
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.generations = GENERATIONS
        self.current_generation = 0
        self.islands = []

    def check_islands_status(self, num_islands: int):
        if len(self.islands) > 0:
            LOGGER.info("Islands already initiated")
            return

        islands_db_dir = os.path.join(os.path.dirname(__file__), 'islands_databases')
        
        if os.path.exists(islands_db_dir) and os.listdir(islands_db_dir):
            island_folders = [f for f in os.listdir(islands_db_dir) 
                            if os.path.isdir(os.path.join(islands_db_dir, f)) 
                            and f.startswith('island')]
            if island_folders:
                island_count = len(island_folders)
                if island_count == num_islands:
                    LOGGER.info(f"Found {island_count} existing islands in database")
                else:
                    LOGGER.error(f"Past islands found ({island_count}) but different from current setting ({num_islands})")
                    raise Exception("Past island numbers mismatch")
            else:
                # Create new island directories
                for i in range(num_islands):
                    island_dir = os.path.join(islands_db_dir, f'island_{i}')
                    os.makedirs(island_dir, exist_ok=True)
                LOGGER.info(f"Created {num_islands} new island directories")
        else:
            # Create islands database directory and island subdirectories
            os.makedirs(islands_db_dir, exist_ok=True)
            for i in range(num_islands):
                island_dir = os.path.join(islands_db_dir, f'island_{i}')
                os.makedirs(island_dir, exist_ok=True)
            LOGGER.info(f"Created islands database with {num_islands} islands")
                    
    def initiate_islands(self, num_islands: int, population_size: int,
                         migration_interval: int, tournament_size: int,
                         clean_run: bool = False):
        if clean_run:
            # Ask for confirmation
            LOGGER.warning("Clean run initiated. This will wipe out all existing islands database.")
            confirmation = input("Type 'yes' to confirm: ")
            if confirmation != 'yes':
                LOGGER.info("Clean run cancelled by user.")
                return
            # Wipe out all existing paths in the islands database
            islands_db_dir = os.path.join(os.path.dirname(__file__), 'islands_databases')
            if os.path.exists(islands_db_dir):
                force_remove_all_files_in_directory(islands_db_dir)
                
        self.check_islands_status(num_islands)
        islands_db_dir = os.path.join(os.path.dirname(__file__), 'islands_databases')
        
        for i in range(num_islands):
            island_folder_location = os.path.join(islands_db_dir, f'island_{i}')
            island = Island(i, island_folder_location, population_size, 
                            migration_interval, tournament_size)
            # Initialize the island population immediately
            island.initialize_population()
            self.islands.append(island)
            
        LOGGER.info(f"Initiated {len(self.islands)} islands")
            
    def sample_programs(self, program_db):
        # Sample the best programs from all islands
        best_programs = {}
        for island in self.islands:
            best_programs = island.run_tournament_selection()
            if best_programs:
                best_programs[island.id] = best_programs
        return best_programs
    
    def check_migration(self):
        # Check if it's time to migrate programs between island
        return self.generation % self.migration_interval == 0
    
    def migrate_programs(self):
        # Migrate programs between islands if it's time
        if self.check_migration():
            for island in self.islands:
                island.migrate = True
        else:
            for island in self.islands:
                island.migrate = False

    def initialize_agents(self):
        # Initialize the optimization agent for each island
        self.agents = []
        for island in self.islands:
            agent = Optimization_agent(island)
            self.agents.append(agent)

    def start_island_evolution(self):
        # Start the evolution process for each island
        for agent in self.agents:
            agent.evolve()

class Island:
    def __init__(self, island_id, island_folder_location, population_size, tournament_size, tournament_finalists_size):
        self.id = island_id
        # Location to save island data
        self.island_folder_location = island_folder_location
        # Max number of solutions in the island
        self.population_size = population_size
        # Whether to migrate programs to other islands
        self.migrate = False
        # The number of programs to sample for tournament selection
        self.tournament_size = tournament_size
        # The number of finalists output from tournament selection
        self.tournament_finalists_size = tournament_finalists_size
        self.population = []

    def initialize_population(self):
        # Initialize the island population
        base_run_status = self.check_if_base_run()
        if not base_run_status:
            # loop through all folders in the island base folder and add programs
            for root, dirs, files in os.walk(self.island_folder_location):
                for file_name in files:
                    if file_name.endswith('.py'):
                        program_path = os.path.join(root, file_name)
                        self.populate_solution(program_path)
        LOGGER.info(f"Initialized island {self.id} with {len(self.population)} programs")
    
    def populate_solution(self, program_path):
        # Create a program data instance from the source path
        with open(program_path, 'r', encoding='utf-8') as f:
            # Partial attribute assignment for the base solution
            code = f.read()
            file_name = program_path.split("\\")[-1]
            if file_name == "base_solution.py":
                base_program = Program(id = file_name,
                                       code=code,
                                       solution_path=program_path)
                self.add_program(base_program)
            else:
                # Create a program data instance for non-base solutions
                # load the json eval results if available
                json_path = program_path.replace('.py', '_results.json')
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        eval_results = json.load(f)
                try:
                    program = Program(
                        id = file_name,
                        code=code,
                        solution_path=program_path,
                        num_squares=eval_results.get('num_squares', NUM_SQUARES),
                        overlaps=eval_results.get('overlaps', 0),
                        is_valid=eval_results.get('is_valid', False),
                        side_length=eval_results.get('side_length', 0.0),
                        efficiency=eval_results.get('efficiency', 0.0),
                        llm_evaluation=eval_results.get('llm_evaluation'),
                        fitness_score=eval_results.get('fitness_score')
                    )
                    self.add_program(program)
                except Exception as e:
                    LOGGER.error(f"Error loading program from {program_path}: {e}")

        pass
    
    def check_if_base_run(self):
        # Check if this is the initial run with no programs
        # if not self population and no solutions exist in the island folder
        if len(self.population) == 0 and not os.listdir(self.island_folder_location):
            print(f"Initial run, setting base solution for island: {self.id}")
            # Copy the base solution to the island
            base_solution_path = os.path.join(os.path.dirname(__file__), 'base_solution')
            if os.path.exists(base_solution_path):
                # Copy all files in the folder to the island base folder
                for file_name in os.listdir(base_solution_path):
                    # Process files that are not README.md
                    if file_name.lower() != 'readme.md':
                        src_file = os.path.join(base_solution_path, file_name)
                        dst_file = os.path.join(self.island_folder_location, 'base_solution/' + file_name)
                    # Create destination folder if not exists
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    # Copy the file
                    if os.path.isfile(src_file):
                        with open(src_file, 'r', encoding='utf-8') as src_f, open(dst_file, 'w', encoding='utf-8') as dst_f:
                            dst_f.write(src_f.read())
            # Add the base solution as the first program in the population
            base_program_path = os.path.join(self.island_folder_location, 'base_solution', 'solution.py')
            if os.path.isfile(base_program_path):
                self.populate_solution(base_program_path)
                return True
            
        return False
        
    def evolve(self):
        self.initialize_population()
        # Evolve the island population for one generation

        self.run_tournament_selection()
        current_program, prior_programs = self.current_prior_program_split()
        return current_program, prior_programs

    def run_tournament_selection(self):
        # Run tournament selection to choose the best programs within the island
        self.best_programs = []
        if len(self.population) < self.tournament_size:
            LOGGER.warning("Not enough programs for full tournament, using all available")
            self.best_programs = self.population
        else:
            self.best_programs = self._select_programs_for_tournament()

    def current_prior_program_split(self):
        # Take the best programs, randomly select one as current, rest as prior
        if self.best_programs:
            current_program = random.choice(self.best_programs)
            prior_programs = [prog for prog in self.best_programs if prog != current_program]
            return current_program, prior_programs
        else:
            raise Exception(f"No best programs available found in island {self.id}")
            
    def _select_programs_for_tournament(self):
        # Randomly select programs for the tournament
        return random.sample(self.population, self.tournament_size)

    def migrate_programs(self, target_island):
        # Migrate some programs to another island, replacing the worst performing solution
        if not self.population:
            LOGGER.warning("No programs to migrate from this island")
            self.migrate = False
            return

        if self.best_programs:
            # randomly select one of the best programs to migrate
            best_program = random.choice(self.best_programs)
            # Replace the worst program in the target island
            target_island._remove_worst_program()
            target_island.add_program(best_program)
            LOGGER.info(f"Migrated program from island {self.id} to island {target_island.id}")
        else:
            LOGGER.warning(f"No best programs found to migrate from island {self.id}")
        # Set migrate flag to False after migration
        self.migrate = False

    def add_program(self, program):
        # Add a new program to the island population
        if len(self.population) >= self.population_size:
            LOGGER.info("Population full, removing worst program")
            self._remove_worst_program()
        self.population.append(program)
        
    def _remove_worst_program(self):
        # Remove the worst performing program from the population
        if self.population:
            worst_program = min(self.population, key=lambda p: p.fitness)
            self.population.remove(worst_program)
    
    def _save_program_to_disk(self, program):
        # Save the program to the island's folder
        program_path = os.path.join(self.island_folder_location, program.id)
        with open(program_path, 'w') as f:
            f.write(program.code)
            
if __name__ == '__main__':
    controller = ea_controller(num_islands=ISLAND_COUNT, migration_interval=MIGRATION_INTERVAL)
    
    
    controller.initiate_islands(num_islands=ISLAND_COUNT, population_size=POPULATION_SIZE,
                                 migration_interval=MIGRATION_INTERVAL, tournament_size=TOURNAMENT_SIZE,
                                 clean_run=False)
    
    controller.initialize_agents()
    controller.start_island_evolution()