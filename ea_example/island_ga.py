"""
Island-Based Genetic Algorithm example for solving the One Max problem.
Find a binary string (a string of 0s and 1s) of a given length that maximizes the number of 1s.

multiple islands - evolving separately - migration periodically to replace the worst individuals
isolation of islands maintain diversity
migrations promote knowledge sharing
"""
 
import random
import numpy as np
from typing import List


class Individual:
    """Represents an individual solution"""
    def __init__(self, genome: List[int]):
        self.genome = genome
        self.fitness = sum(genome)
    
    def __repr__(self):
        return f"Individual(fitness={self.fitness})"


class Island:
    """Single island population"""
    
    def __init__(self, population_size: int, genome_length: int, 
                 mutation_rate: float = 0.02, crossover_rate: float = 0.8):
        self.population_size = population_size
        self.genome_length = genome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = self._init_population()
    
    def _init_population(self) -> List[Individual]:
        """Create random initial population"""
        
        population = []
        for _ in range(self.population_size):
            population.append(Individual([random.randint(0, 1) for _ in range(self.genome_length)]))
            
        return population
    
    def select_parent(self) -> Individual:
        """
        Tournament selection
        random sample - competition - winner selection
        """
        tournament = random.sample(self.population, 3)
        return max(tournament, key=lambda x: x.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        Single-point crossover
        choose crossover point - split and combine - create child
        """
        # Only crossover if random chance is met
        if random.random() > self.crossover_rate:
            return Individual(parent1.genome[:])
        
        point = random.randint(1, self.genome_length - 1)
        # Create child genome by combining the two halves of the parents
        child_genome = parent1.genome[:point] + parent2.genome[point:]
        return Individual(child_genome)
    
    def mutate(self, individual: Individual) -> Individual:
        """
        Bit-flip mutation
        flip bits in the genome with a given mutation rate
        """
        genome = individual.genome[:]
        for i in range(len(genome)):
            if random.random() < self.mutation_rate:
                genome[i] = 1 - genome[i]
        return Individual(genome)
    
    def evolve_generation(self):
        """Evolve population for one generation"""
        new_population = []
        
        # Keep best individual
        best = max(self.population, key=lambda x: x.fitness)
        new_population.append(Individual(best.genome[:]))
        
        # Generate offspring
        while len(new_population) < self.population_size:
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        
        self.population = new_population
    
    def get_best(self) -> Individual:
        """Get best individual in population"""
        return max(self.population, key=lambda x: x.fitness)
    
    def get_best_individuals(self, n: int) -> List[Individual]:
        """Get n best individuals for migration"""
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        return [Individual(ind.genome[:]) for ind in sorted_pop[:n]]
    
    def add_immigrants(self, immigrants: List[Individual]):
        """Add immigrants, replace worst individuals"""
        if not immigrants:
            return
        
        # Sort population by fitness (worst first)
        self.population.sort(key=lambda x: x.fitness)
        
        # Replace worst individuals with immigrants
        for i, immigrant in enumerate(immigrants):
            if i < len(self.population):
                self.population[i] = Individual(immigrant.genome[:])


class IslandGA:
    """Island-based genetic algorithm"""

    def __init__(self, num_islands: int = 4, population_size: int = 50,
                 genome_length: int = 20, migration_interval: int = 10):
        self.num_islands = num_islands
        self.genome_length = genome_length
        self.migration_interval = migration_interval
        
        # Create islands with slightly different parameters
        self.islands = []
        for i in range(num_islands):
            mutation_rate = 0.01 + (i * 0.01)  # 0.01 to 0.04
            crossover_rate = 0.7 + (i * 0.05)  # 0.7 to 0.85
            island = Island(population_size, genome_length, mutation_rate, crossover_rate)
            self.islands.append(island)
    
    def migrate(self):
        """
        Migrate best individuals between islands (ring topology)
        Island 0 → Island 1 → Island 2 → Island 3 → Island 0
        """
        migration_size = 2
        emigrants = []
        
        # Collect emigrants from each island
        for island in self.islands:
            emigrants.append(island.get_best_individuals(migration_size))
        
        # Distribute emigrants (ring topology)
        for i, island in enumerate(self.islands):
            next_island = (i + 1) % self.num_islands
            island.add_immigrants(emigrants[next_island])
    
    def evolve(self, generations: int = 100, verbose: bool = True) -> dict:
        """Run the island GA"""
        if verbose:
            # verbose prints training progress to console
            print(f"Starting Island GA with {self.num_islands} islands")
            print(f"Population per island: {self.islands[0].population_size}")
            print(f"Genome length: {self.genome_length}")
            print("-" * 50)
        
        best_fitness_history = []
        
        for generation in range(generations):
            # Evolve each island
            for island in self.islands:
                island.evolve_generation()
            
            # Migration
            if generation > 0 and generation % self.migration_interval == 0:
                self.migrate()
                if verbose:
                    print(f"Generation {generation}: Migration occurred")
            
            # Track best fitness
            global_best = max(island.get_best().fitness for island in self.islands)
            best_fitness_history.append(global_best)
            
            # Progress output
            if verbose and generation % 5 == 0:
                avg_best = np.mean([island.get_best().fitness for island in self.islands])
                print(f"Generation {generation}: Best = {global_best}, Avg = {avg_best:.1f}")
            
            # Early stopping if optimal found
            if global_best == self.genome_length:
                if verbose:
                    print(f"Optimal solution found at generation {generation}!")
                break
        
        # Final results
        island_bests = [island.get_best() for island in self.islands]
        global_best = max(island_bests, key=lambda x: x.fitness)
        
        if verbose:
            print("-" * 50)
            print("Final Results:")
            for i, best in enumerate(island_bests):
                genome_str = ''.join(map(str, best.genome))
                print(f"Island {i}: {genome_str} (fitness: {best.fitness})")
            print(f"Global Best: {global_best.fitness}/{self.genome_length}")
        
        return {
            "global_best": global_best,
            "island_bests": island_bests,
            "fitness_history": best_fitness_history,
            "success": global_best.fitness == self.genome_length
        }


if __name__ == "__main__":
    # Set hyperparameters
    NUM_ISLANDS = 3
    POPULATION_SIZE = 20
    GENOME_LENGTH = 20
    MIGRATION_INTERVAL = 5
    GENERATIONS = 50

    # Create Island GA instance
    island_ga = IslandGA(
        num_islands=NUM_ISLANDS,
        population_size=POPULATION_SIZE,
        genome_length=GENOME_LENGTH,
        migration_interval=MIGRATION_INTERVAL
    )

    result = island_ga.evolve(generations=GENERATIONS, verbose=True)
    if result['success']:
        print("SUCCESS! Found optimal solution!")
    else:
        print("FAILURE! Did not find optimal solution.")
