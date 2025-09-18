"""
MAP-Elites Algorithm for One Max problem.
Maintains diverse elite solutions across the behavioral grid.

Behavioral dimensions:
1) Max run length: Longest consecutive 1s
2) Transition count: Number of bit changes (01 or 10)
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


class Individual:
    """Individual solution with genome, fitness, and behavioral characteristics."""
    def __init__(self, genome: List[int]):
        self.genome = genome
        self.fitness = sum(genome)  # Count of 1s
        self.behavior = self._calculate_behavior() 
    
    def _calculate_behavior(self) -> Tuple[int, int]:
        """Calculate behavioral descriptors: (max_run, transitions)."""
        genome = self.genome
        
        # Max consecutive run of 1s
        max_run = 0
        current_run = 0
        for bit in genome:
            if bit == 1:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        
        # Number of transitions (bit flips)
        transitions = 0
        for i in range(len(genome) - 1):
            if genome[i] != genome[i + 1]:
                transitions += 1
        
        return (max_run, transitions)
    
    def __repr__(self):
        return f"Individual(fitness={self.fitness}, behavior={self.behavior})"


class MAPElites:
    """MAP-Elites: maintains archive of elite solutions across behavioral niches."""
    
    def __init__(self, genome_length: int = 20, mutation_rate: float = 0.1, crossover_rate: float = 0.7):
        self.genome_length = genome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Behavioral space bounds
        self.behavior_bounds = [
            (0, genome_length),      # Max run: 0 to genome_length
            (0, genome_length - 1),  # Transitions: 0 to genome_length-1
        ]
        
        # Grid size equals behavior bounds (direct mapping)
        self.grid_size = [
            genome_length + 1,
            genome_length
        ]
        
        self.archive = {}  # Maps grid coordinates to elite Individual
    
    def _random_individual(self) -> Individual:
        """Generate random individual."""
        genome = [random.randint(0, 1) for _ in range(self.genome_length)]
        return Individual(genome)
    
    def _mutate(self, individual: Individual) -> Individual:
        """Bit-flip mutation."""
        genome = individual.genome[:]
        for i in range(len(genome)):
            if random.random() < self.mutation_rate:
                genome[i] = 1 - genome[i]
        return Individual(genome)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Single-point crossover."""
        if random.random() > self.crossover_rate:
            return Individual(parent1.genome[:])
        
        point = random.randint(1, self.genome_length - 1)
        child_genome = parent1.genome[:point] + parent2.genome[point:]
        return Individual(child_genome)
    
    def _add_to_archive(self, individual: Individual) -> bool:
        """Add individual to archive if it's elite for its niche."""
        coords = individual.behavior  # Direct mapping 1 to 1
        
        if coords not in self.archive or individual.fitness > self.archive[coords].fitness:
            self.archive[coords] = individual
            return True
        return False
    
    def _select_parent(self) -> Individual:
        """Select random parent from archive."""
        if not self.archive:
            return self._random_individual()
        return random.choice(list(self.archive.values()))
    
    def run(self, iterations: int = 3000, initial_samples: int = 100, verbose: bool = True) -> Dict:
        """Run MAP-Elites algorithm and return results."""
        if verbose:
            print("Starting MAP-Elites Algorithm")
            print(f"Genome length: {self.genome_length}, Grid size: {self.grid_size}, Total niches: {np.prod(self.grid_size)}")
            print("-" * 40)
        
        # Initialize archive with random individuals
        for _ in range(initial_samples):
            individual = self._random_individual()
            self._add_to_archive(individual)
        
        # Main evolutionary loop
        for iteration in range(iterations):
            if len(self.archive) >= 2:
                parent1 = self._select_parent()
                parent2 = self._select_parent()
                child = self._crossover(parent1, parent2)
            else:
                child = self._random_individual()
            
            child = self._mutate(child)
            child = self._mutate(child)
            self._add_to_archive(child)
            
            # Progress reporting
            if verbose and iteration % 1000 == 0:
                max_fitness = max(ind.fitness for ind in self.archive.values()) if self.archive else 0
                coverage = len(self.archive) / np.prod(self.grid_size)
                qd_score = sum(ind.fitness for ind in self.archive.values())
                print(f"Iteration {iteration}: Archive size = {len(self.archive)}, "
                      f"Max fitness = {max_fitness}, Coverage = {coverage:.3f}, QD Score = {qd_score:.0f}")
        
        # Calculate final statistics
        if self.archive:
            max_fitness = max(ind.fitness for ind in self.archive.values())
            coverage = len(self.archive) / np.prod(self.grid_size)
            qd_score = sum(ind.fitness for ind in self.archive.values())
        else:
            max_fitness = coverage = qd_score = 0
        
        if verbose:
            print("-" * 40)
            print("Results:")
            print(f"Archive: {len(self.archive)} solutions, Max fitness: {max_fitness}/{self.genome_length}")
            print(f"Coverage: {coverage:.3f}, Quality Diversity Score: {qd_score:.0f}")
            
        # Return comprehensive results for further analysis
        return {
            "archive": self.archive,
            "archive_size": len(self.archive),
            "max_fitness": max_fitness,
            "coverage": coverage,
            "qd_score": qd_score
        }
    
    def plot_niche_grid(self, title: str = "MAP-Elites Final Solutions Niche Grid", save_path: str = None):
        """Plot niche grid showing fitness distribution across behavioral space."""
        if not self.archive:
            print("No solutions in archive to plot!")
            return
            
        print(f"\nPlotting {len(self.archive)} elite solutions...")
        
        # Create fitness grid
        fitness_grid = np.full(self.grid_size, np.nan)
        for (x, y), individual in self.archive.items():
            fitness_grid[x, y] = individual.fitness
        
        # Create the plot with larger figure size
        plt.figure(figsize=(14, 10))
        
        # Create heatmap with viridis colormap (good for showing gradients)
        im = plt.imshow(fitness_grid.T, cmap='viridis', origin='lower', 
                       vmin=0, vmax=self.genome_length, aspect='auto',
                       interpolation='nearest')
        
        # Add colorbar with proper labeling
        cbar = plt.colorbar(im, shrink=0.8, pad=0.02)
        cbar.set_label('Fitness (Number of 1s)', fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)
        
        # Set labels and title with larger fonts
        plt.xlabel('Max Run Length (Behavioral Dimension 1)', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Transitions (Behavioral Dimension 2)', fontsize=14, fontweight='bold')
        plt.title(f'{title}\n'
                 f'Archive: {len(self.archive)}/{np.prod(self.grid_size)} niches filled '
                 f'({len(self.archive)/np.prod(self.grid_size)*100:.1f}% coverage)', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Add subtle grid for better readability
        plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        
        # Create meaningful tick labels - now using direct mapping
        x_step = max(1, self.grid_size[0] // 10)  # Show ~10 ticks max
        y_step = max(1, self.grid_size[1] // 10)
        
        x_ticks = np.arange(0, self.grid_size[0], x_step)
        y_ticks = np.arange(0, self.grid_size[1], y_step)
        
        # Since we have direct mapping, grid coordinates = behavioral values
        x_labels = [str(int(tick)) for tick in x_ticks]
        y_labels = [str(int(tick)) for tick in y_ticks]
        
        plt.xticks(x_ticks, x_labels, fontsize=12)
        plt.yticks(y_ticks, y_labels, fontsize=12)
        
        # Add text annotations for all solutions
        max_fitness = max(ind.fitness for ind in self.archive.values())
        
        annotation_count = 0
        for (x, y), individual in self.archive.items():
            # Use white text for dark backgrounds, black for light
            text_color = 'white' if individual.fitness > max_fitness * 0.5 else 'black'
            plt.annotate(f'{int(individual.fitness)}', (x, y), 
                       color=text_color, fontweight='bold', fontsize=8,
                       ha='center', va='center')
            annotation_count += 1
        
        # Add a legend explaining the visualization
        legend_text = (f"• Each cell = one behavioral niche\n"
                      f"• Numbers = fitness of all solutions\n")
        
        plt.text(0.02, 0.98, legend_text, transform=plt.gca().transAxes,
                fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='white', alpha=0.8, edgecolor='gray'))
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        
        # Print detailed statistics about niche distribution
        print(f"\nNiche Grid Statistics:")
        print(f"Grid dimensions: {self.grid_size[0]} x {self.grid_size[1]} = {np.prod(self.grid_size)} total niches")
        print(f"Filled niches: {len(self.archive)} ({len(self.archive)/np.prod(self.grid_size)*100:.1f}%)")
        print(f"Empty niches: {np.prod(self.grid_size) - len(self.archive)} ({(1 - len(self.archive)/np.prod(self.grid_size))*100:.1f}%)")


if __name__ == "__main__":
    # Set hyperparameters
    GENOME_LENGTH = 15
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.7
    ITERATIONS = 2000
    INITIAL_SAMPLES = 100
    
    print("MAP-Elites Test")
    print("=" * 20)

    map_elites = MAPElites(genome_length=GENOME_LENGTH, mutation_rate=MUTATION_RATE, crossover_rate=CROSSOVER_RATE)
    results = map_elites.run(iterations=ITERATIONS, initial_samples=INITIAL_SAMPLES, verbose=True)

    success = results['max_fitness'] >= map_elites.genome_length
    print(f"\nResult: {'SUCCESS' if success else 'PARTIAL'}")
    print(f"Found {results['archive_size']} solutions")
    
    map_elites.plot_niche_grid("MAP-Elites Solutions")
    
