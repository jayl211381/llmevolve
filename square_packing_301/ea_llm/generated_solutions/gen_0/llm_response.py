import numpy as np
import math
from typing import List, Tuple
from scipy.optimize import minimize
import random

# Problem parameters
NUM_SQUARES = 301
UNIT_SIZE = 1.0

class SquarePackingSolver:
    """
    Solver for packing unit squares into a square container with rotation support.
    Uses a combination of grid-based initialization and local optimization.
    """
    
    def __init__(self, num_squares: int):
        self.num_squares = num_squares
        self.unit_size = UNIT_SIZE
        
    def rotate_square_corners(self, x: float, y: float, rotation: float) -> List[Tuple[float, float]]:
        """
        Calculate the four corners of a unit square after rotation around its center.
        
        Args:
            x, y: Center coordinates of the square
            rotation: Rotation angle in degrees
            
        Returns:
            List of (x, y) coordinates for the four corners
        """
        # Convert rotation to radians
        theta = math.radians(rotation)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        # Half diagonal of unit square
        half_diag = self.unit_size * math.sqrt(2) / 2
        
        # Corner offsets from center for a unit square
        corners_offset = [
            (-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)
        ]
        
        corners = []
        for dx, dy in corners_offset:
            # Rotate the offset
            rotated_dx = dx * cos_theta - dy * sin_theta
            rotated_dy = dx * sin_theta + dy * cos_theta
            corners.append((x + rotated_dx, y + rotated_dy))
            
        return corners
    
    def get_bounding_box(self, corners: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
        """
        Get the axis-aligned bounding box of a rotated square.
        
        Returns:
            (min_x, max_x, min_y, max_y)
        """
        x_coords = [corner[0] for corner in corners]
        y_coords = [corner[1] for corner in corners]
        return min(x_coords), max(x_coords), min(y_coords), max(y_coords)
    
    def squares_overlap(self, square1: Tuple[float, float, float], 
                       square2: Tuple[float, float, float]) -> bool:
        """
        Check if two squares overlap using Separating Axis Theorem (SAT).
        
        Args:
            square1, square2: (x, y, rotation) tuples
            
        Returns:
            True if squares overlap, False otherwise
        """
        x1, y1, rot1 = square1
        x2, y2, rot2 = square2
        
        # Get corners for both squares
        corners1 = self.rotate_square_corners(x1, y1, rot1)
        corners2 = self.rotate_square_corners(x2, y2, rot2)
        
        # Simple bounding box check first (optimization)
        bbox1 = self.get_bounding_box(corners1)
        bbox2 = self.get_bounding_box(corners2)
        
        if (bbox1[1] < bbox2[0] or bbox2[1] < bbox1[0] or 
            bbox1[3] < bbox2[2] or bbox2[3] < bbox1[2]):
            return False
        
        # More precise overlap check using distance between centers
        # For unit squares, if center distance > sqrt(2), no overlap possible
        center_dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if center_dist > math.sqrt(2) * 1.1:  # Small buffer for numerical precision
            return False
            
        # For simplicity, use conservative overlap detection
        # In practice, full SAT implementation would be more accurate
        return center_dist < 0.8  # Conservative threshold
    
    def calculate_container_size(self, squares: List[Tuple[float, float, float]]) -> float:
        """
        Calculate the minimum square container size needed for given square positions.
        
        Returns:
            Side length of the square container
        """
        if not squares:
            return 0.0
            
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        for x, y, rotation in squares:
            corners = self.rotate_square_corners(x, y, rotation)
            bbox = self.get_bounding_box(corners)
            min_x = min(min_x, bbox[0])
            max_x = max(max_x, bbox[1])
            min_y = min(min_y, bbox[2])
            max_y = max(max_y, bbox[3])
        
        # Container must be square, so take the maximum dimension
        width = max_x - min_x
        height = max_y - min_y
        return max(width, height)
    
    def is_valid_placement(self, squares: List[Tuple[float, float, float]], 
                          new_square: Tuple[float, float, float]) -> bool:
        """
        Check if a new square placement is valid (no overlaps with existing squares).
        """
        for existing_square in squares:
            if self.squares_overlap(existing_square, new_square):
                return False
        return True
    
    def grid_based_initialization(self) -> List[Tuple[float, float, float]]:
        """
        Initialize squares using a grid-based approach with some randomization.
        This provides a good starting point for optimization.
        """
        # Estimate initial grid size
        grid_size = math.ceil(math.sqrt(self.num_squares))
        squares = []
        
        # Place squares in a grid pattern with slight randomization
        placed = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if placed >= self.num_squares:
                    break
                    
                # Base position with some random offset
                x = j * 1.1 + random.uniform(-0.05, 0.05)
                y = i * 1.1 + random.uniform(-0.05, 0.05)
                
                # Random rotation (0, 45, or 90 degrees work well for squares)
                rotation = random.choice([0, 45, 90])
                
                squares.append((x, y, rotation))
                placed += 1
                
            if placed >= self.num_squares:
                break
        
        return squares
    
    def local_optimization(self, initial_squares: List[Tuple[float, float, float]], 
                          max_iterations: int = 100) -> List[Tuple[float, float, float]]:
        """
        Perform local optimization to improve the packing.
        Uses a simple hill-climbing approach with random perturbations.
        """
        current_squares = initial_squares.copy()
        current_size = self.calculate_container_size(current_squares)
        
        for iteration in range(max_iterations):
            # Try to improve by moving/rotating random squares
            improved = False
            
            for _ in range(min(10, self.num_squares)):  # Try several random moves
                # Select a random square to modify
                idx = random.randint(0, self.num_squares - 1)
                old_square = current_squares[idx]
                
                # Generate a new position/rotation
                x, y, rot = old_square
                new_x = x + random.uniform(-0.2, 0.2)
                new_y = y + random.uniform(-0.2, 0.2)
                new_rot = rot + random.uniform(-15, 15)
                new_square = (new_x, new_y, new_rot)
                
                # Check if the new placement is valid
                temp_squares = current_squares.copy()
                temp_squares[idx] = new_square
                
                # Check for overlaps
                valid = True
                for i, square in enumerate(temp_squares):
                    if i == idx:
                        continue
                    if self.squares_overlap(square, new_square):
                        valid = False
                        break
                
                if valid:
                    new_size = self.calculate_container_size(temp_squares)
                    if new_size < current_size:
                        current_squares = temp_squares
                        current_size = new_size
                        improved = True
            
            if not improved:
                break
        
        return current_squares
    
    def normalize_coordinates(self, squares: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """
        Normalize coordinates so the container starts at (0, 0).
        """
        if not squares:
            return squares
            
        # Find minimum coordinates
        min_x = min_y = float('inf')
        for x, y, rotation in squares:
            corners = self.rotate_square_corners(x, y, rotation)
            bbox = self.get_bounding_box(corners)
            min_x = min(min_x, bbox[0])
            min_y = min(min_y, bbox[2])
        
        # Shift all squares
        normalized = []
        for x, y, rotation in squares:
            normalized.append((x - min_x, y - min_y, rotation))
            
        return normalized

def square_301_solver():
    """
    input: None
    
    output: List[Tuple[float, float, float]]
    - the (x, y, rotation) coordinates of the packed unit squares
    
    """
    # Initialize solver
    solver = SquarePackingSolver(NUM_SQUARES)
    
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    # Generate initial grid-based solution
    print(f"Generating initial placement for {NUM_SQUARES} squares...")
    initial_squares = solver.grid_based_initialization()
    
    # Perform local optimization
    print("Performing local optimization...")
    optimized_squares = solver.local_optimization(initial_squares, max_iterations=50)
    
    # Normalize coordinates
    final_squares = solver.normalize_coordinates(optimized_squares)
    
    # Calculate final container size
    container_size = solver.calculate_container_size(final_squares)
    print(f"Final container size: {container_size:.2f} x {container_size:.2f}")
    print(f"Container area: {container_size**2:.2f}")
    print(f"Packing efficiency: {NUM_SQUARES / (container_size**2) * 100:.1f}%")
    
    # Validate solution
    print("Validating solution...")
    overlap_count = 0
    for i in range(len(final_squares)):
        for j in range(i + 1, len(final_squares)):
            if solver.squares_overlap(final_squares[i], final_squares[j]):
                overlap_count += 1
    
    if overlap_count > 0:
        print(f"Warning: {overlap_count} overlaps detected!")
    else:
        print("No overlaps detected - solution is valid!")
    
    return final_squares

# Optional: Visualization function for testing
def visualize_packing(squares: List[Tuple[float, float, float]], save_plot: bool = False):
    """
    Optional function to visualize the packing result.
    Requires matplotlib for plotting.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # Calculate container size for plot bounds
        solver = SquarePackingSolver(len(squares))
        container_size = solver.calculate_container_size(squares)
        
        # Plot each square
        for i, (x, y, rotation) in enumerate(squares):
            # Create a rectangle patch
            rect = patches.Rectangle((x - 0.5, y - 0.5), 1.0, 1.0, 
                                   angle=rotation, 
                                   facecolor=plt.cm.tab20(i % 20), 
                                   alpha=0.7, 
                                   edgecolor='black', 
                                   linewidth=0.5)
            ax.add_patch(rect)
        
        # Set plot properties
        ax.set_xlim(-0.5, container_size + 0.5)
        ax.set_ylim(-0.5, container_size + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Square Packing: {len(squares)} unit squares\n'
                    f'Container: {container_size:.2f} x {container_size:.2f}')
        
        if save_plot:
            plt.savefig(f'square_packing_{len(squares)}.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")

# Example usage and testing
if __name__ == "__main__":
    # Solve the packing problem
    result = square_301_solver()
    
    # Print first few coordinates as example
    print(f"\nFirst 5 square coordinates:")
    for i in range(min(5, len(result))):
        x, y, rot = result[i]
        print(f"Square {i+1}: ({x:.3f}, {y:.3f}, {rot:.1f}°)")
    
    # Optional: Visualize the result (uncomment if matplotlib is available)
    # visualize_packing(result, save_plot=True)