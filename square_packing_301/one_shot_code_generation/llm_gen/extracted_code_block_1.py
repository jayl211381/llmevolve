import numpy as np
import math
from scipy.optimize import minimize, differential_evolution
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import random

# Problem parameters
NUM_SQUARES = 301
UNIT_SIZE = 1.0

class SquarePackingOptimizer:
    def __init__(self, num_squares: int):
        self.num_squares = num_squares
        self.unit_size = UNIT_SIZE
        # Theoretical minimum container size (lower bound)
        self.min_container_size = math.sqrt(num_squares * self.unit_size**2)
        
    def rotate_square_corners(self, x: float, y: float, rotation: float) -> List[Tuple[float, float]]:
        """
        Calculate the four corners of a unit square after rotation.
        
        Args:
            x, y: Center coordinates of the square
            rotation: Rotation angle in degrees
        
        Returns:
            List of (x, y) coordinates for the four corners
        """
        angle_rad = math.radians(rotation)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        # Half diagonal of unit square
        half_diag = self.unit_size / 2
        
        # Original corners relative to center (before rotation)
        corners = [(-half_diag, -half_diag), (half_diag, -half_diag), 
                  (half_diag, half_diag), (-half_diag, half_diag)]
        
        # Apply rotation transformation
        rotated_corners = []
        for cx, cy in corners:
            rx = x + cx * cos_a - cy * sin_a
            ry = y + cx * sin_a + cy * cos_a
            rotated_corners.append((rx, ry))
            
        return rotated_corners
    
    def get_bounding_box(self, x: float, y: float, rotation: float) -> Tuple[float, float, float, float]:
        """
        Get the axis-aligned bounding box of a rotated square.
        
        Returns:
            (min_x, min_y, max_x, max_y)
        """
        corners = self.rotate_square_corners(x, y, rotation)
        xs, ys = zip(*corners)
        return min(xs), min(ys), max(xs), max(ys)
    
    def squares_overlap(self, x1: float, y1: float, rot1: float, 
                       x2: float, y2: float, rot2: float) -> bool:
        """
        Check if two rotated squares overlap using Separating Axes Theorem (SAT).
        """
        corners1 = self.rotate_square_corners(x1, y1, rot1)
        corners2 = self.rotate_square_corners(x2, y2, rot2)
        
        # Get axes to test (perpendicular to edges of both squares)
        axes = []
        
        # Axes for first square
        for i in range(4):
            edge = (corners1[(i+1)%4][0] - corners1[i][0], 
                   corners1[(i+1)%4][1] - corners1[i][1])
            # Perpendicular axis
            axes.append((-edge[1], edge[0]))
        
        # Axes for second square
        for i in range(4):
            edge = (corners2[(i+1)%4][0] - corners2[i][0], 
                   corners2[(i+1)%4][1] - corners2[i][1])
            # Perpendicular axis
            axes.append((-edge[1], edge[0]))
        
        # Test each axis
        for axis in axes:
            # Normalize axis
            length = math.sqrt(axis[0]**2 + axis[1]**2)
            if length == 0:
                continue
            axis = (axis[0]/length, axis[1]/length)
            
            # Project both squares onto this axis
            proj1 = [corner[0]*axis[0] + corner[1]*axis[1] for corner in corners1]
            proj2 = [corner[0]*axis[0] + corner[1]*axis[1] for corner in corners2]
            
            min1, max1 = min(proj1), max(proj1)
            min2, max2 = min(proj2), max(proj2)
            
            # Check for separation
            if max1 < min2 or max2 < min1:
                return False  # Separated on this axis
        
        return True  # No separating axis found, squares overlap
    
    def is_within_container(self, x: float, y: float, rotation: float, 
                           container_size: float) -> bool:
        """
        Check if a rotated square is completely within the square container.
        """
        min_x, min_y, max_x, max_y = self.get_bounding_box(x, y, rotation)
        return (min_x >= 0 and min_y >= 0 and 
                max_x <= container_size and max_y <= container_size)
    
    def calculate_container_size_needed(self, positions: List[Tuple[float, float, float]]) -> float:
        """
        Calculate the minimum square container size needed for given positions.
        """
        if not positions:
            return 0
        
        max_coord = 0
        for x, y, rotation in positions:
            min_x, min_y, max_x, max_y = self.get_bounding_box(x, y, rotation)
            max_coord = max(max_coord, max_x, max_y)
        
        return max_coord
    
    def grid_based_packing(self, container_size: float) -> List[Tuple[float, float, float]]:
        """
        Attempt to pack squares using a grid-based approach with some rotation.
        """
        positions = []
        
        # Calculate grid spacing
        grid_size = int(math.ceil(math.sqrt(self.num_squares)))
        spacing = container_size / grid_size
        
        placed = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if placed >= self.num_squares:
                    break
                
                # Grid position with some offset to center
                x = (i + 0.5) * spacing
                y = (j + 0.5) * spacing
                
                # Try different rotations
                rotations_to_try = [0, 15, 30, 45]
                placed_square = False
                
                for rotation in rotations_to_try:
                    if self.is_within_container(x, y, rotation, container_size):
                        # Check for overlaps with existing squares
                        overlap = False
                        for px, py, prot in positions:
                            if self.squares_overlap(x, y, rotation, px, py, prot):
                                overlap = True
                                break
                        
                        if not overlap:
                            positions.append((x, y, rotation))
                            placed += 1
                            placed_square = True
                            break
                
                if not placed_square:
                    # Try to place without rotation as fallback
                    if self.is_within_container(x, y, 0, container_size):
                        overlap = False
                        for px, py, prot in positions:
                            if self.squares_overlap(x, y, 0, px, py, prot):
                                overlap = True
                                break
                        
                        if not overlap:
                            positions.append((x, y, 0))
                            placed += 1
            
            if placed >= self.num_squares:
                break
        
        return positions
    
    def optimize_positions(self, initial_positions: List[Tuple[float, float, float]], 
                          container_size: float) -> List[Tuple[float, float, float]]:
        """
        Optimize the positions using local search to reduce container size.
        """
        positions = initial_positions.copy()
        
        # Try to improve positions iteratively
        for iteration in range(50):  # Limited iterations for performance
            improved = False
            
            for i in range(len(positions)):
                x, y, rotation = positions[i]
                
                # Try small perturbations
                for dx in [-0.1, 0, 0.1]:
                    for dy in [-0.1, 0, 0.1]:
                        for drot in [-5, 0, 5]:
                            new_x = x + dx
                            new_y = y + dy
                            new_rotation = (rotation + drot) % 360
                            
                            # Check if new position is valid
                            if not self.is_within_container(new_x, new_y, new_rotation, container_size):
                                continue
                            
                            # Check for overlaps with other squares
                            overlap = False
                            for j, (px, py, prot) in enumerate(positions):
                                if i != j and self.squares_overlap(new_x, new_y, new_rotation, px, py, prot):
                                    overlap = True
                                    break
                            
                            if not overlap:
                                positions[i] = (new_x, new_y, new_rotation)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return positions
    
    def solve(self) -> List[Tuple[float, float, float]]:
        """
        Main solving function that tries different container sizes.
        """
        # Start with a reasonable container size estimate
        best_container_size = self.min_container_size * 1.2  # 20% larger than theoretical minimum
        best_positions = []
        
        # Try different container sizes
        for size_multiplier in [1.15, 1.2, 1.25, 1.3, 1.35, 1.4]:
            container_size = self.min_container_size * size_multiplier
            
            # Try grid-based packing
            positions = self.grid_based_packing(container_size)
            
            if len(positions) == self.num_squares:
                # All squares placed successfully
                # Try to optimize positions
                optimized_positions = self.optimize_positions(positions, container_size)
                
                # Calculate actual container size needed
                actual_size = self.calculate_container_size_needed(optimized_positions)
                
                if actual_size <= container_size and (not best_positions or actual_size < best_container_size):
                    best_container_size = actual_size
                    best_positions = optimized_positions
                    break
        
        # If no solution found, try with larger container
        if not best_positions:
            container_size = self.min_container_size * 1.5
            positions = self.grid_based_packing(container_size)
            if positions:
                best_positions = positions[:self.num_squares]  # Take first num_squares positions
        
        return best_positions

def square_301_solver():
    """
    input: None
    
    output: List[Tuple[float, float, float]]
    - the (x, y, rotation) coordinates of the packed unit squares
    
    """
    square_coordinates = []
    
    # Initialize the optimizer
    optimizer = SquarePackingOptimizer(NUM_SQUARES)
    
    # Solve the packing problem
    square_coordinates = optimizer.solve()
    
    # Ensure we have exactly the right number of squares
    if len(square_coordinates) > NUM_SQUARES:
        square_coordinates = square_coordinates[:NUM_SQUARES]
    elif len(square_coordinates) < NUM_SQUARES:
        # Fill remaining squares with simple grid placement
        container_size = optimizer.calculate_container_size_needed(square_coordinates)
        if container_size == 0:
            container_size = optimizer.min_container_size * 1.3
        
        # Add remaining squares in a simple grid
        remaining = NUM_SQUARES - len(square_coordinates)
        grid_size = int(math.ceil(math.sqrt(remaining)))
        spacing = container_size / (grid_size + 2)  # Leave some margin
        
        for i in range(remaining):
            row = i // grid_size
            col = i % grid_size
            x = (col + 1) * spacing
            y = (row + 1) * spacing
            
            # Check if position is valid and doesn't overlap
            valid = True
            for px, py, prot in square_coordinates:
                if optimizer.squares_overlap(x, y, 0, px, py, prot):
                    valid = False
                    break
            
            if valid and optimizer.is_within_container(x, y, 0, container_size * 1.1):
                square_coordinates.append((x, y, 0.0))
            else:
                # Fallback: place at edge with small offset
                square_coordinates.append((container_size + i * 0.1, 0.5, 0.0))
    
    return square_coordinates

# Optional: Visualization function
def visualize_packing(positions: List[Tuple[float, float, float]], 
                     container_size: float = None):
    """
    Visualize the square packing solution.
    """
    if not positions:
        print("No positions to visualize")
        return
    
    optimizer = SquarePackingOptimizer(len(positions))
    
    if container_size is None:
        container_size = optimizer.calculate_container_size_needed(positions)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Draw container boundary
    container_rect = Rectangle((0, 0), container_size, container_size, 
                             linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(container_rect)
    
    # Draw squares
    colors = plt.cm.Set3(np.linspace(0, 1, len(positions)))
    
    for i, (x, y, rotation) in enumerate(positions):
        corners = optimizer.rotate_square_corners(x, y, rotation)
        # Close the polygon
        corners.append(corners[0])
        xs, ys = zip(*corners)
        
        polygon = patches.Polygon(corners[:-1], closed=True, 
                                facecolor=colors[i], edgecolor='black', 
                                alpha=0.7, linewidth=0.5)
        ax.add_patch(polygon)
    
    ax.set_xlim(-0.5, container_size + 0.5)
    ax.set_ylim(-0.5, container_size + 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Square Packing: {len(positions)} squares in {container_size:.2f}×{container_size:.2f} container')
    
    plt.tight_layout()
    plt.show()

# Test the solver
if __name__ == "__main__":
    print(f"Solving square packing problem for {NUM_SQUARES} unit squares...")
    
    # Run the solver
    result = square_301_solver()
    
    print(f"Solution found with {len(result)} squares placed")
    
    if result:
        optimizer = SquarePackingOptimizer(NUM_SQUARES)
        container_size = optimizer.calculate_container_size_needed(result)
        print(f"Container size needed: {container_size:.3f} × {container_size:.3f}")
        print(f"Container area: {container_size**2:.3f}")
        print(f"Theoretical minimum area: {NUM_SQUARES:.3f}")
        print(f"Efficiency: {NUM_SQUARES/container_size**2*100:.1f}%")
        
        # Show first few coordinates as example
        print("\nFirst 5 square coordinates (x, y, rotation):")
        for i, coord in enumerate(result[:5]):
            print(f"  Square {i+1}: ({coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.1f}°)")
        
        # Uncomment to visualize (requires matplotlib)
        # visualize_packing(result, container_size)
    else:
        print("No solution found!")