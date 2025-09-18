import json
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def get_square_corners(cx, cy, rotation=0):
    """Get corners of unit square centered at (cx, cy) with rotation."""
    theta = math.radians(rotation)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    
    corners = []
    for dx, dy in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]:
        x = cx + dx * cos_t - dy * sin_t
        y = cy + dx * sin_t + dy * cos_t
        corners.append((x, y))
    return corners

def plot_solution(coordinates):
    """Plot squares and show dimensions."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Get all corners to find bounds
    all_corners = []
    for cx, cy, rotation in coordinates:
        all_corners.extend(get_square_corners(cx, cy, rotation))
    
    min_x = min(x for x, y in all_corners)
    max_x = max(x for x, y in all_corners)
    min_y = min(y for x, y in all_corners)
    max_y = max(y for x, y in all_corners)
    
    width = max_x - min_x
    height = max_y - min_y
    side = max(width, height)
    
    # Draw squares
    for cx, cy, rotation in coordinates:
        corners = get_square_corners(cx, cy, rotation)
        square = patches.Polygon(corners, facecolor='lightblue', edgecolor='black', alpha=0.7)
        ax.add_patch(square)
    
    # Draw container
    container = patches.Rectangle((min_x, min_y), side, side, 
                                fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(container)
    
    ax.set_aspect('equal')
    ax.set_xlim(min_x - 1, min_x + side + 1)
    ax.set_ylim(min_y - 1, min_y + side + 1)
    ax.grid(alpha=0.3)
    
    efficiency = len(coordinates) / (side ** 2) * 100
    ax.set_title(f"{len(coordinates)} squares, container: {side:.2f}, efficiency: {efficiency:.1f}%")
    
    plt.show()
    print(f"Squares: {len(coordinates)}, Container: {side:.3f}, Efficiency: {efficiency:.1f}%")

if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), '..', 'one_shot_code_generation', 'llm_gen', 'llm_generated_solution.json')
    
    # Load and plot
    with open(file_path) as f:
        coordinates = json.load(f)

    plot_solution(coordinates)