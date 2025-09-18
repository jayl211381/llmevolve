#!/usr/bin/env python3
"""
Simple evaluator for LLM-generated square packing solutions.
"""

import json
import math
import os

def load_solution(file_path: str):
    """Load solution from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data if isinstance(data, list) else data['coordinates']

def get_square_corners(cx: float, cy: float, rotation: float = 0.0):
    """Calculate corners of a unit square."""
    theta = math.radians(rotation)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    
    corners = []
    for dx, dy in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]:
        rotated_x = dx * cos_theta - dy * sin_theta
        rotated_y = dx * sin_theta + dy * cos_theta
        corners.append((cx + rotated_x, cy + rotated_y))
    
    return corners

def squares_overlap(square1, square2):
    """Check if two squares overlap."""
    cx1, cy1, rot1 = square1
    cx2, cy2, rot2 = square2
    
    corners1 = get_square_corners(cx1, cy1, rot1)
    corners2 = get_square_corners(cx2, cy2, rot2)
    
    def point_in_square(px, py, corners):
        for i in range(4):
            x1, y1 = corners[i]
            x2, y2 = corners[(i + 1) % 4]
            cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
            if cross < -1e-6:
                return False
        return True
    
    # Check if any corner of one square is in the other
    for corner in corners1:
        if point_in_square(corner[0], corner[1], corners2):
            return True
    for corner in corners2:
        if point_in_square(corner[0], corner[1], corners1):
            return True
    
    return False

def evaluate_solution(coordinates):
    """Evaluate the solution."""
    print(f"Number of squares: {len(coordinates)}")
    
    # Check overlaps
    overlaps = 0
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            if squares_overlap(coordinates[i], coordinates[j]):
                overlaps += 1
    
    print(f"Overlapping pairs: {overlaps}")
    print("✓ VALID" if overlaps == 0 else "✗ INVALID")
    
    # Calculate container size
    all_corners = []
    for cx, cy, rotation in coordinates:
        all_corners.extend(get_square_corners(cx, cy, rotation))
    
    min_x = min(corner[0] for corner in all_corners)
    max_x = max(corner[0] for corner in all_corners)
    min_y = min(corner[1] for corner in all_corners)
    max_y = max(corner[1] for corner in all_corners)
    
    side_length = max(max_x - min_x, max_y - min_y)
    area = side_length ** 2
    efficiency = len(coordinates) / area * 100
    
    print(f"Square side length: {side_length:.3f}")
    print(f"Packing efficiency: {efficiency:.1f}%")

def main():
    """Main evaluation function."""
    solution_file = os.path.join(os.path.dirname(__file__), '..', 'one_shot_code_generation', 'llm_gen', 'llm_generated_solution.json')
    
    if not os.path.exists(solution_file):
        solution_file = os.path.join(os.path.dirname(__file__), '..', 'one_shot_code_generation', 'llm_generated_solution.json')
    
    if not os.path.exists(solution_file):
        print("Solution file not found!")
        return
    
    coordinates = load_solution(solution_file)
    evaluate_solution(coordinates)

if __name__ == "__main__":
    main()