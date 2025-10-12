#!/usr/bin/env python3
"""
Simple evaluator for LLM-generated square packing solutions.
"""

import json
import math
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from llm.call_llm import call_llm
from jinja2 import Template

class SquarePackingEvaluator:
    
    def __init__(self):
        self.results = {}
    
    def load_solution(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data if isinstance(data, list) else data['coordinates']
    
    def run_extracted_code(self, code_file_path, solution_file, function_name="square_301_solver"):
        if not os.path.exists(code_file_path):
            return None
        
        namespace = {}
        with open(code_file_path, 'r', encoding='utf-8') as f:
            exec(f.read(), namespace)
        
        if function_name in namespace and callable(namespace[function_name]):
            result = namespace[function_name]()
            with open(solution_file, 'w') as f:
                json.dump(result, f, indent=2)
            return result
        return None
    
    def get_square_corners(self, cx, cy, rotation=0.0):
        theta = math.radians(rotation)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        
        corners = []
        for dx, dy in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]:
            rx = dx * cos_t - dy * sin_t
            ry = dx * sin_t + dy * cos_t
            corners.append((cx + rx, cy + ry))
        return corners
    
    def squares_overlap(self, square1, square2):
        corners1 = self.get_square_corners(*square1)
        corners2 = self.get_square_corners(*square2)
        
        def point_in_square(px, py, corners):
            for i in range(4):
                x1, y1 = corners[i]
                x2, y2 = corners[(i + 1) % 4]
                if (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1) < -1e-6:
                    return False
            return True
        
        for corner in corners1:
            if point_in_square(*corner, corners2):
                return True
        for corner in corners2:
            if point_in_square(*corner, corners1):
                return True
        return False
    
    def evaluate_with_llm(self, code_file_path, coordinates, provider="claude", model="sonnet-4"):
        """Call LLM to evaluate solution using prompt template."""
        
        # Load evaluator prompt template
        prompt_path = os.path.join(os.path.dirname(__file__), 'evaluator_prompt', 'evaluator.j2')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template = Template(f.read())
        
        # Load system prompt
        system_prompt_path = os.path.join(os.path.dirname(__file__), 'evaluator_prompt', 'system_prompt.j2')
        with open(system_prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
        
        # Load code content
        with open(code_file_path, 'r', encoding='utf-8') as f:
            code_content = f.read()
        
        # Render prompt with context
        prompt = prompt_template.render(
            NUMBER_OF_SQUARES=len(coordinates),
            CODE_SNIPPET=code_content,
            SIDE_LENGTH=self.results.get('side_length', 'Unknown'),
            EFFICIENCY=self.results.get('efficiency', 'Unknown'),
            OVERLAPS=self.results.get('overlaps', 'Unknown'),
            IS_VALID=self.results.get('is_valid', False)
        )
        
        # call LLM
        llm_response = call_llm(
            message=prompt,
            provider=provider,
            model=model,  
            max_tokens=8000,  # Increased token limit
            temperature=0.3,  
            system_prompt=system_prompt
        )
        
        
        # Store results
        self.results['llm_evaluation'] = {
            'response': llm_response,
            'provider': provider,
            'model': model
        }
        
        return llm_response
    
    def evaluate_solution(self, coordinates, code_file_path, output_path=None):
        num_squares = len(coordinates)
        
        overlaps = sum(1 for i in range(num_squares) 
                      for j in range(i + 1, num_squares)
                      if self.squares_overlap(coordinates[i], coordinates[j]))
        
        all_corners = []
        for coord in coordinates:
            all_corners.extend(self.get_square_corners(*coord))
        
        xs = [corner[0] for corner in all_corners]
        ys = [corner[1] for corner in all_corners]
        side_length = max(max(xs) - min(xs), max(ys) - min(ys))
        
        self.results.update({
            'num_squares': num_squares,
            'overlaps': overlaps,
            'is_valid': overlaps == 0,
            'side_length': round(side_length, 3),
            'efficiency': round(num_squares / (side_length ** 2) * 100, 1)
        })

        self.evaluate_with_llm(code_file_path, coordinates)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2)
        
        return self.results 
    
    def evaluate_from_code(self, code_file_path, solution_file, function_name="square_301_solver", results_file=None):
        coordinates = self.run_extracted_code(code_file_path, solution_file, function_name)
        return self.evaluate_solution(coordinates,code_file_path, results_file) if coordinates else None