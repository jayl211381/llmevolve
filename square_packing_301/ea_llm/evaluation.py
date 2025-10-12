import os
import sys
# Add parent directory to path to access solution_evaluator
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from solution_evaluator.evaluator import SquarePackingEvaluator

def main():
    """Main evaluation function."""
    evaluator = SquarePackingEvaluator()
    
    # Base path for llm_gen directory
    llm_gen_dir = os.path.join(os.path.dirname(__file__), 'llm_gen')
    
    # runs the extracted code to generate solution
    code_file_path = os.path.join(llm_gen_dir, 'extracted_code_block_1.py')
    
    solution_file = os.path.join(llm_gen_dir, 'llm_generated_solution.json')
    results_file = os.path.join(llm_gen_dir, 'eval_results.json')
    # Evalutes the llm generated solution
    
    # Saves the code generated solution and evaluation results as json file
    results = evaluator.evaluate_from_code(code_file_path, solution_file, 
                                           results_file=results_file)

    print(f"Number of squares: {results['num_squares']}")
    print(f"Overlapping pairs: {results['overlaps']}")
    print("✓ VALID" if results['is_valid'] else "✗ INVALID")
    print(f"Square side length: {results['side_length']}")
    print(f"Packing efficiency: {results['efficiency']}%")

if __name__ == "__main__":
    main()