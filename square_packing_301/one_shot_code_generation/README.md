# One-Shot Code Generation for Square Packing

Uses LLMs to generate complete algorithmic solutions for packing 301 unit squares into the smallest possible square container.

## Components

- **`generate_solution.py`** - Main script
- **`util.py`** - Prompt generation and response parsing  
- **`run_extracted_code.py`** - Execute generated code
- **`prompts/`** - LLM prompt templates
- **`llm_gen/`** - Generated solutions

## Usage

```bash
# Generate solution
python generate_solution.py

# Run generated code
python run_extracted_code.py llm_gen/extracted_code_block_1.py
```

## Output Format

Generated function returns coordinates as `(x, y, rotation)` tuples:

```python
def square_301_solver():
    return [(1.0, 1.0, 0.0), (2.0, 1.0, 45.0), ...]  # x, y, rotation_degrees
```

## Current Performance

- 301 squares packed
- Container: 19.843 × 19.843 units  
- Efficiency: 76.4%