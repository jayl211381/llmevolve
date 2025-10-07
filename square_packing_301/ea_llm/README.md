Evolutionary method for discovering new programs

Alpha evolve core loop extracted from 
https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf

The user provides an initial program (with components to evolve marked), evaluation code, and optional configurations
AlphaEvolve then initiates an evolutionary loop. 
The Prompt sampler uses programs from the Program database to construct rich prompts
Given these prompts, the LLMs generate code modifications (diffs), which are applied to create new programs. 
These are then scored by Evaluators and promising solutions are registered back into the Program database.

User configurations:
Prompt sampler - Prompt template and configuration (config.py)
LLMs ensemble - Choice of existing LLMs
Evaluators pool - Evaluation code
Program database - Initial program to evolve

pseudo code:
parent_program, inspirations = database.sample()
prompt = prompt_sampler.build(parent_program, inspirations)
diff = llm.generate(prompt)
child_program = apply_diff(parent_program, diff)
results = evaluator.execute(child_program)
database.add(child_program, results)

Super optimization agent
- Program search, discover algorithms to describe the solution

Evaluation function h
- takes a function and returns a scalar 

Full prompt for each iteration 
- Multiple previous solutions + exec outputs + evalution results + idea (Resurfaced with GA)
  - Evalution results = eval function h + llm feedback (total rotations, code simplicity)
- system prompt on how to propose new changes
- explicit context on the problem
- prompt evolution (evolving the system prompt on how to propose better changes)

Prompt DB (Genetic Algorithms search)
- Balancing exploration and exploitation with a cominbination of island based and map elite algorithms.
- surface new batch of solutions to maximize new and helpful idea generation
- Ideas from different islands cross over, mutate (prompt/idea change with llm)

Heuristic search algorithms
- Take each result and iteratively revise how the llm is going to generate new ideas
- early heruistics allow for large steps and big gains
- late heuristics allow for smaller optimizations and more exploration.
