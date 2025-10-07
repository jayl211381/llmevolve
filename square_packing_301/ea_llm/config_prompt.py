from util import load_prompt_template
from config import NUM_SQUARES

# LLM prompt rendering configuration
# --------------------------------------------------------------------------------------------------
IMPROVE_PROMPT = load_prompt_template('agent_prompts/codebase_improver/codebase_improver.j2')
IMPROVE_SYSTEM_PROMPT = load_prompt_template('agent_prompts/codebase_improver/system_prompt.j2')

EVALUATOR_PROMPT = load_prompt_template('agent_prompts/evaluator/evaluator.j2')
EVALUATOR_PROMPT_SYSTEM = load_prompt_template('agent_prompts/evaluator/system_prompt.j2')

SAMPLER_PROMPT = load_prompt_template('agent_prompts/solution_sampler/solution_sampler.j2')
SAMPLER_PROMPT_SYSTEM = load_prompt_template('agent_prompts/solution_sampler/system_prompt.j2')

PROMPT_IMPROVER_PROMPT = load_prompt_template('agent_prompts/prompt_improver/prompt_improver.j2')
PROMPT_IMPROVER_SYSTEM = load_prompt_template('agent_prompts/prompt_improver/system_prompt.j2')

PROMPT_SOLUTION_GENERATOR = load_prompt_template('solution_prompts/current_prompt.j2',
                                                 context={'NUMBER_OF_SQUARES': NUM_SQUARES})
PROMPT_SOLUTION_GENERATOR_SYSTEM = load_prompt_template('solution_prompts/system_prompt.j2')
