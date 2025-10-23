from util import load_prompt_template
from jinja2 import Template

# LLM prompt rendering configuration
# --------------------------------------------------------------------------------------------------
SAMPLER_PROMPT = Template(load_prompt_template('agent_prompts/prompt_sampler/prompt_sampler.j2'))
SAMPLER_PROMPT_SYSTEM = load_prompt_template('agent_prompts/prompt_sampler/system_prompt.j2')

PROMPT_IMPROVER_PROMPT = Template(load_prompt_template('agent_prompts/prompt_improver/prompt_improver.j2'))
PROMPT_IMPROVER_SYSTEM = load_prompt_template('agent_prompts/prompt_improver/system_prompt.j2')