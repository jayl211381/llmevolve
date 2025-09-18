"""
LLM API calling functions for Claude, Gemini, and OpenAI.
"""

import requests
import sys
from pathlib import Path

# Add the parent directory to the path to import configs
sys.path.append(str(Path(__file__).parent.parent))

from configs.config import (
    get_claude_api_key, CLAUDE_API_URL, CLAUDE_MODEL, MAX_TOKENS, CLAUDE_MODELS,
    get_gemini_api_key, GEMINI_API_URL, GEMINI_MODEL, GEMINI_MAX_TOKENS, GEMINI_MODELS,
    get_openai_api_key, OPENAI_API_URL, OPENAI_MODEL, OPENAI_MAX_TOKENS, OPENAI_MODELS
)


class LLMError(Exception):
    """Base exception for LLM API errors."""
    pass


def resolve_model(model_name: str, provider: str) -> str:
    """Resolve model name to full identifier."""
    model_dict = {
        "claude": CLAUDE_MODELS,
        "gemini": GEMINI_MODELS, 
        "openai": OPENAI_MODELS
    }[provider]
    
    if model_name in model_dict:
        return model_dict[model_name]
    elif model_name in model_dict.values():
        return model_name
    else:
        available = ", ".join(model_dict.keys())
        raise ValueError(f"Unknown {provider} model: {model_name}. Available: {available}")


def call_claude_api(
    message: str,
    model: str = CLAUDE_MODEL,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.7,
    system_prompt: str = None
) -> str:
    """Call Claude API with simplified error handling."""
    
    # Resolve model and get API key
    resolved_model = resolve_model(model, "claude")
    api_key = get_claude_api_key()
    
    # Prepare request
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    
    payload = {
        "model": resolved_model,
        "messages": [{"role": "user", "content": message}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    if system_prompt:
        payload["system"] = system_prompt
    
    # Make API call
    response = requests.post(CLAUDE_API_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise LLMError(f"Claude API error: {response.status_code} - {response.text}")
    
    return response.json()["content"][0]["text"]


def call_gemini_api(
    message: str,
    model: str = GEMINI_MODEL,
    max_tokens: int = GEMINI_MAX_TOKENS,
    temperature: float = 0.7,
    system_prompt: str = None
) -> str:
    """Call Gemini API with simplified error handling."""
    
    # Resolve model and get API key
    resolved_model = resolve_model(model, "gemini")
    api_key = get_gemini_api_key()
    
    # Prepare request
    url = f"{GEMINI_API_URL}/{resolved_model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    # Build content parts
    parts = []
    if system_prompt:
        parts.append({"text": system_prompt})
    parts.append({"text": message})
    
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature
        }
    }
    
    # Make API call
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise LLMError(f"Gemini API error: {response.status_code} - {response.text}")
    
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]


def call_openai_api(
    message: str,
    model: str = OPENAI_MODEL,
    max_tokens: int = OPENAI_MAX_TOKENS,
    temperature: float = 0.7,
    system_prompt: str = None
) -> str:
    """Call OpenAI API with simplified error handling."""
    
    # Resolve model and get API key
    resolved_model = resolve_model(model, "openai")
    api_key = get_openai_api_key()
    
    # Prepare request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": message})
    
    payload = {
        "model": resolved_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    # Make API call
    response = requests.post(OPENAI_API_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise LLMError(f"OpenAI API error: {response.status_code} - {response.text}")
    
    return response.json()["choices"][0]["message"]["content"]


def call_llm(
    message: str,
    provider: str = "claude",
    model: str = None,
    max_tokens: int = None,
    temperature: float = 0.7,
    system_prompt: str = None
) -> str:
    """Universal function to call any LLM provider."""
    
    provider = provider.lower()
    
    if provider == "claude":
        return call_claude_api(
            message=message,
            model=model or CLAUDE_MODEL,
            max_tokens=max_tokens if max_tokens is not None else MAX_TOKENS,
            temperature=temperature,
            system_prompt=system_prompt
        )
    
    elif provider == "gemini":
        return call_gemini_api(
            message=message,
            model=model or GEMINI_MODEL,
            max_tokens=max_tokens if max_tokens is not None else GEMINI_MAX_TOKENS,
            temperature=temperature,
            system_prompt=system_prompt
        )
    
    elif provider == "openai":
        return call_openai_api(
            message=message,
            model=model or OPENAI_MODEL,
            max_tokens=max_tokens if max_tokens is not None else OPENAI_MAX_TOKENS,
            temperature=temperature,
            system_prompt=system_prompt
        )
    
    else:
        raise ValueError(f"Unsupported provider: {provider}. Available: claude, gemini, openai")


def list_available_models():
    """List all available models for each provider."""
    return {
        "claude": list(CLAUDE_MODELS.keys()),
        "gemini": list(GEMINI_MODELS.keys()),
        "openai": list(OPENAI_MODELS.keys())
    }


# Example usage
if __name__ == "__main__":
    print("Testing simplified LLM calls...")
    
    try:
        # Test Claude
        response = call_llm("What is 2+2?", provider="claude", model="haiku-35")
        print(f"Claude: {response[:100]}...")
        
    except Exception as e:
        print(f"Error: {e}")
