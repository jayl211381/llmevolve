"""
Configuration file for API keys and settings.
"""

import os
from pathlib import Path

# Get the API key from environment variable or config file
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')

# Alternative: Load from a local config file (not recommended for production)
CONFIG_DIR = Path(__file__).parent
API_KEY_FILE = CONFIG_DIR / 'api_keys.txt'

def get_claude_api_key():
    """
    Get Claude API key from environment variable or config file.
    
    Returns:
        str: The Claude API key
        
    Raises:
        ValueError: If no API key is found
    """
    # First try environment variable
    if CLAUDE_API_KEY:
        return CLAUDE_API_KEY
    
    # Then try config file
    if API_KEY_FILE.exists():
        with open(API_KEY_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('CLAUDE_API_KEY='):
                    return line.split('=', 1)[1]
    
    raise ValueError(
        "Claude API key not found. Please set CLAUDE_API_KEY environment variable "
        f"or add 'CLAUDE_API_KEY=your_key' to {API_KEY_FILE}"
    )

# Claude API configuration
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"  # Default model
MAX_TOKENS = 16000

# Available Claude models
CLAUDE_MODELS = {
    "sonnet-4": "claude-sonnet-4-20250514",
    "sonnet-37": "claude-3-7-sonnet-latest",
    "opus-4": "claude-opus-4-0",
    "haiku-35": "claude-3-5-haiku-latest"
}

def get_gemini_api_key():
    """
    Get Gemini API key from environment variable or config file.
    
    Returns:
        str: The Gemini API key
        
    Raises:
        ValueError: If no API key is found
    """
    # First try environment variable
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key:
        return gemini_key
    
    # Then try config file
    if API_KEY_FILE.exists():
        with open(API_KEY_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('GEMINI_API_KEY='):
                    return line.split('=', 1)[1]
    
    raise ValueError(
        "Gemini API key not found. Please set GEMINI_API_KEY environment variable "
        f"or add 'GEMINI_API_KEY=your_key' to {API_KEY_FILE}"
    )

# Gemini API configuration
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODEL = "gemini-1.5-flash"  # Default model
GEMINI_MAX_TOKENS = 16000

# Available Gemini models
GEMINI_MODELS = {
    "2.5-pro": "gemini-2.5-pro",
    "2.5-flash": "gemini-2.5-flash",
    "2.5-flash-lite": "gemini-2.5-flash-lite",
    "2.0-flash": "gemini-2.0-flash",
    "2.0-flash-lite": "gemini-2.0-flash-lite",
}

def get_openai_api_key():
    """
    Get OpenAI API key from environment variable or config file.
    
    Returns:
        str: The OpenAI API key
        
    Raises:
        ValueError: If no API key is found
    """
    # First try environment variable
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        return openai_key
    
    # Then try config file
    if API_KEY_FILE.exists():
        with open(API_KEY_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('OPENAI_API_KEY='):
                    return line.split('=', 1)[1]
    
    raise ValueError(
        "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
        f"or add 'OPENAI_API_KEY=your_key' to {API_KEY_FILE}"
    )

# OpenAI API configuration
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"  # Default model
OPENAI_MAX_TOKENS = 16000

# Available OpenAI models
OPENAI_MODELS = {
    "gpt-4o-latest": "chatgpt-4o-latest",
    "gpt-35-turbo": "gpt-3.5-turbo",
    "gpt-4": "gpt-4",
    "gpt-4-turbo": "gpt-4-turbo",
    "gpt-41": "gpt-4.1",
    "gpt-41-mini": "gpt-4.1-mini",
    "gpt-41-nano": "gpt-4.1-nano",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
}
