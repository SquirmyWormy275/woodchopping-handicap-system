"""
LLM Integration for Woodchopping Handicap Predictions

This module handles communication with the Ollama API for AI-enhanced
time predictions.

Functions:
    call_ollama() - Send prompts to local Ollama instance and receive responses
"""

from typing import Optional
import requests

from config import llm_config


def call_ollama(prompt: str, model: str = None) -> Optional[str]:
    """
    Send prompt to local Ollama instance and return response.

    Args:
        prompt: Text prompt to send to the model
        model: Ollama model name (defaults to config value)

    Returns:
        Model response text, or None if error occurs

    Example:
        >>> response = call_ollama("Predict cutting time for...")
        >>> if response:
        ...     print(response)
    """
    if model is None:
        model = llm_config.DEFAULT_MODEL

    try:
        response = requests.post(
            llm_config.OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Low creativity for consistent predictions
                    "num_predict": 50    # Limit response length for speed
                }
            },
            timeout=llm_config.TIMEOUT_SECONDS
        )

        if response.status_code == 200:
            return response.json()['response'].strip()
        else:
            print(f"Ollama error: {response.status_code}")
            return None

    except requests.exceptions.ConnectionError:
        print("\nError: Cannot connect to Ollama. Make sure it's running:")
        print("  Run 'ollama serve' in a terminal")
        return None
    except Exception as e:
        print(f"\nError calling Ollama: {e}")
        return None
