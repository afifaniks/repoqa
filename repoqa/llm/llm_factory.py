from typing import Any, Dict, Optional

import requests
from langchain_ollama import OllamaLLM
from loguru import logger


def get_llm(
    model_name: str,
    backend: str,
    kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Factory function to get LLM model instance based on backend."""
    if backend != "ollama":
        raise ValueError(f"Unsupported backend: {backend}")

    if kwargs is None:
        kwargs = {}

    base_url = kwargs.get("base_url", "http://localhost:11434")
    temperature = kwargs.get("temperature", 0.5)

    # --- Check existing models ---
    logger.info(f"Checking if model '{model_name}' exists on Ollama...")
    try:
        response = requests.get(f"{base_url}/api/tags")
        response.raise_for_status()
        models = [m["name"] for m in response.json().get("models", [])]
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Ollama server: {e}")

    # --- Pull model if missing ---
    if model_name not in models:
        logger.info(f"Model '{model_name}' not found. Pulling from Ollama Hub...")
        pull_response = requests.post(
            f"{base_url}/api/pull",
            json={"name": model_name},
            stream=True,
        )
        if pull_response.status_code != 200:
            raise ValueError(
                f"Failed to pull model '{model_name}'. Response: {pull_response.text}"
            )

        for line in pull_response.iter_lines(decode_unicode=True):
            if line:
                logger.debug(line)

        logger.info(f"Model '{model_name}' installed successfully.")
    else:
        logger.info(f"Model '{model_name}' already available locally.")

    # --- Return LLM instance ---
    return OllamaLLM(
        model=model_name, base_url=base_url, temperature=temperature, reasoning=False
    )
