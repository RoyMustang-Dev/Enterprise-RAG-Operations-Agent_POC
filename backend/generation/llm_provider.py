import requests
import json
import sys
import os
from typing import Dict, Any, Generator

# Add project root to path if running directly
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class OllamaClient:
    """
    Client for interacting with a local Ollama instance.
    Defaults to http://localhost:11434.
    """
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:1b"):
        self.base_url = base_url
        self.model = model
        self.api_generate = f"{base_url}/api/generate"

    def check_connection(self) -> bool:
        """Checks if the Ollama server is reachable."""
        try:
            response = requests.get(self.base_url)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def generate(self, prompt: str, system_prompt: str = None, stream: bool = False) -> str:
        """
        Generates a completion for the given prompt.
        
        Args:
            prompt (str): The user prompt.
            system_prompt (str, optional): System instructions.
            stream (bool): Whether to stream the response (not implemented in v1).
            
        Returns:
            str: The generated text.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False # Enforce no streaming for simplicity in v1
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        try:
            response = requests.post(self.api_generate, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama: {e}")
            return f"Error: Failed to generate response from {self.model}."

if __name__ == "__main__":
    # Quick test
    client = OllamaClient()
    if client.check_connection():
        print(f"Connected to Ollama. Model: {client.model}")
        print("Response:", client.generate("Why is the sky blue?", system_prompt="Answer briefly."))
    else:
        print("Could not connect to Ollama. Is it running?")
