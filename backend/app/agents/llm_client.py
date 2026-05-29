import urllib.request
import json

class OllamaLLMClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url

    def query_llama(self, prompt: str, system_prompt: str = ""):
        # Calls Ollama REST endpoint for Llama 3.2 3B model
        payload = {
            "model": "llama3.2:3b",
            "prompt": prompt,
            "system": system_prompt,
            "stream": False
        }
        
        # Simple client request placeholder
        return "LLM Compliance Inference Output"
