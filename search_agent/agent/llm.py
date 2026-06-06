import requests


class OllamaModel:
    def __init__(
        self,
        model_name: str = "qwen3.5:4b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": 192,
                    "repeat_penalty": 1.15,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
