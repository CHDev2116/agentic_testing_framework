# src/models/gemma_filter.py
from importlib import import_module

class GemmaFilter:
    def __init__(self):
        self.model_name = "gemma2:2b"
        self.ollama = None
        try:
            self.ollama = import_module("ollama")
        except ModuleNotFoundError:
            self.ollama = None

    def check_basic_quality(self, metrics):
        # 這裡必須縮進（4個空格）
        b = metrics.get('brightness', 0)
        
        prompt = f"""
        IMAGE METRIC: brightness = {b}
        
        TASK: Is brightness between 20 and 240?
        
        DECISION RULES:
        - If {b} < 20: return pass=false
        - If {b} > 240: return pass=false
        - Otherwise: return pass=true
        
        JSON OUTPUT ONLY:
        {{"pass": true, "reason": "..."}}
        """
        
        if self.ollama is None:
            raise RuntimeError("ollama package is not installed. Run: pip install ollama")

        response = self.ollama.chat(
            model=self.model_name, 
            messages=[{'role': 'user', 'content': prompt}],
            format='json'
        )
        return response['message']['content']