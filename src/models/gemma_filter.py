# src/models/gemma_filter.py
import ollama

class GemmaFilter:
    def __init__(self):
        self.model_name = "gemma2:2b"

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
        
        response = ollama.chat(
            model=self.model_name, 
            messages=[{'role': 'user', 'content': prompt}],
            format='json'
        )
        return response['message']['content']