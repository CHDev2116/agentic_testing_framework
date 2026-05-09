import requests
import time

def test_llama_health_check(url="http://localhost:8080/v1", model="llama-3.1-8b"):
    endpoint = f"{url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Ping"}],
        "max_tokens": 1, 
        "temperature": 0.0
    }
    
    try:
        start = time.perf_counter() # 使用更精確的計時器
        res = requests.post(endpoint, json=payload, timeout=30)
        res.raise_for_status() # 直接攔截 4xx/5xx 錯誤
        
        latency = time.perf_counter() - start
        data = res.json()
        
        print(f"✅ [{model}] Connected.")
        print(f"⏱️ TTFT (Approx): {latency:.4f}s")
        # 這裡可以整合進 Agentic Testing Framework 的效能報告中
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Failed: {e}")
    except KeyError:
        print(f"❌ Malformed Response: {res.text}")

if __name__ == "__main__":
    test_llama_health_check()