import requests
import time

def test_llama_connection():
    url = "http://localhost:8080/v1/chat/completions"
    payload = {
        "model": "llama-3.1-8b",
        "messages": [
            {"role": "user", "content": "Hi"}
        ],
        "max_tokens": 5,        # Minimal output to save time
        "temperature": 0.0
    }
    
    print("⏳ Sending a tiny request to wake up the M4 GPU...")
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        print(f"✅ Status Code: {response.status_code}")
        print(f"⏱️ Response Time: {time.time() - start_time:.2f}s")
        print(f"🤖 AI Response: {response.json()['choices'][0]['message']['content']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_llama_connection()