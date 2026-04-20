import time
import functools
import tracemalloc # Python 內建的記憶體追蹤工具

def monitor_performance(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 1. 開始追蹤記憶體與時間
        tracemalloc.start()
        start_time = time.perf_counter()
        
        result = func(*args, **kwargs)
        
        # 2. 結束追蹤
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        duration = end_time - start_time
        
        print(f"\n--- 🚀 Performance Report: [{func.__name__}] ---")
        print(f"⏱️  執行耗時: {duration:.4f} 秒")
        print(f"🧠  記憶體高峰: {peak / 10**6:.2f} MB") # 換算成 MB
        print(f"----------------------------------------------")
        return result
    return wrapper