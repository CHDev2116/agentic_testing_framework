import time
import functools
import tracemalloc # Python built-in memory tracing tool.

def monitor_performance(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 1. Start memory and execution-time tracing.
        tracemalloc.start()
        start_time = time.perf_counter()
        
        result = func(*args, **kwargs)
        
        # 2. Stop tracing.
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        duration = end_time - start_time
        
        print(f"\n--- 🚀 Performance Report: [{func.__name__}] ---")
        print(f"⏱️  Execution time: {duration:.4f} sec")
        print(f"🧠  Peak memory: {peak / 10**6:.2f} MB") # Converted to MB.
        print(f"----------------------------------------------")
        return result
    return wrapper