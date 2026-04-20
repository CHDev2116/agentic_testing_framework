class LogAnalyzer:
    def __init__(self, error_tolerance: int):
        """
        error_tolerance (k): 窗口內允許出現的最大錯誤(E)次數
        """
        self.k = error_tolerance

    def find_max_stable_sequence(self, logs: str) -> int:
        """
        輸入：logs = "SSSESSS" (S=Success, E=Error)
        輸出：在容忍 k 個錯誤下，最長的連續測試序列長度
        """
        left = 0
        max_length = 0
        error_count = 0  # 直接追蹤錯誤次數，比維護字典更高效

        for right in range(len(logs)):
            # 如果遇到 Error，增加計數
            if logs[right] == 'E':
                error_count += 1
            
            # 當錯誤超過容忍度，收縮左側窗口
            while error_count > self.k:
                if logs[left] == 'E':
                    error_count -= 1
                left += 1
            
            # 更新最大長度
            max_length = max(max_length, right - left + 1)
            
        return max_length

if __name__ == "__main__":
    # 模擬 ai_quality_agent 跑完後的結果串
    # S = Success, E = Error
    test_logs = "SSSESSS" 
    
    # 容忍 1 個錯誤
    analyzer = LogAnalyzer(error_tolerance=1)
    stable_length = analyzer.find_max_stable_sequence(test_logs)
    
    print(f"測試 Log: {test_logs}")
    print(f"在容忍 {analyzer.k} 個錯誤下，最長穩定區間為: {stable_length}")
    
    # 預期結果：SSSESSS 裡面，包含一個 E 的最長區間就是整段，長度應為 7