import logging

logger = logging.getLogger(__name__)


class LogAnalyzer:
    def __init__(self, error_tolerance: int):
        """
        error_tolerance (k): Maximum number of errors (E) allowed in the window.
        """
        self.k = error_tolerance

    def find_max_stable_sequence(self, logs: str) -> int:
        """
        Input: logs = "SSSESSS" (S=Success, E=Error)
        Output: The longest consecutive test sequence length while tolerating k errors.
        """
        left = 0
        max_length = 0
        error_count = 0  # Track error count directly for better efficiency than a map.

        for right in range(len(logs)):
            # If the current status is Error, increase the counter.
            if logs[right] == "E":
                error_count += 1

            # Shrink the left boundary when error count exceeds tolerance.
            while error_count > self.k:
                if logs[left] == "E":
                    error_count -= 1
                left += 1

            # Update the maximum valid window size.
            max_length = max(max_length, right - left + 1)

        return max_length


if __name__ == "__main__":
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_logs = "SSSESSS"

    analyzer = LogAnalyzer(error_tolerance=1)
    stable_length = analyzer.find_max_stable_sequence(test_logs)

    logger.info("Test log: %s", test_logs)
    logger.info("Longest stable segment with tolerance %s: %s", analyzer.k, stable_length)
