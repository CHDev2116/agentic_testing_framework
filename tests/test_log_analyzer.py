"""Sliding-window stability with bounded error tolerance (golden strings)."""

from eval.log_analyzer import LogAnalyzer


def test_find_max_stable_sequence_k0_all_success():
    assert LogAnalyzer(0).find_max_stable_sequence("SSSS") == 4


def test_find_max_stable_sequence_k1_full_string_with_one_error():
    # "SSSESSS" — one E inside; k=1 allows entire window.
    assert LogAnalyzer(1).find_max_stable_sequence("SSSESSS") == 7


def test_find_max_stable_sequence_k0_breaks_at_each_error():
    assert LogAnalyzer(0).find_max_stable_sequence("SSEESS") == 2


def test_find_max_stable_sequence_empty():
    assert LogAnalyzer(1).find_max_stable_sequence("") == 0
