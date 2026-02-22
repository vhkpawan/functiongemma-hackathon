#!/usr/bin/env python3
"""
Temporary script to find best THRESHOLD_EASY and THRESHOLD_MEDIUM for generate_hybrid.
Set THRESHOLD_EASY and THRESHOLD_MEDIUM env vars; main.py reads them in _get_dynamic_threshold.
Do not submit this file.
"""
import os
import sys

# Ensure we can import benchmark (and thus main)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_quiet_benchmark():
    """Run benchmark and return (results, score), suppressing stdout."""
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        from benchmark import run_benchmark, compute_total_score
        results = run_benchmark()
        score = compute_total_score(results)
        return results, score
    finally:
        sys.stdout = old_stdout


def main():
    # Grid: step 0.03 from 0.72 to 0.96 for both easy and medium
    values = [round(0.72 + i * 0.03, 2) for i in range(9)]  # 0.72 .. 0.96
    best_score = -1.0
    best_easy = best_medium = None
    total = len(values) * len(values)
    n = 0
    for easy in values:
        for medium in values:
            n += 1
            os.environ["THRESHOLD_EASY"] = str(easy)
            os.environ["THRESHOLD_MEDIUM"] = str(medium)
            try:
                _, score = run_quiet_benchmark()
            except Exception as e:
                print(f"[{n}/{total}] easy={easy} medium={medium} ERROR: {e}", flush=True)
                continue
            if score > best_score:
                best_score = score
                best_easy = easy
                best_medium = medium
            print(f"[{n}/{total}] easy={easy:.2f} medium={medium:.2f} -> score={score:.1f}% (best={best_score:.1f}%)", flush=True)
    print()
    print("Best thresholds (set in main.py or export for runs):")
    print(f"  THRESHOLD_EASY={best_easy}   THRESHOLD_MEDIUM={best_medium}")
    print(f"  Best score: {best_score:.1f}%")
    return best_easy, best_medium, best_score


if __name__ == "__main__":
    main()
