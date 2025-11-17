#!/usr/bin/env python3
"""
run_evaluation.py
-----------------

Run the Shouji filter over a synthetic dataset and compute:

  - False accept rate
  - False reject rate
  - Overall accuracy
  - Runtime and throughput

The script expects a TSV file with the columns:
    pattern   reference   true_edits

Usage:
    python scripts/run_evaluation.py \
        --input data/synthetic/small_synth.tsv \
        --out ci_results.json \
        --threshold 10
"""

import argparse
import json
import time

import edlib
from tqdm import tqdm

# Import Shouji implementation
from src.shouji.shouji import shouji_single


def compute_edit_distance(p: str, t: str) -> int:
    """Return the Levenshtein edit distance using edlib."""
    result = edlib.align(p, t, mode="NW")  # Needleman-Wunsch (global)
    return result["editDistance"]


def evaluate_dataset(input_tsv: str, threshold: int) -> dict:
    """
    Evaluate Shouji on a dataset and return metrics dict.
    """
    # Counters
    total = 0
    true_similar = 0           # True edit distance <= threshold
    true_dissimilar = 0        # > threshold
    false_accept = 0
    false_reject = 0
    correct = 0

    start_time = time.time()

    # Count lines first (for progress bar)
    print("Counting dataset size...")
    with open(input_tsv, "r") as f:
        # Skip header
        next(f)
        num_lines = sum(1 for _ in f)

    print(f"Dataset contains {num_lines} pairs.\n")

    with open(input_tsv, "r") as f:
        header = next(f).strip().split("\t")
        if header != ["pattern", "reference", "true_edits"]:
            raise ValueError(
                "Input TSV must have header: pattern\\treference\\ttrue_edits"
            )

        for line in tqdm(f, total=num_lines, desc="Evaluating"):
            pattern, reference, _ = line.strip().split("\t")

            # Apply Shouji filter
            accept, _ = shouji_single(pattern, reference, E=threshold)

            # Compute true edit distance via edlib
            true_dist = compute_edit_distance(pattern, reference)

            # Track tallies
            total += 1

            if true_dist <= threshold:
                true_similar += 1
                if accept:
                    correct += 1
                else:
                    false_reject += 1
            else:
                true_dissimilar += 1
                if accept:
                    false_accept += 1
                else:
                    correct += 1

    end_time = time.time()
    elapsed = end_time - start_time
    throughput = total / elapsed if elapsed > 0 else 0.0

    # Build metrics dictionary
    metrics = {
        "total_pairs": total,
        "threshold": threshold,
        "true_similar": true_similar,
        "true_dissimilar": true_dissimilar,
        "false_accept": false_accept,
        "false_reject": false_reject,
        "false_accept_rate": false_accept / true_dissimilar if true_dissimilar > 0 else 0.0,
        "false_reject_rate": false_reject / true_similar if true_similar > 0 else 0.0,
        "overall_accuracy": correct / total if total > 0 else 0.0,
        "runtime_seconds": elapsed,
        "pairs_per_second": throughput,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Shouji on a synthetic dataset.")
    parser.add_argument("--input", type=str, required=True,
                        help="Input TSV dataset (pattern, reference, true_edits).")
    parser.add_argument("--threshold", type=int, default=10,
                        help="Edit distance threshold E.")
    parser.add_argument("--out", type=str, default="evaluation_results.json",
                        help="Where to write the metrics JSON file.")

    args = parser.parse_args()

    print(f"Running evaluation with threshold E={args.threshold}")
    print(f"Loading dataset: {args.input}")

    metrics = evaluate_dataset(args.input, args.threshold)

    print("\nEvaluation complete:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Write metrics to JSON
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()
