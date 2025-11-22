#!/usr/bin/env python3
"""
run_evaluation.py
-----------------

Evaluate the Shouji filter over a dataset and produce JSON metrics.

Supports:
  - Single threshold mode (--threshold)
  - Multi-threshold sweep (--thresholds E1 E2 E3 ...)

Output:
  One JSON file per threshold, written into --out_dir.

Example:
    python scripts/run_evaluation.py \
        --input data/synthetic/test.tsv \
        --thresholds 1 2 3 5 7 10 \
        --out_dir results/
"""

import argparse
import json
import time
import os

import edlib
from tqdm import tqdm

from src.shouji.shouji import shouji_single


def compute_edit_distance(p: str, t: str) -> int:
    """Return the Levenshtein edit distance using edlib."""
    result = edlib.align(p, t, mode="NW")
    return result["editDistance"]


def evaluate_dataset(input_tsv: str, threshold: int) -> dict:
    """Evaluate Shouji at a specific threshold."""
    total = 0
    true_similar = 0
    true_dissimilar = 0
    false_accept = 0
    false_reject = 0
    correct = 0

    start_time = time.time()

    # Count lines for progress bar
    with open(input_tsv, "r") as f:
        next(f)
        num_lines = sum(1 for _ in f)

    print(f"\nEvaluating threshold E={threshold} on {num_lines} pairs...")

    with open(input_tsv, "r") as f:
        header = next(f).strip().split("\t")
        if header != ["pattern", "reference", "true_edits"]:
            raise ValueError("TSV header must be: pattern\treference\ttrue_edits")

        for line in tqdm(f, total=num_lines, desc=f"E={threshold}"):
            p, t, _ = line.strip().split("\t")

            # Shouji filter decision
            accept, _ = shouji_single(p, t, E=threshold)

            # Ground truth distance
            true_dist = compute_edit_distance(p, t)

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

    elapsed = time.time() - start_time
    throughput = total / elapsed if elapsed > 0 else 0.0

    return {
        "threshold": threshold,
        "total_pairs": total,
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate Shouji over a dataset.")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--threshold", type=int, default=None,
                        help="Single threshold mode.")
    parser.add_argument("--thresholds", nargs="*", type=int,
                        help="List of thresholds, e.g. --thresholds 1 2 5 10")
    parser.add_argument("--out_dir", type=str, default="results",
                        help="Directory to write output JSON files.")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Determine thresholds
    if args.threshold is not None:
        thresholds = [args.threshold]
    elif args.thresholds:
        thresholds = args.thresholds
    else:
        raise ValueError("You must specify --threshold or --thresholds.")

    # Run evaluation for each threshold
    for E in thresholds:
        metrics = evaluate_dataset(args.input, E)

        out_path = os.path.join(args.out_dir, f"ci_results_E{E}.json")
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
