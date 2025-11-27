#!/usr/bin/env python3
"""
plot_shouji_results.py
----------------------

Load evaluation JSON files produced by run_evaluation.py
and generate plots similar to the SHOUJI paper.

Usage:
    python scripts/plot_shouji_results.py --results_dir results/ --out plots/
"""

import argparse
import os
import json
import matplotlib.pyplot as plt
from glob import glob

# Use non-interactive backend
import matplotlib
matplotlib.use("Agg")


def load_results(results_dir):
    """Load all JSON result files into a dictionary keyed by threshold."""
    json_files = sorted(glob(os.path.join(results_dir, "*.json")))
    results = {}

    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)
            E = data.get("threshold", None)
            if E is None:
                print(f"Warning: File {jf} missing 'threshold'; skipping.")
                continue
            results[E] = data

    if not results:
        raise ValueError("No usable JSON files found.")

    return results


def plot_false_accept_rate(results, outdir):
    Es = sorted(results.keys())
    fa = [results[E]["false_accept_rate"] for E in Es]

    plt.figure(figsize=(7,5))
    plt.plot(Es, fa, marker="o")
    plt.xlabel("Edit-distance Threshold (E)")
    plt.ylabel("False Accept Rate")
    plt.title("False Accept Rate vs. Edit Threshold")
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "false_accept_rate.png"))
    plt.close()


def plot_false_reject_rate(results, outdir):
    Es = sorted(results.keys())
    fr = [results[E]["false_reject_rate"] for E in Es]

    plt.figure(figsize=(7,5))
    plt.plot(Es, fr, marker="o")
    plt.xlabel("Edit-distance Threshold (E)")
    plt.ylabel("False Reject Rate")
    plt.title("False Reject Rate vs. Edit Threshold")
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "false_reject_rate.png"))
    plt.close()


def plot_accuracy(results, outdir):
    Es = sorted(results.keys())
    acc = [results[E]["overall_accuracy"] for E in Es]

    plt.figure(figsize=(7,5))
    plt.plot(Es, acc, marker="o")
    plt.xlabel("Edit-distance Threshold (E)")
    plt.ylabel("Overall Accuracy")
    plt.title("Accuracy vs. Edit Threshold")
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "accuracy_vs_E.png"))
    plt.close()


def plot_throughput(results, outdir):
    Es = sorted(results.keys())
    thr = [results[E]["pairs_per_second"] for E in Es]

    plt.figure(figsize=(7,5))
    plt.plot(Es, thr, marker="o")
    plt.xlabel("Edit-distance Threshold (E)")
    plt.ylabel("Throughput (pairs/sec)")
    plt.title("Shouji Throughput vs. Edit Threshold")
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "throughput_vs_E.png"))
    plt.close()


def plot_roc(results, outdir):
    Es = sorted(results.keys())
    FPR = [results[E]["false_accept_rate"] for E in Es]
    TPR = [1 - results[E]["false_reject_rate"] for E in Es]

    plt.figure(figsize=(6,6))
    plt.plot(FPR, TPR, marker="o")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC-style Curve for Shouji")
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "roc_curve.png"))
    plt.close()


def save_summary_table(results, outdir):
    summary_path = os.path.join(outdir, "summary_table.txt")
    with open(summary_path, "w") as f:
        f.write("E\tFA_rate\tFR_rate\tAccuracy\tThroughput\n")
        for E in sorted(results.keys()):
            m = results[E]
            f.write(
                f"{E}\t{m['false_accept_rate']:.4f}\t{m['false_reject_rate']:.4f}\t"
                f"{m['overall_accuracy']:.4f}\t{m['pairs_per_second']:.2f}\n"
            )
    print(f"Summary table saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing JSON files.")
    parser.add_argument("--out", type=str, default="plots",
                        help="Output directory for PNG plots.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    results = load_results(args.results_dir)

    plot_false_accept_rate(results, args.out)
    plot_false_reject_rate(results, args.out)
    plot_accuracy(results, args.out)
    plot_throughput(results, args.out)
    plot_roc(results, args.out)
    save_summary_table(results, args.out)

    print(f"All plots saved to {args.out}")


if __name__ == "__main__":
    main()
