"""
Tests for the evaluation logic using a tiny handcrafted dataset.

We simulate a small dataset in-memory (no file I/O needed) by:
  - Writing a temporary TSV
  - Running Shouji + Edlib on it
  - Checking that metrics match expected values
"""

import os
import json
import tempfile

from scripts.run_evaluation import evaluate_dataset


def write_tsv(lines, filename):
    """Helper to write a small TSV dataset."""
    with open(filename, "w") as f:
        f.write("pattern\treference\ttrue_edits\n")
        for p, t, e in lines:
            f.write(f"{p}\t{t}\t{e}\n")


def test_evaluation_small_dataset():
    """
    Create a tiny dataset:
        Pair 1: 0 edits -> should be accepted
        Pair 2: 1 edit -> accepted for E=1
        Pair 3: 2 edits -> rejected for E=1
    Validate Shouji + edlib evaluation.
    """
    data = [
        ("ACGT", "ACGT", 0),
        ("ACGT", "AGGT", 1),
        ("ACGT", "AGTT", 2),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        tsv_path = os.path.join(tmp, "tiny.tsv")
        write_tsv(data, tsv_path)

        metrics = evaluate_dataset(tsv_path, threshold=1)

        # Basic checks
        assert metrics["total_pairs"] == 3
        assert metrics["true_similar"] == 2      # under E=1, first two are similar
        assert metrics["true_dissimilar"] == 1   # last one has 2 edits

        # Shouji should accept first two, reject last
        assert metrics["false_reject"] == 0      # should never reject truly similar
        assert metrics["false_accept"] == 0      # Shouji should reject the 2-edit case for E=1

        # Accuracy should be perfect on this small dataset
        assert metrics["overall_accuracy"] == 1.0

        # Runtime sanity check
        assert metrics["runtime_seconds"] >= 0
        assert metrics["pairs_per_second"] >= 0
