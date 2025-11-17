#!/usr/bin/env python3
"""
generate_synthetic.py
----------------------

Generate a synthetic dataset of (pattern, reference) sequence pairs
with known edit distances for testing the Shouji algorithm.

Output TSV format:
    pattern<TAB>reference<TAB>true_edit_distance

Usage:
    python scripts/generate_synthetic.py \
        --num_pairs 5000 \
        --length 150 \
        --max_edits 10 \
        --out data/synthetic/small_synth.tsv
"""

import os
import argparse
import random
from typing import Tuple, List

DNA = ("A", "C", "G", "T")


def random_dna(length: int) -> str:
    """Generate a random DNA sequence."""
    return "".join(random.choice(DNA) for _ in range(length))


def introduce_edits(seq: str, num_edits: int) -> str:
    """
    Introduce EXACTLY `num_edits` substitutions into the sequence.
    (No insertions or deletions, just mismatches.)
    """
    seq_list = list(seq)
    m = len(seq_list)

    # Choose unique edit positions
    edit_positions = random.sample(range(m), num_edits)

    for pos in edit_positions:
        old = seq_list[pos]
        choices = [b for b in DNA if b != old]
        seq_list[pos] = random.choice(choices)

    return "".join(seq_list)


def generate_pair(length: int, max_edits: int) -> Tuple[str, str, int]:
    """
    Generate a (pattern, reference, true_edits) tuple.
    Randomly choose an edit distance in [0, max_edits].
    """
    p = random_dna(length)
    E = random.randint(0, max_edits)
    t = introduce_edits(p, E)
    return p, t, E


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Shouji dataset.")
    parser.add_argument("--num_pairs", type=int, default=2000,
                        help="Number of sequence pairs to generate.")
    parser.add_argument("--length", type=int, default=150,
                        help="Sequence length.")
    parser.add_argument("--max_edits", type=int, default=10,
                        help="Maximum number of edits to introduce.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--out", type=str, default="data/synthetic/synthetic.tsv",
                        help="Output TSV file.")

    args = parser.parse_args()

    random.seed(args.seed)

    # Ensure output directory exists
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"Generating {args.num_pairs} synthetic pairs...")
    print(f" - length: {args.length}")
    print(f" - max edits: {args.max_edits}")
    print(f" - saving to: {args.out}")

    with open(args.out, "w") as f:
        f.write("pattern\treference\ttrue_edits\n")
        for _ in range(args.num_pairs):
            p, t, e = generate_pair(args.length, args.max_edits)
            f.write(f"{p}\t{t}\t{e}\n")

    print("Done.")


if __name__ == "__main__":
    main()
