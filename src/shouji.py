"""
shouji.py
---------
Software reimplementation of the Shouji pre-alignment filter (Alser et al., 2019).

This version:
  - builds the neighborhood map within ±E diagonals,
  - scans sliding windows of size 4,
  - selects the 4-bit vector with the maximum zeros,
  - constructs the Shouji bit-vector,
  - accepts/rejects based on number of mismatches > E.

This is a CPU educational implementation intended for research replication.
"""

from typing import List, Tuple


def _best_window_vector(window_bits: List[List[int]]) -> List[int]:
    """
    Given a list of 4-bit vectors (one per diagonal), select the best one:
        1) maximize number of zeros
        2) break ties by preferring those with a leading zero
    """
    best = None
    best_zero_count = -1
    best_leading_zero = -1

    for vec in window_bits:
        zero_count = vec.count(0)
        leading_zero = 1 if vec[0] == 0 else 0

        if (zero_count > best_zero_count) or (
            zero_count == best_zero_count and leading_zero > best_leading_zero
        ):
            best = vec
            best_zero_count = zero_count
            best_leading_zero = leading_zero

    return best


def shouji_single(
    seq_p: str,
    seq_t: str,
    E: int,
    window_width: int = 4,
) -> Tuple[bool, List[int]]:
    """
    Apply the Shouji filter to a single pair of sequences.

    Parameters
    ----------
    seq_p : str
        Pattern (query) sequence.
    seq_t : str
        Text (reference substring) sequence.
    E : int
        Edit-distance threshold.
    window_width : int
        Sliding window width. Paper uses 4.

    Returns
    -------
    accept : bool
        True if pair may have edit distance <= E (alignment needed).
        False if pair is safely rejected.
    bitvector : List[int]
        Final Shouji bit-vector, where 0=match support, 1=mismatch support.

    Notes
    -----
    For simplicity, this version assumes equal-length sequences.
    Local/glocal handling can be added (paper trims boundaries).
    """

    if len(seq_p) != len(seq_t):
        raise ValueError(
            "This simple version requires equal sequence lengths. "
            "Pad or trim for local/glocal alignment if needed."
        )

    m = len(seq_p)
    shouji_bits = [1] * m  # default worst-case (all mismatches)

    # Range of diagonal offsets
    diag_offsets = list(range(-E, E + 1))

    # Slide window over columns 0..m-1
    for c in range(0, m):
        window_vectors = []

        # Collect 4-bit windows across each diagonal
        for d in diag_offsets:
            vec = []
            for col in range(c, c + window_width):
                i = col           # index in seq_p
                j = col + d       # matching index in seq_t

                if 0 <= i < m and 0 <= j < m:
                    bit = 0 if seq_p[i] == seq_t[j] else 1
                else:
                    # Out-of-bounds treated as mismatch (paper uses conservative rule)
                    bit = 1
                vec.append(bit)

            window_vectors.append(vec)

        best = _best_window_vector(window_vectors)

        # Update Shouji bit-vector:
        # only set zeros (matches) and never override existing zeros with ones.
        for k in range(window_width):
            pos = c + k
            if pos >= m:
                break
            if best[k] == 0:
                shouji_bits[pos] = 0

    # Count mismatches in final bitvector
    ones = sum(shouji_bits)
    accept = ones <= E
    return accept, shouji_bits


# ---------------------------------------------------------------
# Basic internal tests (not full unit tests)
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("Running internal checks...")

    ok, bv = shouji_single("ACGT", "ACGT", E=0)
    assert ok and sum(bv) == 0
    print("✓ Exact match OK")

    ok, bv = shouji_single("ACGTAC", "ACGGAC", E=1)
    assert ok
    print("✓ Single edit OK")

    ok, bv = shouji_single("ACGTAC", "ACGGTC", E=1)
    assert not ok
    print("✓ Reject too many edits OK")

    print("All internal checks passed.")
