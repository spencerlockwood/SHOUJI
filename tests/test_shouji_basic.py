"""
Unit tests for the Shouji algorithm (shouji_single)
"""

import pytest
from src.shouji.shouji import shouji_single


def test_exact_match_E0():
    """Exact matches should always be accepted when E=0."""
    seq = "ACGTACGT"
    accept, bits = shouji_single(seq, seq, E=0)
    assert accept
    assert sum(bits) == 0  # all positions should be marked as matches (0s)


def test_single_substitution_E1():
    """One mismatch should be accepted for E=1."""
    p = "ACGTACGT"
    t = "ACGTTCGT"  # one edit
    accept, bits = shouji_single(p, t, E=1)
    assert accept


def test_single_substitution_E0_reject():
    """One mismatch must be rejected if E=0."""
    p = "ACGTACGT"
    t = "ACGTTCGT"
    accept, bits = shouji_single(p, t, E=0)
    assert not accept


def test_two_edits_E1_reject():
    """Two mismatches should be rejected when E=1."""
    p = "ACGTACGT"
    t = "ACGTTCAT"
    accept, bits = shouji_single(p, t, E=1)
    assert not accept


def test_random_small_sequences():
    """Random small sequences: trivial check that code runs and produces bitvector of correct length."""
    p = "ACGTAC"
    t = "ACGGAC"
    accept, bits = shouji_single(p, t, E=2)
    assert isinstance(accept, bool)
    assert len(bits) == len(p)
