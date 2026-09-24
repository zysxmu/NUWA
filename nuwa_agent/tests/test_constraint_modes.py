"""Focused checks for target-specific normalized-MFE feasibility."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from constraint_checker import ConstraintChecker  # noqa: E402


def result(sequence="AUG UUU UAA", mfe=-3.0):
    return SimpleNamespace(
        sequence=sequence,
        cai=0.8,
        gc_content=0.4,
        mfe=mfe,
        max_stem_len=1,
        max_homopolymer=1,
        is_feasible=False,
        constraint_violations=[],
    )


def test_mfe_per_nt_uses_whitespace_free_nucleotide_length_and_takes_precedence():
    checker = ConstraintChecker({
        "mfe_per_nt_min": -0.4,
        "mfe_per_nt_max": -0.2,
        "mfe_min": -100,
        "mfe_max": -50,
    })
    checked = checker.check(result(sequence="AUG \n UUU\tUAA", mfe=-3.0))
    assert checked.is_feasible  # -3 / 9 = -0.333, despite absolute MFE limits
    assert checked.constraint_violations == []

    too_weak = checker.check(result(mfe=-1.0))
    assert not too_weak.is_feasible
    assert too_weak.constraint_violations == ["MFE/nt=-0.1111 > -0.2"]

    too_folded = checker.check(result(mfe=-5.0))
    assert not too_folded.is_feasible
    assert too_folded.constraint_violations == ["MFE/nt=-0.5556 < -0.4 (over-folded)"]


def test_legacy_absolute_mfe_window_remains_available():
    checker = ConstraintChecker({"mfe_min": -4, "mfe_max": -2})
    assert checker.check(result(mfe=-3)).is_feasible
    checked = checker.check(result(mfe=-1))
    assert checked.constraint_violations == ["MFE=-1.0 > -2"]


@pytest.mark.parametrize("bounds", [
    {"mfe_per_nt_min": -0.4},
    {"mfe_per_nt_max": -0.2},
    {"mfe_per_nt_min": -0.2, "mfe_per_nt_max": -0.4},
    {"mfe_per_nt_min": float("nan"), "mfe_per_nt_max": -0.2},
    {"mfe_per_nt_min": -0.4, "mfe_per_nt_max": float("inf")},
    {"mfe_per_nt_min": True, "mfe_per_nt_max": -0.2},
])
def test_invalid_mfe_per_nt_window_fails_early(bounds):
    with pytest.raises(ValueError, match="MFE/nt|mfe_per_nt"):
        ConstraintChecker(bounds)


def test_normalized_mfe_rejects_empty_sequence():
    checker = ConstraintChecker({"mfe_per_nt_min": -0.4, "mfe_per_nt_max": -0.2})
    with pytest.raises(ValueError, match="empty sequence"):
        checker.check(result(sequence=" \n "))
