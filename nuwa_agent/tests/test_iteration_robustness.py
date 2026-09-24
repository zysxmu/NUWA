"""Controller-level regression guards for a noisy optimization trajectory."""

import os
import sys
from pathlib import Path


os.environ.setdefault("NUWA_API_KEY", "test-only")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iteration_controller import IterationController  # noqa: E402


def controller(*, declines=0, best_hv=0.23, stable_rounds=0):
    item = IterationController.__new__(IterationController)
    item._consecutive_hv_declines = declines
    item._best_hv = best_hv
    item._stable_rounds = stable_rounds
    return item


def decision(**updates):
    value = {
        "temperature": 0.5,
        "mutate_fraction": 0.4,
        "substitutions_per_candidate": 2,
        "parent_focus": "balanced",
        "evidence_ids": [],
        "rationale": "test",
    }
    value.update(updates)
    return value


def test_strategy_jumps_are_bounded_even_without_regression():
    ctl = controller()
    guarded, audit = ctl._guard_next_decision(
        decision(temperature=0.9, mutate_fraction=0.75),
        decision(),
        {"best_hv": 0.23, "hv": 0.23},
    )

    assert guarded["temperature"] == 0.7
    assert guarded["mutate_fraction"] == 0.55
    assert audit["applied"]
    assert {item["reason"] for item in audit["adjustments"]} == {"step_limit"}


def test_repeated_regression_forces_conservative_balanced_recovery():
    ctl = controller(declines=2)
    guarded, audit = ctl._guard_next_decision(
        decision(temperature=0.8, mutate_fraction=0.7,
                 substitutions_per_candidate=3, parent_focus="expression"),
        decision(temperature=0.5, mutate_fraction=0.6,
                 parent_focus="expression"),
        {"best_hv": 0.23, "hv": 0.20},
    )

    assert guarded["temperature"] == 0.5
    assert guarded["mutate_fraction"] == 0.5
    assert guarded["substitutions_per_candidate"] == 2
    assert guarded["parent_focus"] == "balanced"
    assert audit["best_hv_gap"] == 0.03

    summary = ctl._execution_summary(
        decision(temperature=0.8, mutate_fraction=0.7,
                 substitutions_per_candidate=3, parent_focus="expression"),
        guarded,
        audit,
    )
    assert "temperature=0.5" in summary
    assert "parent_focus='balanced'" in summary
    assert "regression_recovery" in summary


def test_step_limit_summary_does_not_claim_observed_regression():
    ctl = controller()
    requested = decision(temperature=0.9, mutate_fraction=0.75)
    guarded, audit = ctl._guard_next_decision(
        requested, decision(), {"best_hv": 0.23, "hv": 0.23}
    )
    summary = ctl._execution_summary(requested, guarded, audit)

    assert "step_limit" in summary
    assert "observed regression" not in summary.lower()


def test_small_rebound_far_below_best_is_not_convergence():
    ctl = controller(best_hv=0.232686, stable_rounds=2)
    assert not ctl._has_converged(round_num=9, feasible_count=34, hv=0.206792)
    assert ctl._has_converged(round_num=9, feasible_count=34, hv=0.228)


def test_cai_stage_diff_identifies_edits_to_evaluated_sequence():
    edits = IterationController._codon_edit_diff(
        "AUG UUU GCU UAA", "AUG UUC GCC UAA"
    )
    assert edits == [
        {"codon_index": 1, "old_codon": "UUU", "new_codon": "UUC"},
        {"codon_index": 2, "old_codon": "GCU", "new_codon": "GCC"},
    ]
