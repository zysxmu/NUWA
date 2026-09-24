"""Audit metadata must describe the values that were actually selected."""

import os
import sys
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("NUWA_API_KEY", "test-only")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iteration_controller import IterationController  # noqa: E402
import main as main_module  # noqa: E402
from batch_run import best_rank0  # noqa: E402
from main import (  # noqa: E402
    _atomic_write_text,
    _build_run_metadata,
    _canonical_rna_sequence,
    _codon_spaced_sequence,
    _save_reviewer_summary,
    _serialize_good_sequences,
    _validate_output_artifact,
)
from orchestrator import OrchestratorAgent  # noqa: E402
from species_resolver import CandidateSpecies, SpeciesInfo  # noqa: E402


def test_final_proxy_metadata_matches_selected_candidate():
    resolver_result = SpeciesInfo(
        domain="bacteria", class_id=5833, confidence="genus_match",
        matched_name="Escherichia coli O157:H7 str. Sakai",
        reason="same genus", candidates=[
            CandidateSpecies("Escherichia coli O157:H7 str. Sakai", 5833, "bacteria"),
            CandidateSpecies("Escherichia coli str. K-12 substr. MG1655", 5834, "bacteria"),
        ],
    )
    decision = {
        "selected_model": "bacteria", "class_id": 5834,
        "class_id_confidence": "genus_match",
        "chosen_candidate": {
            "name": "E. coli K-12", "class_id": 5834,
            "domain": "bacteria", "why": "reference strain",
        },
        "level2_species": {"reason": "K-12 selected as proxy"},
    }

    metadata = OrchestratorAgent._reconcile_species_selection(
        decision, resolver_result, "Escherichia coli"
    )

    assert decision["class_id"] == metadata["class_id"] == 5834
    assert metadata["matched_name"] == "Escherichia coli str. K-12 substr. MG1655"
    assert metadata["is_resolved"] is True
    assert metadata["proxy_resolved"] is True
    assert metadata["resolver_match"]["suggested_class_id"] == 5833


def test_unlisted_proxy_is_replaced_by_validated_resolver_candidate():
    resolver_result = SpeciesInfo(
        domain="bacteria", class_id=10, confidence="genus_match",
        matched_name="Valid species", reason="same genus",
        candidates=[CandidateSpecies("Valid species", 10, "bacteria")],
    )
    decision = {
        "selected_model": "bacteria", "class_id": 999,
        "chosen_candidate": {"name": "invented", "class_id": 999},
        "level2_species": {},
    }

    metadata = OrchestratorAgent._reconcile_species_selection(
        decision, resolver_result, "query"
    )

    assert decision["class_id"] == 10
    assert metadata["class_id"] == 10
    assert metadata["validation_note"]


def test_round_summary_exposes_only_active_mfe_window():
    controller = IterationController.__new__(IterationController)
    controller.constraint_bounds = {
        "cai_min": 0.7, "mfe_min": -400, "mfe_max": -100,
        "mfe_per_nt_min": -0.35, "mfe_per_nt_max": -0.2,
    }
    controller.prev_hv = 0.1
    controller._best_hv = 0.2
    controller._best_round = 2
    controller._consecutive_hv_declines = 0
    controller._stable_rounds = 0
    controller.history = []
    controller._elite_source = "current_round_2"
    controller.codon_table = {}
    result = SimpleNamespace(
        is_feasible=True, constraint_violations=[], sequence="AUG UAA",
    )

    summary = controller._build_round_summary(
        2, "M", "host", [result], [], 0.2, 0.1,
        {"temperature": 0.5}, {}, {"new_count": 1, "mutated_count": 0},
    )

    assert summary["constraint_modes"]["mfe"] == "per_nt"
    assert "mfe_min" not in summary["constraint_bounds"]
    assert summary["inactive_constraint_bounds"] == {
        "mfe_min": -400, "mfe_max": -100,
    }


def test_atomic_artifact_write_leaves_complete_file():
    target = Path(__file__).with_name("_atomic_write_test.json")
    try:
        _atomic_write_text(str(target), '{"complete": true}')
        assert target.read_text(encoding="utf-8") == '{"complete": true}'
        assert list(target.parent.glob(f"{target.name}.tmp-*")) == []
    finally:
        if target.exists():
            target.unlink()


def test_public_sequence_serialization_is_unspaced_and_explicit():
    assert _canonical_rna_sequence("AUG GCU TAA\n") == "AUGGCUUAA"
    assert _codon_spaced_sequence("AUGGCUUAA") == "AUG GCU UAA"

    records = _serialize_good_sequences([{"round": 1, "sequence": "AUG GCU UAA"}])
    assert records == [{
        "round": 1,
        "sequence": "AUGGCUUAA",
        "sequence_codon_spaced": "AUG GCU UAA",
        "sequence_length": 9,
        "sequence_alphabet": "RNA",
    }]


def test_formal_run_requires_an_explicit_seed(monkeypatch):
    monkeypatch.setattr(main_module, "STRICT_EVALUATION", True)
    monkeypatch.delenv("NUWA_RUN_SEED", raising=False)
    with pytest.raises(ValueError, match="Formal runs require NUWA_RUN_SEED"):
        main_module._configure_run_seed()


def test_run_metadata_contains_exact_input_and_content_addressed_models(monkeypatch):
    model_dir = Path(__file__).with_name("_audit_model_fixture")
    model_dir.mkdir(exist_ok=True)
    model_file = model_dir / "model.safetensors"
    model_file.write_bytes(b"formal-model-bytes")
    monkeypatch.setenv("NUWA_AUDIT_HASH_MODELS", "1")
    monkeypatch.setitem(main_module.NUWA_MODELS, "bacteria", {
        **main_module.NUWA_MODELS["bacteria"], "path": str(model_dir),
    })
    monkeypatch.setattr(main_module, "FINETUNED_TE_MODEL", str(model_dir))
    monkeypatch.setattr(main_module, "FINETUNED_STABILITY_MODEL", str(model_dir))
    monkeypatch.setattr(main_module, "FINETUNED_EXPR_MODEL", str(model_dir))

    try:
        metadata = _build_run_metadata(
            protein_seq="MTEST",
            host_organism="Escherichia coli",
            run_seed=42,
            timestamp="20260920_120000",
            selected_model="bacteria",
            class_id=5834,
            effective_constraints={"cai_min": 0.7},
            codon_table={"AUG": 1.0},
        )

        assert metadata["input"]["protein_sequence"] == "MTEST"
        assert metadata["input"]["protein_sha256"] == hashlib.sha256(b"MTEST").hexdigest()
        assert metadata["reproducibility"]["run_seed"] == 42
        assert metadata["software"]["source_fingerprint"]["source_tree_sha256"]
        expected_file_hash = hashlib.sha256(b"formal-model-bytes").hexdigest()
        assert metadata["model_artifacts"]["generation"]["files"][0]["sha256"] == expected_file_hash
        assert metadata["model_artifacts"]["generation"]["manifest_sha256"]
    finally:
        if model_file.exists():
            model_file.unlink()
        if model_dir.exists():
            model_dir.rmdir()


def test_batch_collector_selects_feasible_rank_zero_solution():
    target = Path(__file__).with_name("_batch_rank_test.json")
    payload = {
        "model_selection": {"selected_model": "bacteria"},
        "pareto_solutions": [
            {"rank": 1, "is_feasible": True, "objectives": {"Expression": 0.99}},
            {"rank": 0, "is_feasible": False, "objectives": {"Expression": 0.90}},
            {"rank": 0, "is_feasible": True, "objectives": {"Expression": 0.40}},
            {"rank": 0, "is_feasible": True, "objectives": {"Expression": 0.60}},
        ],
    }
    try:
        target.write_text(json.dumps(payload), encoding="utf-8")
        model, solution = best_rank0(str(target))
        assert model == "bacteria"
        assert solution["rank"] == 0
        assert solution["is_feasible"] is True
        assert solution["objectives"]["Expression"] == 0.60
    finally:
        if target.exists():
            target.unlink()


def test_reviewer_summary_exposes_seed_input_hash_and_canonical_sequence(monkeypatch):
    output_dir = str(Path(__file__).resolve().parent)
    monkeypatch.setattr(main_module, "OUTPUT_DIR", output_dir)
    timestamp = "reviewer_test"
    target = Path(output_dir) / f"nuwa_agent_reviewer_{timestamp}.md"
    output = {
        "run_seed": 42,
        "run_metadata": {
            "run_id": timestamp,
            "created_at_local": "2026-09-20T12:00:00+08:00",
            "audit_schema_version": "2.0",
            "input": {
                "protein_sequence": "M",
                "protein_length_aa": 1,
                "protein_sha256": "protein-hash",
                "host_organism": "host",
            },
            "reproducibility": {
                "run_seed": 42,
                "llm_model": "test-model",
                "strict_evaluation": True,
                "note": "test note",
            },
            "software": {"source_fingerprint": {"source_tree_sha256": "source-hash"}},
            "model_artifacts": {},
        },
        "model_selection": {"selected_model": "bacteria", "species_info": {}},
        "constraint_modes": {"mfe": "per_nt"},
        "constraint_evaluation": {"active_bounds": {}, "inactive_legacy_bounds": {}},
        "optimization_result": {
            "total_rounds": 1, "converged": True, "best_round": 1,
            "best_hv": 0.1, "last_round_hv": 0.1,
        },
        "iteration_history": [],
        "pareto_solutions": [{
            "rank": 0,
            "sequence": "AUGUAA",
            "sequence_length": 6,
            "objectives": {"TE": 0.1, "Stability": 0.2, "Expression": 0.3},
            "constraints": {
                "CAI": 0.8, "GC%": 0.4, "MFE_per_nt": -0.3,
                "max_stem_len": 1, "max_homopolymer": 2,
            },
            "is_feasible": True,
        }],
    }
    try:
        created = _save_reviewer_summary(output, timestamp)
        text = Path(created).read_text(encoding="utf-8")
        assert "Run seed: `42`" in text
        assert "Protein SHA256: `protein-hash`" in text
        assert "AUGUAA" in text
    finally:
        if target.exists():
            target.unlink()


def test_output_self_validation_checks_best_round_and_constraints():
    protein_hash = hashlib.sha256(b"M").hexdigest()
    output = {
        "run_metadata": {
            "input": {"protein_sequence": "M", "protein_sha256": protein_hash},
            "reproducibility": {"strict_evaluation": False, "run_seed": 42},
            "model_artifacts": {},
        },
        "model_selection": {
            "class_id": 1,
            "species_info": {"class_id": 1},
        },
        "constraint_evaluation": {"active_bounds": {
            "cai_min": 0.7, "gc_min": 0.3, "gc_max": 0.7,
            "mfe_per_nt_min": -0.35, "mfe_per_nt_max": -0.2,
            "max_stem_length": 33, "max_homopolymer": 6,
        }},
        "optimization_result": {"best_round": 1},
        "iteration_history": [{
            "round": 1,
            "decision_for_round": None,
            "pareto_front_details": [{"rank": 0, "sequence": "AUG UAA"}],
        }],
        "pareto_solutions": [{
            "rank": 0,
            "sequence": "AUGUAA",
            "sequence_length": 6,
            "constraints": {
                "CAI": 0.8, "GC%": 0.5, "MFE": -1.8, "MFE_per_nt": -0.3,
                "max_stem_len": 1, "max_homopolymer": 2,
            },
            "is_feasible": True,
        }],
    }

    validation = _validate_output_artifact(output)
    assert validation["passed"] is True
    assert validation["errors"] == []
    assert "best_round_solution_provenance" in validation["checks"]

    output["pareto_solutions"][0]["sequence"] = "AUG UAA"
    invalid = _validate_output_artifact(output)
    assert invalid["passed"] is False
    assert any("canonical RNA" in error for error in invalid["errors"])
