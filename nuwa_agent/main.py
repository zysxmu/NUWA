"""NUWA-Agent — Entry Point

完整流程:
  1. SpeciesResolver (内嵌于 Orchestrator): 物种名 → (domain, class_id, confidence)
  2. Orchestrator: LLM CoT 三层决策 → 选模型 + class_id + 约束 (详细推理链 400+ 字)
  3. IterationController: 生成→评估→Pareto→专家讨论→Central LLM 决策→迭代
  4. 输出: Pareto最优序列 + 好序列队列 + 完整推理链 + 全部迭代历史

温度由每轮实际执行的策略记录；首轮使用基础退火计划。
收敛: 连续稳定、当前 HV 接近历史最优且 round > 最少轮数 → 停止
"""

import json
import argparse
import hashlib
import math
import os
import platform
import random
import re
import sys
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from orchestrator import OrchestratorAgent
from evaluator import MultiObjectiveEvaluator
from iteration_controller import IterationController
from config import (OUTPUT_DIR, CODON_TABLES_DIR, HV_CONVERGENCE_THRESHOLD,
                    MAX_ITERATION_ROUNDS, MIN_ITERATION_ROUNDS, STRICT_EVALUATION,
                    MULTIAGENT_ENABLED, LLM_MODEL, NUWA_MODELS,
                    FINETUNED_TE_MODEL, FINETUNED_STABILITY_MODEL,
                    FINETUNED_EXPR_MODEL)


AUDIT_SCHEMA_VERSION = "2.0"


AUDIT_HISTORY_FIELDS = (
    "round_summary",
    "discussion_transcript",
    "decision_for_round",
    "central_requested_decision",
    "guardrail_adjusted_decision",
    "central_decision",
    "applied_decision",
    "applied_decision_origin_round",
    "applied_decision_source",
    "discussion_fallback_used",
    "discussion_error",
    "discussion_skip_reason",
    "discussion_decision_source",
    "decision_guardrail",
    "score_correlations",
    "generation_directives",
    "next_round_execution_summary",
)


def _make_json_safe(o, _path=None):
    """只检测当前递归路径上的环, 不误伤 DAG 重复引用; 顺便把 numpy/torch/set/dataclass 转 JSON 安全类型。"""
    if _path is None:
        _path = set()
    if id(o) in _path:
        return "<circular-ref>"
    if isinstance(o, dict):
        _path = _path | {id(o)}
        return {str(k): _make_json_safe(v, _path) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        _path = _path | {id(o)}
        return [_make_json_safe(v, _path) for v in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    if hasattr(o, "tolist"):
        try:
            return _make_json_safe(o.tolist(), _path)
        except Exception:
            pass
    if hasattr(o, "__dict__"):
        return _make_json_safe({k: v for k, v in vars(o).items()
                                if not k.startswith("_")}, _path)
    return str(o)


def _redact_api_key(text: str) -> str:
    """Prevent the configured API key from entering result artifacts."""
    api_key = os.environ.get("NUWA_API_KEY", "")
    return text.replace(api_key, "[REDACTED_API_KEY]") if api_key else text


def _atomic_write_text(path: str, text: str) -> None:
    """Write a complete artifact or leave the previous file untouched."""
    target = os.path.abspath(path)
    temp_path = f"{target}.tmp-{os.getpid()}"
    try:
        with open(temp_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _configure_run_seed() -> int | None:
    """Seed the local generators used during selection, generation and mutation."""
    raw_seed = os.environ.get("NUWA_RUN_SEED")
    if raw_seed is None or not raw_seed.strip():
        if STRICT_EVALUATION:
            raise ValueError(
                "Formal runs require NUWA_RUN_SEED (integer 0..4294967295) "
                "so local stochastic operations can be reproduced"
            )
        return None
    try:
        run_seed = int(raw_seed)
    except ValueError as exc:
        raise ValueError("NUWA_RUN_SEED must be an integer from 0 to 4294967295") from exc
    if not 0 <= run_seed <= 0xFFFFFFFF:
        raise ValueError("NUWA_RUN_SEED must be an integer from 0 to 4294967295")

    import numpy as np
    import torch

    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    return run_seed


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_rna_sequence(sequence: str) -> str:
    """Return the machine-readable RNA representation used in public results."""
    return re.sub(r"\s+", "", sequence).upper().replace("T", "U")


def _codon_spaced_sequence(sequence: str) -> str:
    compact = _canonical_rna_sequence(sequence)
    if len(compact) % 3:
        raise ValueError(f"RNA sequence length must be divisible by 3, got {len(compact)}")
    return " ".join(compact[index:index + 3] for index in range(0, len(compact), 3))


def _source_fingerprint(source_dir: Path | None = None) -> dict:
    """Hash the exact Python source used for the run without requiring Git."""
    root = (source_dir or Path(__file__).resolve().parent).resolve()
    files = sorted(path for path in root.glob("*.py") if path.is_file())
    digest = hashlib.sha256()
    entries = []
    for path in files:
        content = path.read_bytes()
        relative = path.relative_to(root).as_posix()
        file_hash = _sha256_bytes(content)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
        entries.append({"path": relative, "sha256": file_hash, "size_bytes": len(content)})
    return {
        "algorithm": "sha256",
        "source_tree_sha256": digest.hexdigest(),
        "file_count": len(entries),
        "files": entries,
    }


def _dependency_versions() -> dict:
    versions = {}
    for distribution in ("numpy", "torch", "transformers", "cai2", "ViennaRNA"):
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _artifact_identity(path_value: str, content_hashes: bool) -> dict:
    """Record a deterministic model/checkpoint manifest; hash contents for formal runs."""
    path = Path(path_value).expanduser().resolve()
    result = {
        "path": str(path),
        "checkpoint": path.name,
        "exists": path.exists(),
        "content_hashes_enabled": content_hashes,
        "files": [],
    }
    if not path.exists():
        result["manifest_sha256"] = None
        return result

    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    manifest_digest = hashlib.sha256()
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        size = item.stat().st_size
        file_hash = _sha256_file(item) if content_hashes else None
        entry = {"path": relative, "size_bytes": size, "sha256": file_hash}
        result["files"].append(entry)
        manifest_digest.update(relative.encode("utf-8"))
        manifest_digest.update(b"\0")
        manifest_digest.update(str(size).encode("ascii"))
        manifest_digest.update(b"\0")
        if file_hash:
            manifest_digest.update(file_hash.encode("ascii"))
        manifest_digest.update(b"\0")
    result["manifest_sha256"] = manifest_digest.hexdigest()
    return result


def _build_run_metadata(protein_seq: str, host_organism: str, run_seed: int | None,
                        timestamp: str, selected_model: str, class_id: int,
                        effective_constraints: dict, codon_table: dict | None,
                        protein_id: str | None = None) -> dict:
    hash_models = os.environ.get(
        "NUWA_AUDIT_HASH_MODELS", "1" if STRICT_EVALUATION else "0"
    ).strip().lower() not in {"0", "false", "no", "off"}
    if STRICT_EVALUATION and not hash_models:
        raise ValueError(
            "Formal runs require NUWA_AUDIT_HASH_MODELS=1 so checkpoint contents "
            "are recorded with SHA256"
        )
    model_paths = {
        "generation": NUWA_MODELS[selected_model]["path"],
        "te": FINETUNED_TE_MODEL,
        "stability": FINETUNED_STABILITY_MODEL,
        "expression": FINETUNED_EXPR_MODEL,
    }
    codon_payload = json.dumps(
        codon_table or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    model_artifacts = {
        name: _artifact_identity(path, hash_models) for name, path in model_paths.items()
    }
    missing_artifacts = [
        name for name, artifact in model_artifacts.items() if not artifact["exists"]
    ]
    if STRICT_EVALUATION and missing_artifacts:
        raise FileNotFoundError(
            "Formal-run model artifacts are missing: " + ", ".join(missing_artifacts)
        )
    metadata = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "run_id": timestamp,
        "created_at_local": datetime.now().astimezone().isoformat(),
        "input": {
            "protein_id": protein_id,
            "protein_sequence": protein_seq,
            "protein_length_aa": len(protein_seq),
            "protein_sha256": _sha256_bytes(protein_seq.encode("ascii")),
            "host_organism": host_organism,
        },
        "reproducibility": {
            "run_seed": run_seed,
            "strict_evaluation": STRICT_EVALUATION,
            "multiagent_enabled": MULTIAGENT_ENABLED,
            "llm_model": LLM_MODEL,
            "model_content_hashes_enabled": hash_models,
            "note": (
                "The seed controls local Python/NumPy/PyTorch stochastic operations. "
                "Remote LLM and some accelerator kernels may still be nondeterministic; "
                "complete prompts and responses are retained in the audit log."
            ),
        },
        "selection": {
            "selected_model": selected_model,
            "class_id": class_id,
            "effective_constraints": effective_constraints,
        },
        "software": {
            "declared_version": os.environ.get("NUWA_CODE_VERSION"),
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": _dependency_versions(),
            "source_fingerprint": _source_fingerprint(),
        },
        "model_artifacts": model_artifacts,
        "codon_table": {
            "loaded": bool(codon_table),
            "entry_count": len(codon_table or {}),
            "canonical_sha256": _sha256_bytes(codon_payload),
        },
        "sequence_serialization": {
            "canonical_field": "sequence",
            "canonical_format": "uppercase_unspaced_RNA",
            "display_field": "sequence_codon_spaced",
            "length_excludes_whitespace": True,
            "audit_history_note": "Raw generation history may retain codon-spaced RNA for provenance.",
        },
    }
    return metadata


def _serialize_good_sequences(queue: list) -> list:
    serialized = []
    for item in queue:
        record = dict(item)
        original = str(record.get("sequence", ""))
        compact = _canonical_rna_sequence(original)
        record["sequence"] = compact
        record["sequence_codon_spaced"] = _codon_spaced_sequence(compact)
        record["sequence_length"] = len(compact)
        record["sequence_alphabet"] = "RNA"
        serialized.append(record)
    return serialized


def _validate_output_artifact(output: dict) -> dict:
    """Cross-check reviewer-visible invariants before committing an artifact."""
    errors = []
    checks = []
    metadata = output.get("run_metadata", {})
    inputs = metadata.get("input", {})
    reproducibility = metadata.get("reproducibility", {})
    protein = inputs.get("protein_sequence", "")
    expected_protein_hash = _sha256_bytes(protein.encode("ascii")) if protein else None
    if not protein or inputs.get("protein_sha256") != expected_protein_hash:
        errors.append("input protein sequence/hash mismatch")
    else:
        checks.append("input_protein_sha256")

    if reproducibility.get("strict_evaluation") and reproducibility.get("run_seed") is None:
        errors.append("strict run has no run seed")
    else:
        checks.append("run_seed_recorded")

    artifacts = metadata.get("model_artifacts", {})
    if reproducibility.get("strict_evaluation"):
        for name in ("generation", "te", "stability", "expression"):
            artifact = artifacts.get(name, {})
            if not artifact.get("exists") or not artifact.get("manifest_sha256"):
                errors.append(f"model artifact missing or unhashed: {name}")
            elif any(not item.get("sha256") for item in artifact.get("files", [])):
                errors.append(f"model artifact contains unhashed files: {name}")
        if not any(message.startswith("model artifact") for message in errors):
            checks.append("model_artifact_sha256")

    selection = output.get("model_selection", {})
    species = selection.get("species_info", {})
    if species and selection.get("class_id") != species.get("class_id"):
        errors.append("selected class_id differs from final species metadata")
    else:
        checks.append("species_class_id_consistency")

    history = output.get("iteration_history", [])
    decision_fields = (
        "temperature", "mutate_fraction", "substitutions_per_candidate", "parent_focus"
    )
    for index, round_record in enumerate(history):
        round_number = round_record.get("round")
        if index:
            previous = history[index - 1].get("guardrail_adjusted_decision") or {}
            applied = round_record.get("applied_decision") or {}
            for field in decision_fields:
                if applied.get(field) != previous.get(field):
                    errors.append(
                        f"round {round_number} applied {field} does not match prior adjusted decision"
                    )
        expected_target = round_number + 1 if index < len(history) - 1 else None
        if round_record.get("decision_for_round") != expected_target:
            errors.append(f"round {round_number} has invalid decision_for_round")
    if not any(message.startswith("round ") for message in errors):
        checks.append("decision_timeline_consistency")

    bounds = output.get("constraint_evaluation", {}).get("active_bounds", {})
    solutions = output.get("pareto_solutions", [])
    if not solutions:
        errors.append("no returned Pareto solutions")
    for index, solution in enumerate(solutions, start=1):
        prefix = f"solution S{index:03d}"
        sequence = solution.get("sequence", "")
        if (sequence != _canonical_rna_sequence(sequence)
                or any(base not in "ACGU" for base in sequence)
                or len(sequence) != solution.get("sequence_length")
                or len(sequence) % 3):
            errors.append(f"{prefix} has invalid canonical RNA serialization")
        elif sequence[-3:] not in {"UAA", "UAG", "UGA"}:
            errors.append(f"{prefix} has no terminal stop codon")

        values = solution.get("constraints", {})
        constraint_checks = (
            ("CAI", values.get("CAI"), bounds.get("cai_min"), None),
            ("GC%", values.get("GC%"), bounds.get("gc_min"), bounds.get("gc_max")),
            ("MFE", values.get("MFE"), bounds.get("mfe_min"), bounds.get("mfe_max")),
            ("MFE_per_nt", values.get("MFE_per_nt"), bounds.get("mfe_per_nt_min"), bounds.get("mfe_per_nt_max")),
            ("max_stem_len", values.get("max_stem_len"), None, bounds.get("max_stem_length")),
            ("max_homopolymer", values.get("max_homopolymer"), None, bounds.get("max_homopolymer")),
        )
        for name, value, minimum, maximum in constraint_checks:
            if value is None:
                errors.append(f"{prefix} lacks {name}")
            elif minimum is not None and value < minimum:
                errors.append(f"{prefix} violates {name} minimum")
            elif maximum is not None and value > maximum:
                errors.append(f"{prefix} violates {name} maximum")
        if not solution.get("is_feasible"):
            errors.append(f"{prefix} is not feasible")

    best_round = output.get("optimization_result", {}).get("best_round")
    if solutions and best_round is not None:
        source_record = next(
            (item for item in history if item.get("round") == best_round), None
        )
        source_sequences = {
            _canonical_rna_sequence(item.get("sequence", ""))
            for item in (source_record or {}).get("pareto_front_details", [])
            if item.get("rank") == 0
        }
        returned_sequences = {item.get("sequence") for item in solutions}
        if not returned_sequences.issubset(source_sequences):
            errors.append("returned Pareto set does not come from recorded best round")
        else:
            checks.append("best_round_solution_provenance")

    if solutions and not any(message.startswith("solution ") for message in errors):
        checks.extend(["canonical_sequences", "active_constraints", "pareto_feasibility"])

    return {
        "passed": not errors,
        "validated_at_local": datetime.now().astimezone().isoformat(),
        "checks": sorted(set(checks)),
        "errors": errors,
    }


def _experiment_constraints() -> dict:
    """Read explicitly configured experimental constraints without inventing a window."""
    constraints = {}
    mfe_min_raw = os.environ.get("NUWA_MFE_PER_NT_MIN")
    mfe_max_raw = os.environ.get("NUWA_MFE_PER_NT_MAX")
    if (mfe_min_raw is None) != (mfe_max_raw is None):
        raise ValueError("Set both NUWA_MFE_PER_NT_MIN and NUWA_MFE_PER_NT_MAX, or neither")
    if mfe_min_raw is not None:
        try:
            mfe_min = float(mfe_min_raw)
            mfe_max = float(mfe_max_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("NUWA_MFE_PER_NT_MIN/MAX must be finite numbers") from exc
        if not all(map(math.isfinite, (mfe_min, mfe_max))) or mfe_min >= mfe_max:
            raise ValueError("NUWA_MFE_PER_NT_MIN/MAX must be finite with min < max")
        constraints.update(mfe_per_nt_min=mfe_min, mfe_per_nt_max=mfe_max)

    return constraints


def _mfe_per_nt(mfe: float, sequence: str) -> float | None:
    nt_length = len("".join(sequence.split()))
    return round(mfe / nt_length, 6) if nt_length else None


def _normalize_protein_input(value: str) -> str:
    """Validate an unambiguous amino-acid sequence; retain no terminal stop marker."""
    sequence = re.sub(r"\s+", "", value).upper()
    if sequence.endswith("*"):
        sequence = sequence[:-1]
    if not sequence:
        raise ValueError("Target protein sequence is empty")
    invalid = sorted(set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"))
    if invalid:
        raise ValueError(f"Target protein contains unsupported residues: {''.join(invalid)}")
    return sequence


def _read_protein_fasta(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip() and not line.startswith(">")]
    return _normalize_protein_input("".join(lines))


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="NUWA-Agent host-aware mRNA optimization")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--protein", help="target amino-acid sequence")
    source.add_argument("--protein-file", help="single-record protein FASTA file")
    parser.add_argument("--protein-id", help="stable source identifier for audit/batch runs")
    parser.add_argument("--host", help="target host species")
    parser.add_argument("--gc-min", type=float)
    parser.add_argument("--gc-max", type=float)
    args = parser.parse_args(argv)
    if (args.gc_min is None) != (args.gc_max is None):
        parser.error("--gc-min and --gc-max must be supplied together")
    if args.gc_min is not None and not 0 <= args.gc_min < args.gc_max <= 1:
        parser.error("GC bounds must satisfy 0 <= min < max <= 1")
    return args


def load_codon_table(organism: str) -> dict:
    """加载宿主参考密码子权重表

    从 CODON_TABLES_DIR 加载对应物种的密码子权重 JSON，
    提取 codon_usage_weights 字段供 cai2.CAI 使用。

    物种名映射规则:
      Escherichia coli → escherichia_coli
      Homo sapiens     → homo_sapiens
      S. cerevisiae    → saccharomyces_cerevisiae
      其他 → 模糊匹配文件名，无匹配返回 None (使用 fallback)
    """
    if not organism:
        return None

    # 常用物种名 → 文件名映射
    KNOWN_MAP = {
        "escherichia_coli": "escherichia_coli",
        "escherichia coli": "escherichia_coli",
        "e. coli": "escherichia_coli",
        "ecoli": "escherichia_coli",
        "homo_sapiens": "homo_sapiens",
        "homo sapiens": "homo_sapiens",
        "human": "homo_sapiens",
        "saccharomyces_cerevisiae": "saccharomyces_cerevisiae",
        "saccharomyces cerevisiae": "saccharomyces_cerevisiae",
        "s. cerevisiae": "saccharomyces_cerevisiae",
        "yeast": "saccharomyces_cerevisiae",
    }

    organism_slug = organism.lower().strip()
    filename = KNOWN_MAP.get(organism_slug)

    if filename is None:
        # 模糊匹配: 检查文件名是否包含物种关键词
        table_dir = CODON_TABLES_DIR
        if os.path.isdir(table_dir):
            for fname in os.listdir(table_dir):
                if fname.endswith(".json") and not fname.startswith("_"):
                    # 下划线分隔的物种名 → 关键词
                    key_words = fname.replace(".json", "").split("_")
                    # 物种名字段也在文件名中
                    org_words = organism_slug.replace(" ", "_").split("_")
                    overlap = sum(1 for w in org_words if w in key_words)
                    if overlap >= 1:
                        filename = fname.replace(".json", "")
                        break

    if filename is None:
        print(f"  [Warning] 密码子表未匹配到物种 '{organism}', 使用 CAI fallback")
        return None

    table_path = os.path.join(CODON_TABLES_DIR, f"{filename}.json")
    if not os.path.exists(table_path):
        print(f"  [Warning] 密码子表文件不存在: {table_path}, 使用 CAI fallback")
        return None

    try:
        with open(table_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        weights = data.get("codon_usage_weights", {})
        if weights:
            print(f"  [OK] 加载密码子表: {data.get('species', filename)} "
                  f"({len(weights)} codons, {data.get('total_codons', 0):,} total)")
        return weights
    except Exception as e:
        print(f"  [Warning] 密码子表加载失败: {e}, 使用 CAI fallback")
        return None


def main(argv=None):
    args = _parse_args(argv)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_seed = _configure_run_seed()
    experimental_constraints = _experiment_constraints()
    if STRICT_EVALUATION and "mfe_per_nt_min" not in experimental_constraints:
        raise ValueError(
            "Formal runs require target-specific NUWA_MFE_PER_NT_MIN and "
            "NUWA_MFE_PER_NT_MAX values. Set NUWA_STRICT_EVALUATION=0 only for a demo."
        )
    print("=" * 60)
    print("  NUWA-Agent — CoT 三层决策选模型 + 迭代多目标优化")
    print("  Phase 1: SpeciesResolver + CoT 三层决策 → 选模型 + class_id + 约束")
    print(f"  Phase 2: 迭代优化 (生成→评估→Pareto→多 Agent 讨论→Central LLM 决策) — 最多{MAX_ITERATION_ROUNDS}轮")
    print("  Phase 3: Pareto 最优输出 + 好序列队列")
    if run_seed is not None:
        print(f"  Run seed: {run_seed}")
    print("=" * 60)

    # ====== 输入 ======
    if args.protein_file:
        protein_seq = _read_protein_fasta(args.protein_file)
    elif args.protein:
        protein_seq = _normalize_protein_input(args.protein)
    else:
        protein_seq = _normalize_protein_input(input("\n请输入目标蛋白序列: ").strip())
    host_organism = (args.host or input("请输入宿主物种 (如 Escherichia coli): ")).strip()
    if not host_organism:
        raise ValueError("Host organism is required")

    user_constraints = {}
    if args.gc_min is not None:
        user_constraints.update(gc_min=args.gc_min, gc_max=args.gc_max)
    elif not (args.protein or args.protein_file or args.host):
        gc_input = input("GC 含量约束 (如 0.3-0.7, 回车跳过): ").strip()
        if gc_input:
            gc_min, gc_max = map(float, gc_input.split("-"))
            if not 0 <= gc_min < gc_max <= 1:
                raise ValueError("GC bounds must satisfy 0 <= min < max <= 1")
            user_constraints.update(gc_min=gc_min, gc_max=gc_max)
    user_constraints.update(experimental_constraints)

    # ====== Phase 1: SpeciesResolver + CoT 三层决策 ======
    print("\n" + "=" * 60)
    print("[Phase 1] SpeciesResolver + Orchestrator Agent — CoT 三层决策...")

    agent = OrchestratorAgent()
    decision = agent.run(protein_seq, host_organism, user_constraints or None)
    # Explicit user/experiment constraints take precedence over LLM suggestions.
    decision["constraint_bounds"].update(user_constraints)

    selected_model = decision["selected_model"]
    class_id = decision.get("class_id", 0)
    class_id_confidence = decision.get("class_id_confidence", "default")

    print(f"\n  === Decision Results ===")
    print(f"  Selected model: {selected_model}")
    print(f"  Class ID: {class_id} (confidence: {class_id_confidence})")
    print(f"  Rationale: {decision.get('rationale', 'N/A')}")

    # 物种解析详情
    species_info = decision.get("species_info", {})
    if species_info:
        print(f"  Species: matched='{species_info.get('matched_name', 'N/A')}', "
              f"domain={species_info.get('domain', 'N/A')}, "
              f"confidence={species_info.get('confidence', 'N/A')}")

    print(f"  Reasoning: {decision.get('reasoning_chain', 'N/A')}")
    print(f"  Constraints: {json.dumps(decision['constraint_bounds'], indent=2)}")

    # ====== Phase 2: 迭代优化 ======
    print("\n" + "=" * 60)
    print(f"[Phase 2] Iterative Optimization Loop... (最大 {MAX_ITERATION_ROUNDS} 轮, 最少 {MIN_ITERATION_ROUNDS} 轮, "
          f"稳定阈值 |ΔHV| < {HV_CONVERGENCE_THRESHOLD}，且须接近历史最优)")
    # 加载宿主参考密码子表
    codon_table = load_codon_table(host_organism)
    mfe_mode = "per_nt" if "mfe_per_nt_min" in user_constraints else "absolute"
    effective_constraint_bounds = dict(decision["constraint_bounds"])
    inactive_constraint_bounds = {}
    if mfe_mode == "per_nt":
        for key in ("mfe_min", "mfe_max"):
            if key in effective_constraint_bounds:
                inactive_constraint_bounds[key] = effective_constraint_bounds.pop(key)

    print("  Building formal-run provenance and checkpoint SHA256 manifests...")
    run_metadata = _build_run_metadata(
        protein_seq=protein_seq,
        host_organism=host_organism,
        run_seed=run_seed,
        timestamp=timestamp,
        selected_model=selected_model,
        class_id=class_id,
        effective_constraints=effective_constraint_bounds,
        codon_table=codon_table,
        protein_id=args.protein_id,
    )

    # 创建评估器 (外部生物信息学工具 + 微调 NUWA 回归模型)
    # domain 和 class_id 传入以匹配微调模型的 token_type_ids
    evaluator = MultiObjectiveEvaluator(
        codon_table=codon_table,
        host_organism=host_organism,
        domain=selected_model,  # "bacteria" / "eukaryote" / "archaea"
        class_id=class_id,      # 来自 SpeciesResolver + CoT 决策
        strict=STRICT_EVALUATION,
    )
    backend_validation = evaluator.validate_backends()

    controller = IterationController(
        selected_model=decision["selected_model"],
        constraint_bounds=decision["constraint_bounds"],
        evaluator=evaluator,
        class_id=class_id,      # 来自 SpeciesResolver + CoT 决策
        codon_table=codon_table,  # 宿主密码子表 (用于 CAI 定向优化)
    )

    result = controller.run(protein_seq, host_organism)

    # ====== Phase 3: 输出 ======
    print("\n" + "=" * 60)
    print("[Phase 3] Output")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    history = result.get("history", [])
    # The final temperature is the one that actually generated the final round.
    final_temperature = history[-1].get("temperature") if history else None

    output = {
        "run_seed": run_seed,
        "input_protein_id": args.protein_id,
        "run_metadata": run_metadata,
        "requested_constraints": user_constraints,
        "constraint_modes": {
            "mfe": mfe_mode,
        },
        "constraint_evaluation": {
            "active_bounds": effective_constraint_bounds,
            "inactive_legacy_bounds": inactive_constraint_bounds,
            "note": ("MFE/nt bounds take precedence; absolute MFE bounds are not evaluated."
                     if mfe_mode == "per_nt" else
                     "Absolute MFE bounds are active."),
        },
        "model_selection": {
            "selected_model": decision["selected_model"],
            "model_name": decision.get("model_name", ""),
            "class_id": class_id,
            "class_id_confidence": class_id_confidence,
            "rationale": decision.get("rationale", ""),
            "reasoning_chain": decision.get("reasoning_chain", ""),
            "host_analysis": decision.get("host_analysis", {}),
            "model_discussion": decision.get("model_discussion", {}),
            "level1_domain": decision.get("level1_domain", {}),
            "level2_species": decision.get("level2_species", {}),
            "level3_cross_domain": decision.get("level3_cross_domain", {}),
            "chosen_candidate": decision.get("chosen_candidate"),
            "species_info": species_info,
        },
        "constraint_bounds": decision["constraint_bounds"],
        "evaluation_backends": {
            "validation": backend_validation,
            "actual_usage": evaluator.backend_usage(),
        },
        "optimization_result": {
            "total_rounds": result["total_rounds"],
            "converged": result["converged"],
            "final_hv": result["final_hv"],
            "best_hv": result.get("best_hv", result["final_hv"]),
            "last_round_hv": result.get("last_round_hv", result["final_hv"]),
            "best_round": result.get("best_round"),
            "final_temperature": final_temperature,
        },
        # 每轮收集的好序列队列 (可行 Pareto 前沿, 去重)
        "good_sequences_queue": _serialize_good_sequences(
            result.get("good_sequences_queue", [])
        ),
        # 完整序列, 不截断
        "pareto_solutions": [
            {
                "rank": s.rank,
                "crowding_distance": round(s.crowding_distance, 6),
                "sequence": _canonical_rna_sequence(s.result.sequence),
                "sequence_codon_spaced": _codon_spaced_sequence(s.result.sequence),
                "sequence_length": len(_canonical_rna_sequence(s.result.sequence)),
                "sequence_alphabet": "RNA",
                "sequence_format": "uppercase_unspaced",
                "objectives": {
                    "TE": round(s.result.te_score, 6),
                    "Stability": round(s.result.stability_score, 6),
                    "Expression": round(s.result.expression_score, 6),
                },
                "constraints": {
                    "CAI": round(s.result.cai, 6),
                    "GC%": round(s.result.gc_content, 6),
                    "MFE": round(s.result.mfe, 3),
                    "MFE_per_nt": _mfe_per_nt(s.result.mfe, s.result.sequence),
                    "max_stem_len": s.result.max_stem_len,
                    "max_homopolymer": s.result.max_homopolymer,
                },
                "is_feasible": s.result.is_feasible,
            }
            for s in result["best_solutions"]
        ],
        # 完整迭代历史 (不截断反馈/序列)
        "iteration_history": [
            {
                "round": h["round"],
                "temperature": h["temperature"],
                "num_candidates": h["num_candidates"],
                "feasible_count": h["feasible_count"],
                "pareto_front_size": h["pareto_front_size"],
                "hv": h["hv"],
                "hv_improvement": h["hv_improvement"],
                # 完整反馈分析
                "feedback": h["feedback"],
                "weaknesses": h["weaknesses"],
                "suggestions": h["suggestions"],
                "overall_assessment": h["overall_assessment"],
                # 生成元数据
                "generation_meta": h["generation_meta"],
                # 本轮全部候选打分
                "candidate_scores": h["candidate_scores"],
                # 本轮 Pareto 前沿
                "pareto_front_details": h["pareto_front_details"],
                # 精英序列
                "elite_sequences": h["elite_sequences"],
                # 多 Agent 审计信息；旧结果可能没有这些字段。
                **{key: h[key] for key in AUDIT_HISTORY_FIELDS if key in h},
            }
            for h in history
        ],
        "full_llm_log": decision.get("full_log", []),
    }
    output["artifact_validation"] = _validate_output_artifact(output)
    if not output["artifact_validation"]["passed"]:
        print("  [AUDIT WARNING] Output self-validation failed:")
        for error in output["artifact_validation"]["errors"]:
            print(f"    - {error}")

    output_path = os.path.join(OUTPUT_DIR, f"nuwa_agent_{timestamp}.json")
    _atomic_write_text(
        output_path,
        _redact_api_key(json.dumps(_make_json_safe(output), ensure_ascii=False, indent=2)),
    )

    # ====== 保存推理链 Markdown ======
    _save_chain_markdown(decision, result, output, timestamp)
    _save_reviewer_summary(output, timestamp)

    # 摘要
    print(f"\nModel: {decision['selected_model']} (class_id={class_id}, confidence={class_id_confidence})")
    print(f"Rounds: {result['total_rounds']}, Converged: {result['converged']}")
    print(f"Best HV: {result.get('best_hv', result['final_hv']):.4f}; "
          f"last-round HV: {result.get('last_round_hv', result['final_hv']):.4f}")
    good_count = len(output.get("good_sequences_queue", []))
    print(f"Good sequences queue: {good_count} 条 (每轮收集的可行 Pareto 前沿解)")
    print(f"\nTop-3 Pareto solutions:")
    for s in output["pareto_solutions"][:3]:
        o = s["objectives"]
        c = s["constraints"]
        print(f"  Rank {s['rank']}: TE={o['TE']:.3f} Stab={o['Stability']:.3f} Expr={o['Expression']:.3f} "
              f"CAI={c['CAI']:.3f} GC={c['GC%']} Feasible={s['is_feasible']}")

    print(f"\nJSON saved: {output_path}")
    print(f"Chain MD saved: {os.path.join(OUTPUT_DIR, f'nuwa_agent_chain_{timestamp}.md')}")
    print(f"Reviewer summary saved: {os.path.join(OUTPUT_DIR, f'nuwa_agent_reviewer_{timestamp}.md')}")
    print("=" * 60)


def _append_discussion_audit(lines: list[str], round_history: dict) -> None:
    """Render a short, readable view of the full JSON deliberation audit."""
    if not any(key in round_history for key in AUDIT_HISTORY_FIELDS):
        return

    lines.extend(["### 多 Agent 讨论与决策记录", ""])
    if "applied_decision" in round_history:
        lines.append("**本轮实际执行参数**：")
        lines.extend(["```json", json.dumps(round_history["applied_decision"], ensure_ascii=False, indent=2), "```", ""])
        lines.append(
            f"**本轮参数来源**：{round_history.get('applied_decision_source', 'N/A')}"
            f"（来源轮次：{round_history.get('applied_decision_origin_round', 'baseline')}）"
        )
        lines.append("")

    if "round_summary" in round_history:
        lines.append("**本轮讨论依据**：")
        lines.extend(["```json", json.dumps(round_history["round_summary"], ensure_ascii=False, indent=2), "```", ""])

    transcript = round_history.get("discussion_transcript") or []
    if transcript:
        lines.append(f"**专家讨论**（{len(transcript)} 条；展开查看完整发言）：")
        lines.append("")
        for index, entry in enumerate(transcript, 1):
            if isinstance(entry, dict):
                agent = entry.get("agent", "Unknown")
                phase = entry.get("phase", "discussion")
            else:
                agent, phase = "Unknown", "discussion"
            lines.append(f"<details><summary>{index}. {agent} · {phase}</summary>")
            lines.extend(["", "```json", json.dumps(entry, ensure_ascii=False, indent=2), "```", "", "</details>", ""])

    target_round = round_history.get("decision_for_round")
    requested = round_history.get("central_requested_decision")
    adjusted = round_history.get("guardrail_adjusted_decision")
    if requested is not None:
        lines.append(f"**Central LLM 原始请求（用于 Round {target_round}）**：")
        lines.extend(["```json", json.dumps(requested, ensure_ascii=False, indent=2), "```", ""])
    if adjusted is not None:
        lines.append(f"**控制器校验后的可执行决定（用于 Round {target_round}）**：")
        lines.extend(["```json", json.dumps(adjusted, ensure_ascii=False, indent=2), "```", ""])
    elif round_history.get("central_decision") is not None:
        lines.append(f"**下一轮可执行决定（用于 Round {target_round}）**：")
        lines.extend(["```json", json.dumps(round_history["central_decision"], ensure_ascii=False, indent=2), "```", ""])
    elif round_history.get("generation_directives"):
        lines.append("**基准策略给下一轮的指令**：")
        lines.extend(["```json", json.dumps(round_history["generation_directives"], ensure_ascii=False, indent=2), "```", ""])

    if "score_correlations" in round_history:
        lines.append("**目标分数相关性**：")
        lines.extend(["```json", json.dumps(round_history["score_correlations"], ensure_ascii=False, indent=2), "```", ""])

    if "discussion_fallback_used" in round_history:
        lines.append(f"**讨论发生降级/回退**：{bool(round_history['discussion_fallback_used'])}")
    if round_history.get("discussion_decision_source"):
        lines.append(f"**决策来源**：{round_history['discussion_decision_source']}")
    if round_history.get("decision_guardrail", {}).get("applied"):
        lines.append("**控制器护栏调整**：")
        lines.extend(["```json", json.dumps(round_history["decision_guardrail"], ensure_ascii=False, indent=2), "```", ""])
    if round_history.get("discussion_error"):
        lines.append(f"**讨论异常**：{round_history['discussion_error']}")
    if round_history.get("discussion_skip_reason"):
        lines.append(f"**未生成下一轮决定的原因**：{round_history['discussion_skip_reason']}")
    lines.append("")


def _save_chain_markdown(decision: dict, result: dict, output: dict, timestamp: str):
    """保存完整推理链 + 优化过程为可读 Markdown"""
    md_path = os.path.join(OUTPUT_DIR, f"nuwa_agent_chain_{timestamp}.md")

    species_info = decision.get("species_info", {})
    run_metadata = output.get("run_metadata", {})
    input_metadata = run_metadata.get("input", {})
    reproducibility = run_metadata.get("reproducibility", {})
    source_fingerprint = run_metadata.get("software", {}).get("source_fingerprint", {})

    lines = [
        f"# NUWA-Agent 完整推理链 & 优化过程报告",
        f"",
        f"**时间**: {timestamp}",
        f"**随机种子**: {reproducibility.get('run_seed', output.get('run_seed', 'N/A'))}",
        f"**输入蛋白 SHA256**: `{input_metadata.get('protein_sha256', 'N/A')}`",
        f"**代码指纹 SHA256**: `{source_fingerprint.get('source_tree_sha256', 'N/A')}`",
        f"**选定模型**: {decision.get('selected_model', 'N/A')} ({decision.get('model_name', '')})",
        f"**Class ID**: {decision.get('class_id', 'N/A')} (confidence: {decision.get('class_id_confidence', 'N/A')})",
        f"**选模理由**: {decision.get('rationale', '')}",
        f"**总轮数**: {result['total_rounds']}, **收敛**: {result['converged']}, "
        f"**历史最优 HV**: {result.get('best_hv', result['final_hv']):.6f}, "
        f"**末轮 HV**: {result.get('last_round_hv', result['final_hv']):.6f}",
        f"",
        f"---",
        f"",
        f"# Run Metadata — 可复现性信息",
        f"",
        f"- **Audit schema**: {run_metadata.get('audit_schema_version', 'N/A')}",
        f"- **宿主**: {input_metadata.get('host_organism', 'N/A')}",
        f"- **蛋白长度**: {input_metadata.get('protein_length_aa', 'N/A')} aa",
        f"- **LLM 模型**: {reproducibility.get('llm_model', 'N/A')}",
        f"- **严格评估**: {reproducibility.get('strict_evaluation', 'N/A')}",
        f"- **模型文件内容哈希**: {reproducibility.get('model_content_hashes_enabled', 'N/A')}",
        f"",
        f"## 完整输入蛋白序列",
        f"",
        f"```text",
        input_metadata.get("protein_sequence", "N/A"),
        f"```",
        f"",
        f"## 模型制品标识",
        f"",
        f"| 用途 | Checkpoint | Manifest SHA256 | 路径 |",
        f"|---|---|---|---|",
        *[
            f"| {name} | {artifact.get('checkpoint', 'N/A')} | "
            f"`{artifact.get('manifest_sha256') or 'N/A'}` | `{artifact.get('path', 'N/A')}` |"
            for name, artifact in run_metadata.get("model_artifacts", {}).items()
        ],
        f"",
        f"---",
        f"",
        f"# Phase 0: SpeciesResolver — 物种解析",
        f"",
    ]

    # SpeciesResolver 结果
    if species_info:
        lines.append(f"- **最终选定物种/代理**: {species_info.get('matched_name', 'N/A')}")
        lines.append(f"- **最终 class_id**: {species_info.get('class_id', 'N/A')}")
        lines.append(f"- **域**: {species_info.get('domain', 'N/A')}")
        lines.append(f"- **置信度**: {species_info.get('confidence', 'N/A')}")
        lines.append(f"- **是否解析成功**: {species_info.get('is_resolved', False)}")
        lines.append(f"- **是否为代理解析**: {species_info.get('proxy_resolved', False)}")
        lines.append(f"- **解析类型**: {species_info.get('resolution_type', 'N/A')}")
        lines.append(f"- **理由**: {species_info.get('reason', '')}")
        resolver_match = species_info.get("resolver_match", {})
        if resolver_match:
            lines.append(
                f"- **Resolver 初始建议（未必为最终选择）**: "
                f"{resolver_match.get('matched_name', 'N/A')} / "
                f"class_id={resolver_match.get('suggested_class_id', 'N/A')}"
            )
        if species_info.get("validation_note"):
            lines.append(f"- **选择校验说明**: {species_info['validation_note']}")

        # 候选项列表
        candidates = species_info.get('candidates', [])
        if candidates:
            lines.append("")
            lines.append("### 候选项列表")
            lines.append("")
            lines.append("| # | 物种名 | 域 | class_id |")
            lines.append("|---|--------|-----|----------|")
            for i, c in enumerate(candidates):
                lines.append(f"| {i+1} | {c.get('name', c) if isinstance(c, dict) else c} "
                            f"| {c.get('domain', '?') if isinstance(c, dict) else '?'} "
                            f"| {c.get('class_id', '?') if isinstance(c, dict) else '?'} |")
        lines.append("")

    lines.append(f"---")
    lines.append("")
    lines.append(f"# Phase 1: CoT 三层决策推理链")
    lines.append("")

    # 三层决策详情
    level1 = decision.get("level1_domain", {})
    if level1:
        lines.append(f"## Level 1: 域级决策")
        lines.append(f"- **域**: {level1.get('domain', 'N/A')} (置信度: {level1.get('confidence', 'N/A')})")
        lines.append(f"- **理由**: {level1.get('reason', 'N/A')}")
        lines.append("")

    level2 = decision.get("level2_species", {})
    if level2:
        lines.append(f"## Level 2: 物种级决策")
        lines.append(f"- **class_id**: {level2.get('class_id', 'N/A')} (置信度: {level2.get('class_id_confidence', 'N/A')})")
        lines.append(f"- **理由**: {level2.get('reason', 'N/A')}")
        lines.append("")

    # LLM 选定的近亲
    chosen = decision.get("chosen_candidate")
    if chosen:
        lines.append(f"### LLM 选定的近亲物种")
        lines.append(f"- **物种**: {chosen.get('name', 'N/A')}")
        lines.append(f"- **class_id**: {chosen.get('class_id', 'N/A')}")
        lines.append(f"- **选择理由**: {chosen.get('why', 'N/A')}")
        lines.append("")

    level3 = decision.get("level3_cross_domain", {})
    if level3:
        lines.append(f"## Level 3: 跨域决策")
        lines.append(f"- **考虑跨域**: {level3.get('consider_cross_domain', False)}")
        lines.append(f"- **备选模型**: {level3.get('alternative_model', 'N/A')}")
        lines.append(f"- **理由**: {level3.get('reason', 'N/A')}")
        lines.append("")

    # 模型讨论
    model_discussion = decision.get("model_discussion", {})
    if model_discussion:
        lines.append(f"## 模型适用性讨论")
        lines.append("")
        lines.append("| 模型 | 适用性 | 理由 | 风险 |")
        lines.append("|------|--------|------|------|")
        for mk in ["bacteria", "eukaryote", "archaea"]:
            if mk in model_discussion:
                md = model_discussion[mk]
                lines.append(f"| {mk} | {md.get('suitability', '?')} "
                            f"| {md.get('reason', '?')} "
                            f"| {md.get('risk', '?')} |")
        lines.append("")

    # host_analysis
    host_analysis = decision.get("host_analysis", {})
    if host_analysis:
        lines.append(f"## 宿主分析")
        lines.append(f"- **域**: {host_analysis.get('domain', 'N/A')}")
        lines.append(f"- **密码子偏好**: {host_analysis.get('codon_bias', 'N/A')}")
        lines.append(f"- **GC倾向**: {host_analysis.get('gc_tendency', 'N/A')}")
        lines.append(f"- **特殊特征**: {host_analysis.get('special_features', 'N/A')}")
        lines.append("")

    # 约束边界
    lines.append(f"## 约束边界")
    constraint_eval = output.get("constraint_evaluation", {})
    validation = output.get("artifact_validation", {})
    lines.append(f"- **输出制品自检通过**: {validation.get('passed', 'N/A')}")
    if validation.get("errors"):
        lines.append("- **自检错误**: " + "; ".join(validation["errors"]))
    if constraint_eval:
        lines.append(f"- **MFE 评估模式**: {output.get('constraint_modes', {}).get('mfe', 'N/A')}")
        lines.append(f"- **说明**: {constraint_eval.get('note', '')}")
        if constraint_eval.get("inactive_legacy_bounds"):
            lines.append("- **未启用的兼容边界**: " +
                         json.dumps(constraint_eval["inactive_legacy_bounds"], ensure_ascii=False))
    lines.append(f"```json")
    lines.append(json.dumps(constraint_eval.get("active_bounds",
                                                decision.get("constraint_bounds", {})),
                            ensure_ascii=False, indent=2))
    lines.append(f"```")
    lines.append("")

    # 推理链
    lines.append(f"## 完整推理链")
    lines.append("")
    chain = decision.get("reasoning_chain", "")
    if chain:
        lines.append(f"```")
        lines.append(chain)
        lines.append(f"```")
    lines.append("")

    # ---- LLM 完整调用日志 ----
    full_log = decision.get("full_log", [])
    if full_log:
        lines.append(f"---")
        lines.append("")
        lines.append(f"# LLM 完整调用日志")
        lines.append("")
        for log_entry in full_log:
            lines.append(f"## {log_entry.get('step_name', 'Unknown')}")
            lines.append("")
            lines.append(f"<details open><summary>📤 发送给 LLM 的 Prompt</summary>")
            lines.append("")
            lines.append(f"```")
            lines.append(log_entry.get('prompt', ''))
            lines.append(f"```")
            lines.append("")
            lines.append(f"</details>")
            lines.append("")
            lines.append(f"<details open><summary>📥 LLM 原始返回</summary>")
            lines.append("")
            lines.append(f"```")
            lines.append(log_entry.get('raw_response', ''))
            lines.append(f"```")
            lines.append("")
            lines.append(f"</details>")
            lines.append("")

    # ====================================================================
    # Phase 2: 完整迭代优化过程
    # ====================================================================
    lines.append(f"---")
    lines.append("")
    lines.append(f"# Phase 2: 完整迭代优化过程")
    lines.append("")

    history = result.get("history", [])
    for h in history:
        round_num = h['round']
        lines.append(f"## Round {round_num} (实际温度: {h.get('temperature', 1.0):.4f})")
        lines.append("")
        lines.append(f"- **候选总数**: {h['num_candidates']}")
        lines.append(f"- **可行解**: {h['feasible_count']}/{h['num_candidates']}")
        lines.append(f"- **Pareto 前沿大小**: {h['pareto_front_size']}")
        lines.append(f"- **HV**: {h['hv']:.6f} (Δ{h['hv_improvement']:+.6f})")
        lines.append("")

        _append_discussion_audit(lines, h)

        # 生成策略
        gen_meta = h.get("generation_meta", {})
        if gen_meta:
            n_new = gen_meta.get("new_count", 0)
            n_mut = gen_meta.get("mutated_count", 0)
            elite_parents = gen_meta.get("elite_parents", [])
            lines.append(f"### 生成策略")
            lines.append(f"- 新生成: **{n_new}** 条")
            lines.append(f"- 精英突变: **{n_mut}** 条")
            if elite_parents:
                lines.append(f"- 精英父本 ({len(elite_parents)} 条):")
                for ep_idx, ep in enumerate(elite_parents):
                    lines.append(f"  - Elite #{ep_idx}: `{ep[:60]}{'...' if len(ep) > 60 else ''}` (长度: {len(ep.replace(' ', ''))})")
            lines.append("")

        # 突变详情
        mutation_details = gen_meta.get("mutation_details", [])
        if mutation_details:
            lines.append(f"### 突变详情 ({len(mutation_details)} 条)")
            lines.append("")
            lines.append("| # | 父本 # | 突变数 | 突变列表 |")
            lines.append("|---|--------|--------|----------|")
            for mi, md in enumerate(mutation_details):
                mut_list = "; ".join(
                    f"[{m['codon_index']}] {m['old_codon']}→{m['new_codon']}({m['amino_acid']})"
                    for m in md.get("mutations", [])
                )
                lines.append(f"| {mi+1} | Elite #{md.get('elite_index', '?')} "
                            f"| {md.get('num_mutations', 0)} "
                            f"| {mut_list} |")
            lines.append("")

        # Pareto 前沿完整打分
        pareto_front = h.get("pareto_front_details", [])
        if pareto_front:
            lines.append(f"### Pareto 前沿完整打分 ({len(pareto_front)} 条)")
            lines.append("")
            lines.append("| Rank | TE | Stability | Expression | CAI | GC% | MFE | Stem | Homo | Feasible |")
            lines.append("|------|-----|-----------|------------|-----|-----|-----|------|------|----------|")
            for pf in pareto_front:
                o = pf["objectives"]
                c = pf["constraints"]
                lines.append(f"| {pf['rank']} | {o['TE']:.4f} | {o['Stability']:.4f} | {o['Expression']:.4f} "
                            f"| {c['CAI']:.4f} | {c['GC%']:.1%} | {c['MFE']:.1f} "
                            f"| {c['max_stem_len']} | {c['max_homopolymer']} | {pf['is_feasible']} |")
            lines.append("")

            # 每条 Pareto 序列
            lines.append(f"### Pareto 前沿序列")
            lines.append("")
            for pf in pareto_front:
                seq = pf["sequence"]
                o = pf["objectives"]
                lines.append(f"- **Rank {pf['rank']}** (CD={pf['crowding_distance']:.4f}): "
                            f"TE={o['TE']:.4f} Stab={o['Stability']:.4f} Expr={o['Expression']:.4f}")
                lines.append(f"  ```\n  {seq}\n  ```")
            lines.append("")

        # 精英序列
        elite_seqs = h.get("elite_sequences", [])
        if elite_seqs:
            lines.append(f"### 传递到下一轮的精英序列")
            lines.append("")
            for ei, es in enumerate(elite_seqs):
                lines.append(f"- Elite #{ei}: `{es[:60]}{'...' if len(es) > 60 else ''}`")
            lines.append("")

        # 全部候选打分表
        candidate_scores = h.get("candidate_scores", [])
        if candidate_scores:
            lines.append(f"### 全部候选打分表 ({len(candidate_scores)} 条)")
            lines.append("")
            lines.append("<details><summary>展开查看详细打分</summary>")
            lines.append("")
            lines.append("| # | Source | TE | Stability | Expression | CAI | GC% | MFE | Feasible |")
            lines.append("|---|--------|-----|-----------|------------|-----|-----|-----|----------|")
            for cs in candidate_scores:
                lines.append(f"| {cs['index']} | {cs['source']} "
                            f"| {cs['te_score']:.4f} | {cs['stability_score']:.4f} "
                            f"| {cs['expression_score']:.4f} | {cs['cai']:.4f} "
                            f"| {cs['gc_content']:.1%} | {cs['mfe']:.1f} "
                            f"| {cs['is_feasible']} |")
            lines.append("")
            lines.append("</details>")
            lines.append("")

        # 完整反馈
        feedback = h.get("feedback", "")
        feedback_prompt = h.get("feedback_prompt", "")
        if feedback_prompt:
            lines.append(f"### 发送给反馈 LLM 的 Prompt")
            lines.append("")
            lines.append(f"<details><summary>📤 prompt</summary>")
            lines.append("")
            lines.append(f"```")
            lines.append(feedback_prompt)
            lines.append(f"```")
            lines.append("")
            lines.append(f"</details>")
            lines.append("")
        if feedback:
            lines.append(f"### 下一轮执行摘要（控制器权威值）")
            lines.append("")
            lines.append(f"```")
            lines.append(feedback)
            lines.append(f"```")
            lines.append("")

        # 弱点 & 建议
        weaknesses = h.get("weaknesses", [])
        suggestions = h.get("suggestions", [])
        if weaknesses:
            lines.append(f"### 识别到的弱点")
            lines.append("")
            for w in weaknesses:
                lines.append(f"- {w}")
            lines.append("")
        if suggestions:
            lines.append(f"### 优化建议")
            lines.append("")
            for s in suggestions:
                lines.append(f"- {s}")
            lines.append("")

        # 总体评估
        overall = h.get("overall_assessment", "")
        if overall:
            lines.append(f"### 总体评估")
            lines.append("")
            lines.append(f"> {overall}")
            lines.append("")

        # 反馈 LLM 原始输出
        feedback_raw = h.get("feedback_raw", "")
        if feedback_raw:
            lines.append(f"<details><summary>反馈 LLM 原始输出</summary>")
            lines.append("")
            lines.append(f"```json")
            lines.append(feedback_raw)
            lines.append(f"```")
            lines.append("")
            lines.append(f"</details>")
            lines.append("")

    # ====================================================================
    # Phase 3: Pareto 最优解
    # ====================================================================
    lines.append(f"---")
    lines.append("")
    lines.append(f"# Phase 3: 最终 Pareto 最优解")
    lines.append("")

    for s in output.get("pareto_solutions", []):
        o = s["objectives"]
        c = s["constraints"]
        lines.append(f"## Rank {s['rank']} (CD={s['crowding_distance']:.4f})")
        lines.append(f"- **TE**: {o['TE']:.6f}")
        lines.append(f"- **Stability**: {o['Stability']:.6f}")
        lines.append(f"- **Expression**: {o['Expression']:.6f}")
        lines.append(f"- **CAI**: {c['CAI']:.6f}")
        lines.append(f"- **GC%**: {c['GC%']:.1%}")
        lines.append(f"- **MFE**: {c['MFE']:.3f}")
        lines.append(f"- **max_stem_len**: {c['max_stem_len']}")
        lines.append(f"- **max_homopolymer**: {c['max_homopolymer']}")
        lines.append(f"- **Feasible**: {s['is_feasible']}")
        lines.append("")
        lines.append(f"### 完整序列")
        lines.append(f"```")
        lines.append(s.get('sequence_codon_spaced', s['sequence']))
        lines.append(f"```")
        lines.append("")

    # ====================================================================
    # 好序列队列 (每轮收集的可行 Pareto 前沿解)
    # ====================================================================
    good_queue = output.get("good_sequences_queue", [])
    if good_queue:
        lines.append(f"---")
        lines.append("")
        lines.append(f"# 好序列队列 (每轮收集的可行 Pareto 前沿解, 共 {len(good_queue)} 条)")
        lines.append("")
        lines.append("| # | 轮次 | TE | Stability | Expression | CAI | GC% | MFE | Stem | Homo |")
        lines.append("|---|------|-----|-----------|------------|-----|-----|-----|------|------|")
        for i, gq in enumerate(good_queue):
            lines.append(f"| {i+1} | R{gq['round']} | {gq['te_score']:.4f} | {gq['stability_score']:.4f} "
                        f"| {gq['expression_score']:.4f} | {gq['cai']:.4f} | {gq['gc_content']:.1%} "
                        f"| {gq['mfe']:.1f} | {gq['max_stem_len']} | {gq['max_homopolymer']} |")
        lines.append("")

        # 按轮次分组列出序列
        lines.append(f"## 好序列详情 (按轮次)")
        lines.append("")
        from itertools import groupby
        for rnd, group in groupby(good_queue, key=lambda x: x["round"]):
            group_list = list(group)
            lines.append(f"### Round {rnd} ({len(group_list)} 条)")
            lines.append("")
            for i, gq in enumerate(group_list):
                lines.append(f"- **R{rnd} #{i+1}** (CD={gq['crowding_distance']:.4f}): "
                            f"TE={gq['te_score']:.4f} Stab={gq['stability_score']:.4f} "
                            f"Expr={gq['expression_score']:.4f} CAI={gq['cai']:.4f}")
                lines.append(f"  ```")
                lines.append(gq.get('sequence_codon_spaced', gq['sequence']))
                lines.append(f"  ```")
            lines.append("")

    # ====================================================================
    # 优化过程摘要
    # ====================================================================
    lines.append(f"---")
    lines.append("")
    lines.append(f"# 优化过程摘要")
    lines.append("")
    lines.append(f"## HV 变化曲线")
    lines.append("")
    lines.append("| Round | HV | ΔHV | Feasible | Pareto Front |")
    lines.append("|-------|-----|-----|----------|--------------|")
    for h in history:
        lines.append(f"| {h['round']} | {h['hv']:.6f} | {h['hv_improvement']:+.6f} "
                    f"| {h['feasible_count']}/{h['num_candidates']} "
                    f"| {h['pareto_front_size']} |")
    lines.append("")

    # 收敛信息
    lines.append(f"## 收敛信息")
    lines.append(f"- **总轮数**: {result['total_rounds']}")
    lines.append(f"- **收敛**: {result['converged']}")
    lines.append(f"- **收敛阈值**: {HV_CONVERGENCE_THRESHOLD}")
    lines.append(f"- **历史最优 HV（返回解）**: {result.get('best_hv', result['final_hv']):.6f}")
    lines.append(f"- **末轮 HV**: {result.get('last_round_hv', result['final_hv']):.6f}")
    lines.append("")

    _atomic_write_text(md_path, _redact_api_key("\n".join(lines)))


def _save_reviewer_summary(output: dict, timestamp: str) -> str:
    """Write a compact, reviewer-facing report; keep raw prompts in the chain appendix."""
    path = os.path.join(OUTPUT_DIR, f"nuwa_agent_reviewer_{timestamp}.md")
    metadata = output.get("run_metadata", {})
    inputs = metadata.get("input", {})
    reproducibility = metadata.get("reproducibility", {})
    software = metadata.get("software", {})
    source = software.get("source_fingerprint", {})
    optimization = output.get("optimization_result", {})
    selection = output.get("model_selection", {})
    species = selection.get("species_info", {})
    constraint_eval = output.get("constraint_evaluation", {})
    validation = output.get("artifact_validation", {})

    lines = [
        "# NUWA-Agent Reviewer Summary",
        "",
        "> This is the authoritative concise report. Raw LLM proposals are preserved in the "
        f"> companion `nuwa_agent_chain_{timestamp}.md` audit appendix and are not executable decisions.",
        "",
        "## Run identity",
        "",
        f"- Run ID: `{metadata.get('run_id', timestamp)}`",
        f"- Created: `{metadata.get('created_at_local', 'N/A')}`",
        f"- Audit schema: `{metadata.get('audit_schema_version', 'N/A')}`",
        f"- Host: `{inputs.get('host_organism', 'N/A')}`",
        f"- Protein length: `{inputs.get('protein_length_aa', 'N/A')} aa`",
        f"- Protein SHA256: `{inputs.get('protein_sha256', 'N/A')}`",
        f"- Run seed: `{reproducibility.get('run_seed', output.get('run_seed', 'N/A'))}`",
        f"- Source-tree SHA256: `{source.get('source_tree_sha256', 'N/A')}`",
        f"- LLM model: `{reproducibility.get('llm_model', 'N/A')}`",
        f"- Strict evaluation: `{reproducibility.get('strict_evaluation', 'N/A')}`",
        f"- Artifact self-validation: `{validation.get('passed', 'N/A')}`",
        f"- Validation checks: `{', '.join(validation.get('checks', [])) or 'N/A'}`",
        "",
        "## Exact input protein",
        "",
        "```text",
        inputs.get("protein_sequence", "N/A"),
        "```",
        "",
        "## Model and species selection",
        "",
        f"- NUWA model: `{selection.get('model_name', selection.get('selected_model', 'N/A'))}`",
        f"- Final species/proxy: `{species.get('matched_name', 'N/A')}`",
        f"- Final class ID: `{selection.get('class_id', 'N/A')}`",
        f"- Resolution type: `{species.get('resolution_type', 'N/A')}`",
        f"- Resolver initial suggestion: `"
        f"{species.get('resolver_match', {}).get('matched_name', 'N/A')}` / class_id="
        f"`{species.get('resolver_match', {}).get('suggested_class_id', 'N/A')}`",
        "",
        "## Effective constraints",
        "",
        f"- MFE mode: `{output.get('constraint_modes', {}).get('mfe', 'N/A')}`",
        f"- Active bounds: `{json.dumps(constraint_eval.get('active_bounds', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- Inactive legacy bounds: `{json.dumps(constraint_eval.get('inactive_legacy_bounds', {}), ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Optimization result",
        "",
        f"- Total rounds: `{optimization.get('total_rounds', 'N/A')}`",
        f"- Converged: `{optimization.get('converged', 'N/A')}`",
        f"- Best round: `{optimization.get('best_round', 'N/A')}`",
        f"- Best HV (returned set): `{optimization.get('best_hv', 'N/A')}`",
        f"- Last-round HV: `{optimization.get('last_round_hv', 'N/A')}`",
        "",
        "| Round | HV | Delta HV | Feasible | Pareto front | Applied decision source |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for round_record in output.get("iteration_history", []):
        lines.append(
            f"| {round_record.get('round')} | {round_record.get('hv', 0):.6f} | "
            f"{round_record.get('hv_improvement', 0):+.6f} | "
            f"{round_record.get('feasible_count')}/{round_record.get('num_candidates')} | "
            f"{round_record.get('pareto_front_size')} | "
            f"{round_record.get('applied_decision_source', 'N/A')} |"
        )

    lines.extend([
        "",
        "## Model artifact manifests",
        "",
        "| Purpose | Checkpoint | Content-addressed manifest SHA256 | Exists |",
        "|---|---|---|---|",
    ])
    for name, artifact in metadata.get("model_artifacts", {}).items():
        lines.append(
            f"| {name} | {artifact.get('checkpoint', 'N/A')} | "
            f"`{artifact.get('manifest_sha256') or 'N/A'}` | {artifact.get('exists', False)} |"
        )

    lines.extend([
        "",
        "## Returned Pareto solutions",
        "",
        "Canonical `sequence` values are uppercase, unspaced RNA. The codon-spaced form is display-only.",
        "",
        "| ID | Rank | TE | Stability | Expression | CAI | GC | MFE/nt | Stem | Homopolymer | nt | Feasible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for index, solution in enumerate(output.get("pareto_solutions", []), start=1):
        objective = solution.get("objectives", {})
        constraints = solution.get("constraints", {})
        lines.append(
            f"| S{index:03d} | {solution.get('rank')} | {objective.get('TE', 0):.6f} | "
            f"{objective.get('Stability', 0):.6f} | {objective.get('Expression', 0):.6f} | "
            f"{constraints.get('CAI', 0):.6f} | {constraints.get('GC%', 0):.6f} | "
            f"{constraints.get('MFE_per_nt', 0):.6f} | {constraints.get('max_stem_len')} | "
            f"{constraints.get('max_homopolymer')} | {solution.get('sequence_length')} | "
            f"{solution.get('is_feasible')} |"
        )

    lines.extend(["", "## Canonical sequences", ""])
    for index, solution in enumerate(output.get("pareto_solutions", []), start=1):
        lines.extend([
            f"### S{index:03d}",
            "",
            "```text",
            solution.get("sequence", ""),
            "```",
            "",
        ])

    lines.extend([
        "## Reproducibility note",
        "",
        reproducibility.get("note", ""),
        "",
        f"Machine-readable record: `nuwa_agent_{timestamp}.json`.",
    ])
    _atomic_write_text(path, _redact_api_key("\n".join(lines)))
    return path


if __name__ == "__main__":
    main()
