"""Constraint Checker: 验证候选序列的约束满足性。"""

from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluator import EvaluationResult


class ConstraintChecker:
    """约束检查器 — 5 项序列级约束的可行性判定"""

    def __init__(self, bounds: dict):
        self.bounds = dict(bounds)
        has_mfe_nt_min = "mfe_per_nt_min" in self.bounds
        has_mfe_nt_max = "mfe_per_nt_max" in self.bounds
        if has_mfe_nt_min != has_mfe_nt_max:
            raise ValueError("mfe_per_nt_min and mfe_per_nt_max must be supplied together")
        self.use_mfe_per_nt = has_mfe_nt_min
        if self.use_mfe_per_nt:
            lower = self.bounds["mfe_per_nt_min"]
            upper = self.bounds["mfe_per_nt_max"]
            if not all(isinstance(value, Real) and not isinstance(value, bool)
                       and math.isfinite(value) for value in (lower, upper)):
                raise ValueError("MFE/nt bounds must be finite numbers")
            if lower > upper:
                raise ValueError("mfe_per_nt_min must be <= mfe_per_nt_max")


    def check(self, result: EvaluationResult) -> EvaluationResult:
        """检查约束, 设置 is_feasible 和 violations"""
        violations = []

        # g1: CAI
        cai_min = self.bounds.get("cai_min", 0.0)
        if result.cai <= cai_min:
            violations.append(f"CAI={result.cai:.3f} <= {cai_min}")

        # g2: GC%
        gc_min = self.bounds.get("gc_min", 0.0)
        gc_max = self.bounds.get("gc_max", 1.0)
        if result.gc_content < gc_min or result.gc_content > gc_max:
            violations.append(f"GC={result.gc_content:.1%} not in [{gc_min:.0%}, {gc_max:.0%}]")

        # g3: Use target-specific MFE/nt bounds when supplied. Older runs only
        # specify absolute mfe_min/mfe_max; preserve their original behavior.
        if self.use_mfe_per_nt:
            nt_length = len("".join(result.sequence.split()))
            if nt_length == 0:
                raise ValueError("cannot evaluate MFE/nt for an empty sequence")
            mfe_per_nt = result.mfe / nt_length
            mfe_min = self.bounds["mfe_per_nt_min"]
            mfe_max = self.bounds["mfe_per_nt_max"]
            if mfe_per_nt > mfe_max:
                violations.append(f"MFE/nt={mfe_per_nt:.4f} > {mfe_max}")
            if mfe_per_nt < mfe_min:
                violations.append(f"MFE/nt={mfe_per_nt:.4f} < {mfe_min} (over-folded)")
        else:
            mfe_max = self.bounds.get("mfe_max", 0)
            mfe_min = self.bounds.get("mfe_min", -99999)
            if result.mfe > mfe_max:
                violations.append(f"MFE={result.mfe:.1f} > {mfe_max}")
            if result.mfe < mfe_min:
                violations.append(f"MFE={result.mfe:.1f} < {mfe_min} (over-folded)")

        # g4: max stem
        max_stem = self.bounds.get("max_stem_length", 999)
        if result.max_stem_len >= max_stem:
            violations.append(f"Stem={result.max_stem_len}bp >= {max_stem}bp")

        # g5: max homopolymer
        max_homo = self.bounds.get("max_homopolymer", 999)
        if result.max_homopolymer >= max_homo:
            violations.append(f"Homopolymer={result.max_homopolymer}nt >= {max_homo}nt")

        result.is_feasible = len(violations) == 0
        result.constraint_violations = violations
        return result

    def check_batch(self, results: list) -> list:
        """批量检查"""
        return [self.check(r) for r in results]
