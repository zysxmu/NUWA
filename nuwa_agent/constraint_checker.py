"""Constraint Checker: 验证 6 维约束满足性"""

from evaluator import EvaluationResult


class ConstraintChecker:
    """约束检查器 — 6 维约束可行性判定"""

    def __init__(self, bounds: dict):
        self.bounds = bounds

    def check(self, result: EvaluationResult) -> EvaluationResult:
        """检查约束, 设置 is_feasible 和 violations"""
        violations = []

        # g1: CAI
        if result.cai < self.bounds.get("cai_min", 0.0):
            violations.append(f"CAI={result.cai:.3f} < {self.bounds['cai_min']}")

        # g2: GC%
        gc_min = self.bounds.get("gc_min", 0.0)
        gc_max = self.bounds.get("gc_max", 1.0)
        if result.gc_content < gc_min or result.gc_content > gc_max:
            violations.append(f"GC={result.gc_content:.1%} not in [{gc_min:.0%}, {gc_max:.0%}]")

        # g3: MFE — 双侧约束 [mfe_min, mfe_max]
        # mfe_max: 结构必须达到一定稳定性 (MFE ≤ mfe_max, 如 -100)
        # mfe_min: 防止过度折叠 (MFE ≥ mfe_min, 如 -400)
        #   Stability 微调模型与 MFE 强负相关 (r≈-0.79)，过负 MFE 会被模型惩罚
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

        # g6: immunogenicity (一票否决 — 高风险序列即使其他全部通过也不可行)
        safety = self.bounds.get("safety_threshold", 1.0)
        if result.immunogenicity_risk >= safety:
            violations.append(f"Immunogenicity={result.immunogenicity_risk:.3f} >= {safety}")

        result.is_feasible = len(violations) == 0
        result.constraint_violations = violations
        return result

    def check_batch(self, results: list) -> list:
        """批量检查"""
        return [self.check(r) for r in results]
