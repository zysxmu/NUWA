"""Iteration Controller: 管理迭代优化循环 + 收敛判定

流程: 生成 → 评估 → 约束检查 → Pareto 选择 → LLM 反馈 → 收敛判定
温度逐轮递减: 1.0 → 0.3 (在前40%轮次内线性衰减, 之后保持0.3)

迭代优化策略:
  - 第 1 轮: NUWA 从头生成
  - 第 2 轮起: NUWA 生成 + 对上轮 Pareto 前沿序列做同义密码子突变
  - 突变比例: 50% 新生成 + 50% 突变体 (可配置)

每轮好的序列 (可行 Pareto 前沿解) 会被收集到 _good_sequences_queue,
去重后随最终结果一起输出。
"""

from evaluator import MultiObjectiveEvaluator, EvaluationResult
from constraint_checker import ConstraintChecker
from pareto_selector import ParetoSelector
from feedback_analyzer import FeedbackAnalyzer
from model_registry import registry
from config import MAX_ITERATION_ROUNDS, MIN_ITERATION_ROUNDS, HV_CONVERGENCE_THRESHOLD, NUM_CANDIDATES, PARETO_TOP_K

import random
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

# 同义密码子替换所需常量 (来自 model_registry)
from model_registry import CODON_TO_AA, AA_TO_CODONS


class IterationController:
    """迭代优化控制器

    优化策略:
      - 第 1 轮: NUWA 从头生成 NUM_CANDIDATES 条序列
      - 第 2 轮起: NUWA 生成 50% + 上轮 Pareto 前沿同义突变 50%
      - 突变方式: 随机选择 1~3 个密码子做同义替换
    """

    MUTATE_RATIO = 0.5       # 突变体占每轮候选的比例
    MAX_MUTATIONS_PER_SEQ = 3  # 每条序列最多突变几个密码子
    CAI_OPTIMIZE_RATIO = 0.3   # 新生成候选中应用 CAI 优化的比例
    CAI_OPTIMIZE_INTENSITY = 0.25  # CAI 优化强度: 替换最低适应性密码子的比例 (0.25 = 25%)

    def __init__(self, selected_model: str, constraint_bounds: dict,
                 evaluator: MultiObjectiveEvaluator,
                 class_id: int = 0,
                 codon_table: dict = None):
        self.selected_model = selected_model
        self.constraint_bounds = constraint_bounds
        self.evaluator = evaluator
        self.class_id = class_id
        self.codon_table = codon_table    # 宿主参考密码子表 (用于 CAI 优化)
        self.checker = ConstraintChecker(constraint_bounds)
        self.selector = ParetoSelector()
        self.analyzer = FeedbackAnalyzer()

        self.history = []
        self.prev_hv = 0.0
        self._elite_sequences: list = []  # 上轮 Pareto 前沿序列 (RNA 字符串)
        self._elite_solutions: list = []  # 上轮 Pareto 前沿解 (含分数, 供 focus 偏置)
        self._last_directives: dict = None  # 上一轮 feedback 的 generation_directives
        self._best_hv: float = 0.0        # 历史最高 HV
        self._best_solutions: list = []   # HV 最高轮的 Pareto 解
        self._best_round: int = 0         # HV 最高轮的轮次
        self._good_sequences_queue: list = []  # 每轮收集的好的序列 (可行 Pareto 前沿)
        self._good_seq_seen: set = set()       # 已入队序列的去重集合

    def run(self, protein_seq: str, host_organism: str) -> dict:
        """
        执行迭代优化循环

        Returns:
            {
                "best_solutions": list,  # List[ParetoSolution]
                "history": list,         # List[dict]
                "total_rounds": int,
                "final_hv": float,
                "converged": bool,
            }
        """
        feedback = None
        self._elite_sequences = []   # 每轮更新
        self._elite_solutions = []
        self._last_directives = None

        for round_num in range(1, MAX_ITERATION_ROUNDS + 1):
            print(f"\n[Round {round_num}/{MAX_ITERATION_ROUNDS}]")

            # 温度在前40%轮次内从1.0线性衰减到0.3, 之后保持0.3
            decay_rounds = max(1, int(MAX_ITERATION_ROUNDS * 0.4))
            base_temperature = max(0.3, 1.0 - 0.7 * (round_num - 1) / decay_rounds)

            # 读取上一轮 feedback 的 generation_directives (机器指令, 默认保留原行为)
            directives = self._sanitize_directives(self._last_directives)

            # explore 调节探索度: 默认 1.0 时温度不变, 越小越利用
            explore = directives.get("explore", 1.0)
            temperature = max(0.3, base_temperature * (0.4 + 0.6 * explore))

            # Step A: 生成候选 (新生成 + 精英突变), 指令随 feedback 动态调整
            candidates, gen_meta = self._generate_candidates(
                protein_seq, round_num, temperature, feedback, directives=directives
            )

            # Step B: 评估 (外部生物信息学工具)
            print("  Evaluating with external bioinformatics tools...")
            results = self.evaluator.evaluate_batch(candidates, protein_seq)

            # Step C: 约束 + Pareto
            results = self.checker.check_batch(results)
            feasible_count = sum(1 for r in results if r.is_feasible)
            print(f"  Feasible: {feasible_count}/{len(results)}")

            pareto_solutions = self.selector.select(results, top_k=PARETO_TOP_K)

            # 更新精英序列 (供下一轮突变用)
            self._elite_sequences = [
                s.result.sequence for s in pareto_solutions if s.rank == 0
            ]
            self._elite_solutions = [s for s in pareto_solutions if s.rank == 0]

            hv = self.selector.compute_hypervolume(pareto_solutions)
            hv_improvement = hv - self.prev_hv
            front_size = len([s for s in pareto_solutions if s.rank == 0])

            # 追踪历史最优 HV 和对应解 (修复 P0: 收敛时返回最优轮而非当前轮)
            if hv > self._best_hv:
                self._best_hv = hv
                self._best_solutions = list(pareto_solutions)  # 深拷贝引用
                self._best_round = round_num
                print(f"  📈 新最优 HV: {hv:.4f} (Round {round_num})")

            # ---- 收集本轮好的序列到输出队列 ----
            # 收集所有 rank=0 且 feasible 的 Pareto 前沿解, 去重后入队
            round_good = 0
            for s in pareto_solutions:
                if s.rank == 0 and s.result.is_feasible:
                    seq_key = s.result.sequence.replace(" ", "").upper()
                    if seq_key not in self._good_seq_seen:
                        self._good_seq_seen.add(seq_key)
                        r = s.result
                        self._good_sequences_queue.append({
                            "round": round_num,
                            "sequence": r.sequence,
                            "te_score": round(r.te_score, 6),
                            "stability_score": round(r.stability_score, 6),
                            "expression_score": round(r.expression_score, 6),
                            "cai": round(r.cai, 6),
                            "gc_content": round(r.gc_content, 6),
                            "mfe": round(r.mfe, 3),
                            "max_stem_len": r.max_stem_len,
                            "max_homopolymer": r.max_homopolymer,
                            "immunogenicity_risk": round(r.immunogenicity_risk, 6),
                            "crowding_distance": round(s.crowding_distance, 6),
                            "is_feasible": True,
                        })
                        round_good += 1
            if round_good > 0:
                print(f"  📦 本轮新增 {round_good} 条好序列到输出队列 (累计 {len(self._good_sequences_queue)} 条)")

            # 打印 Pareto 前沿打分明细
            front_solutions = [s for s in pareto_solutions if s.rank == 0]
            if front_solutions:
                print(f"  Pareto front ({front_size} solutions):")
                for s in front_solutions:
                    r = s.result
                    print(f"    Rank {s.rank}: TE={r.te_score:.3f} Stab={r.stability_score:.3f} "
                          f"Expr={r.expression_score:.3f} | CAI={r.cai:.3f} GC={r.gc_content:.1%} "
                          f"MFE={r.mfe:.1f} Stem={r.max_stem_len} Homo={r.max_homopolymer} "
                          f"Immuno={r.immunogenicity_risk:.3f} | Feasible={r.is_feasible}")

            # 打印次优前沿 (rank=1) 摘要
            rank1 = [s for s in pareto_solutions if s.rank == 1]
            if rank1:
                best_r1 = max(rank1, key=lambda s: s.result.expression_score)
                r = best_r1.result
                print(f"  Rank-1 best: TE={r.te_score:.3f} Stab={r.stability_score:.3f} "
                      f"Expr={r.expression_score:.3f}")

            print(f"  HV: {hv:.4f} (Δ{hv_improvement:+.4f})")
            print(f"  Pareto front: {front_size} solutions (top-{PARETO_TOP_K} shown above)")

            # Step D: LLM 反馈 (传入完整评估结果用于相关性分析)
            if round_num < MAX_ITERATION_ROUNDS:
                analysis = self.analyzer.analyze(
                    pareto_solutions, round_num, host_organism, self.constraint_bounds,
                    all_results=results,   # ← 全部候选评估结果, 用于计算相关性
                )
                feedback = analysis["feedback"]
                self._last_directives = analysis.get("generation_directives", {}) or {}
                print(f"  Feedback: {feedback[:80]}...")
                print(f"  Directives: {self._last_directives}")
                # 打印本轮相关性摘要
                score_corr = analysis.get("score_correlations", {})
                if score_corr:
                    for obj_name, corrs in score_corr.items():
                        sig = [(m, v) for m, v in corrs.items() if abs(v) > 0.2]
                        if sig:
                            sig_str = ", ".join(
                                f"{m}({v:+.2f})" for m, v in sorted(sig, key=lambda x: -abs(x[1]))
                            )
                            print(f"    [{obj_name} drivers] {sig_str}")
            else:
                analysis = {"weaknesses": [], "suggestions": [],
                            "feedback": "", "overall_assessment": ""}

            # ---- 构建本轮完整候选打分表 (所有候选) ----
            candidate_scores = []
            for i, (cand_seq, r) in enumerate(zip(candidates, results)):
                source = gen_meta.get("sources", {}).get(i, "unknown")
                candidate_scores.append({
                    "index": i,
                    "sequence": cand_seq,
                    "source": source,  # "new_generation" | "elite_mutation:parent_N"
                    "te_score": round(r.te_score, 6),
                    "stability_score": round(r.stability_score, 6),
                    "expression_score": round(r.expression_score, 6),
                    "cai": round(r.cai, 6),
                    "gc_content": round(r.gc_content, 6),
                    "mfe": round(r.mfe, 3),
                    "max_stem_len": r.max_stem_len,
                    "max_homopolymer": r.max_homopolymer,
                    "immunogenicity_risk": round(r.immunogenicity_risk, 6),
                    "is_feasible": r.is_feasible,
                })

            # ---- 构建本轮 Pareto 前沿完整信息 ----
            pareto_front_details = []
            for s in pareto_solutions:
                r = s.result
                pareto_front_details.append({
                    "rank": s.rank,
                    "crowding_distance": round(s.crowding_distance, 6),
                    "sequence": r.sequence,
                    "objectives": {
                        "TE": round(r.te_score, 6),
                        "Stability": round(r.stability_score, 6),
                        "Expression": round(r.expression_score, 6),
                    },
                    "constraints": {
                        "CAI": round(r.cai, 6),
                        "GC%": round(r.gc_content, 6),
                        "MFE": round(r.mfe, 3),
                        "max_stem_len": r.max_stem_len,
                        "max_homopolymer": r.max_homopolymer,
                        "immunogenicity_risk": round(r.immunogenicity_risk, 6),
                    },
                    "is_feasible": r.is_feasible,
                })

            # 记录完整历史 (不截断)
            self.history.append({
                "round": round_num,
                "temperature": temperature,
                "num_candidates": len(candidates),
                "feasible_count": feasible_count,
                "pareto_front_size": front_size,
                "hv": round(hv, 6),
                "hv_improvement": round(hv_improvement, 6),
                # 完整反馈 (不截断)
                "feedback": analysis["feedback"],
                "weaknesses": analysis["weaknesses"],
                "suggestions": analysis["suggestions"],
                "overall_assessment": analysis.get("overall_assessment", ""),
                "generation_directives": analysis.get("generation_directives", {}),
                "feedback_raw": analysis.get("raw_response", ""),
                # 生成元数据
                "generation_meta": gen_meta,
                # 本轮全部候选打分
                "candidate_scores": candidate_scores,
                # 本轮 Pareto 前沿
                "pareto_front_details": pareto_front_details,
                # 精英序列 (供下一轮突变参考)
                "elite_sequences": list(self._elite_sequences),
            })

            # 收敛判定 (仅在达到最少轮数后检查)
            # 修复 P0: 原逻辑 hv_improvement < threshold 会将 HV 下降 (负值) 误判为收敛
            # 新逻辑: 只有 0 ≤ ΔHV < threshold 才判定为收敛 (正改进但很小)
            #         ΔHV < 0 (HV 下降) 时不收敛，继续迭代
            if round_num <= MIN_ITERATION_ROUNDS:
                print(f"  ⏳ 最少 {MIN_ITERATION_ROUNDS} 轮保护中 (当前第 {round_num} 轮)，跳过收敛检查")
            elif 0 <= hv_improvement < HV_CONVERGENCE_THRESHOLD:
                print(f"  Converged! ΔHV {hv_improvement:.4f} ∈ [0, {HV_CONVERGENCE_THRESHOLD})")
                # 安全回退: 如果历史最优为空 (所有轮都无可行解), 返回当前轮的解
                return_solutions = self._best_solutions if self._best_solutions else pareto_solutions
                return_hv = self._best_hv if self._best_solutions else hv
                return {
                    "best_solutions": return_solutions,
                    "history": self.history,
                    "good_sequences_queue": self._good_sequences_queue,
                    "total_rounds": round_num,
                    "final_hv": return_hv,
                    "converged": True,
                }
            elif hv_improvement < 0:
                print(f"  ⚠️ HV 下降 ({hv_improvement:.4f}), 继续迭代寻找更优解...")

            self.prev_hv = hv

        print(f"  Reached max rounds ({MAX_ITERATION_ROUNDS})")
        return_solutions = self._best_solutions if self._best_solutions else pareto_solutions
        return_hv = self._best_hv if self._best_solutions else hv
        return {
            "best_solutions": return_solutions,
            "history": self.history,
            "good_sequences_queue": self._good_sequences_queue,
            "total_rounds": MAX_ITERATION_ROUNDS,
            "final_hv": return_hv,
            "converged": False,
        }

    # ------------------------------------------------------------------
    #  候选生成: 新生成 + 精英突变
    # ------------------------------------------------------------------

    def _generate_candidates(self, protein_seq: str,
                            round_num: int,
                            temperature: float,
                            feedback: Optional[str],
                            directives: Optional[dict] = None) -> Tuple[List[str], Dict[str, Any]]:
        """生成候选序列, 返回 (sequences, generation_meta)

        第 1 轮: 全部 NUWA 生成
        第 2 轮起: NUWA 生成 (1-MUTATE_RATIO) + 精英突变 MUTATE_RATIO

        generation_meta = {
            "new_count": N,
            "mutated_count": M,
            "sources": {index: "new_generation" | "elite_mutation:parent_0" | ...},
            "mutation_details": [{...}],   # 每条突变体的详细记录
            "elite_parents": [...],
        }
        """
        num_total = NUM_CANDIDATES
        gen_meta: Dict[str, Any] = {
            "new_count": 0,
            "mutated_count": 0,
            "sources": {},
            "mutation_details": [],
            "elite_parents": list(self._elite_sequences),
        }

        # 从 directives 提取生成指令 (缺省用类常量, 保留原行为)
        directives = directives or {}
        cai_intensity = directives.get("cai_intensity", self.CAI_OPTIMIZE_INTENSITY)
        mutate_fraction = directives.get("mutate_fraction", self.MUTATE_RATIO)
        focus_objective = directives.get("focus_objective", "balanced")

        # 第 1 轮或没有精英序列时: 全部新生成
        if round_num == 1 or not self._elite_sequences:
            gen_meta["new_count"] = num_total
            print(f"  Generating {num_total} candidates (temp={temperature:.1f})...")
            candidates = registry.generate(
                self.selected_model, protein_seq,
                num=num_total,
                feedback=feedback,
                temperature=temperature,
                class_id=self.class_id,
            )
            # Round 1 也对 30% 候选做 CAI 优化 (提升初始 CAI 基线)
            if self.codon_table:
                candidates = self._apply_cai_optimization_batch(
                    candidates, self.CAI_OPTIMIZE_RATIO, gen_meta, "new_generation",
                    intensity=cai_intensity
                )
            for i in range(len(candidates)):
                gen_meta["sources"][i] = "new_generation"
            return candidates, gen_meta

        # 第 2 轮起: 混合策略
        num_mutate = int(round(num_total * mutate_fraction))
        num_new = num_total - num_mutate
        gen_meta["new_count"] = num_new
        gen_meta["mutated_count"] = num_mutate

        print(f"  Generating {num_new} new + {num_mutate} mutated candidates (temp={temperature:.1f})...")

        # 新生成
        new_candidates: List[str] = []
        if num_new > 0:
            new_candidates = registry.generate(
                self.selected_model, protein_seq,
                num=num_new,
                feedback=feedback,
                temperature=temperature,
                class_id=self.class_id,
            )
            # 对新生成候选应用 CAI 优化 (比例随轮次递增, 后期更激进优化 CAI)
            if self.codon_table:
                cai_ratio = min(0.6, self.CAI_OPTIMIZE_RATIO + 0.1 * (round_num - 1))
                new_candidates = self._apply_cai_optimization_batch(
                    new_candidates, cai_ratio, gen_meta, "new_generation",
                    intensity=cai_intensity
                )

        # 精英突变 (按 focus_objective 偏置选择精英)
        mutated_candidates, mutation_details = self._mutate_elites(
            protein_seq, num_mutate, focus_objective=focus_objective
        )

        # 对精英突变也应用 CAI 优化 (突破 CAI 停滞)
        if self.codon_table and mutated_candidates:
            mutated_candidates = self._apply_cai_optimization_batch(
                mutated_candidates, 0.5, gen_meta, "elite_mutation",
                intensity=cai_intensity
            )

        # 合并 + 标记来源
        all_candidates = new_candidates + mutated_candidates
        for i in range(len(new_candidates)):
            gen_meta["sources"][i] = "new_generation"
        for j, detail in enumerate(mutation_details):
            idx = len(new_candidates) + j
            gen_meta["sources"][idx] = f"elite_mutation:{detail.get('elite_index', '?')}"
        gen_meta["mutation_details"] = mutation_details

        return all_candidates, gen_meta

    @staticmethod
    def _sanitize_directives(d: Optional[dict]) -> dict:
        """清洗 generation_directives: 钳制范围并补默认值, 保证非法值不破坏生成。"""
        d = d or {}
        def clamp(v, lo, hi, default):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return default
            return max(lo, min(hi, v))
        return {
            "focus_objective": d.get("focus_objective", "balanced"),
            "cai_intensity": clamp(d.get("cai_intensity"), 0.1, 0.5, 0.25),
            "mutate_fraction": clamp(d.get("mutate_fraction"), 0.0, 1.0, 0.5),
            "explore": clamp(d.get("explore"), 0.0, 1.0, 1.0),
        }

    def _mutate_elites(self, protein_seq: str, num_needed: int,
                      focus_objective: str = "balanced") -> Tuple[List[str], List[Dict]]:
        """对精英序列做同义密码子替换, 生成突变体

        修复 P1: 突变后检查结构约束 (homopolymer/MFE/stem/GC)，
        拒绝违反约束的突变体，避免产生全 Feasible=False 的突变批次。

        返回: (mutated_sequences, mutation_details)
          mutation_details[i] = {
            "elite_index": 0,           # 精英序列在 _elite_sequences 中的索引
            "parent_sequence": "...",   # 父本序列
            "mutated_sequence": "...",  # 突变后序列
            "num_mutations": 2,
            "mutations": [             # 具体突变位点
              {"codon_index": 5, "old_codon": "AUG", "new_codon": "AUA", "amino_acid": "M"},
              ...
            ],
          }
        """
        if not self._elite_sequences:
            return [], []

        sequences: List[str] = []
        details: List[Dict] = []
        attempts = 0
        max_attempts = num_needed * 10
        rejected = 0

        elites = self._elite_solutions
        # 按 focus_objective 偏置: 该目标越弱 (分数越低) 越优先被突变
        weights = None
        if focus_objective in ("te", "stability", "expression") and elites:
            key = {"te": "te_score", "stability": "stability_score",
                   "expression": "expression_score"}[focus_objective]
            raw = [max(0.0, 1.0 - getattr(e.result, key, 0.5)) for e in elites]
            weights = raw if sum(raw) > 0 else None

        while len(sequences) < num_needed and attempts < max_attempts:
            attempts += 1
            if weights:
                elite_idx = random.choices(range(len(elites)), weights=weights)[0]
            else:
                elite_idx = random.randrange(len(elites))
            parent_seq = elites[elite_idx].result.sequence
            mutated_seq, mutation_record = self._synonymous_substitute(parent_seq, protein_seq)
            if mutated_seq is not None and mutated_seq != parent_seq:
                # 约束预检: 快速检查结构约束 (不需要完整评估)
                if self._quick_constraint_check(mutated_seq):
                    sequences.append(mutated_seq)
                    details.append({
                        "elite_index": elite_idx,
                        "parent_sequence": parent_seq,
                        "mutated_sequence": mutated_seq,
                        "num_mutations": mutation_record.get("num_mutations", 0),
                        "mutations": mutation_record.get("mutations", []),
                    })
                else:
                    rejected += 1

        if rejected > 0:
            print(f"    (突变约束预检: {rejected} 条被拒绝, {len(sequences)} 条通过)")

        return sequences, details

    def _quick_constraint_check(self, sequence: str) -> bool:
        """快速结构约束预检 — 不需要完整评估, 仅检查可直接从序列计算的约束

        检查: GC%, homopolymer, MFE, max_stem
        跳过: CAI (需要密码子表), immunogenicity (需要蛋白序列)
        """
        bounds = self.constraint_bounds

        # GC 含量
        gc = self.evaluator._compute_gc(sequence)
        if gc < bounds.get("gc_min", 0) or gc > bounds.get("gc_max", 1):
            return False

        # 最大同聚物
        max_homo = self.evaluator._compute_max_homopolymer(sequence)
        if max_homo >= bounds.get("max_homopolymer", 999):
            return False

        # MFE (ViennaRNA, 可能耗时但值得检查)
        mfe = self.evaluator._compute_mfe(sequence)
        if mfe > bounds.get("mfe_max", 0):
            return False
        if mfe < bounds.get("mfe_min", -99999):
            return False

        # 最大茎区
        max_stem = self.evaluator._compute_max_stem(sequence)
        if max_stem >= bounds.get("max_stem_length", 999):
            return False

        return True

    @staticmethod
    def _synonymous_substitute(seq: str, protein_seq: str) -> Tuple[Optional[str], Dict]:
        """对 mRNA 序列做 1~3 个同义密码子替换

        Returns: (mutated_sequence_or_None, mutation_record)
          mutation_record = {"num_mutations": N, "mutations": [{"codon_index":..., "old_codon":..., "new_codon":..., "amino_acid":...}]}
        """
        import re
        mutation_record: Dict[str, Any] = {"num_mutations": 0, "mutations": []}

        # 清理序列: 去除空格, 转为 RNA (U 代替 T)
        clean = seq.replace(" ", "").upper().replace("T", "U")
        if len(clean) < 3:
            return None, mutation_record

        # 按密码子拆分
        codons = [clean[i:i+3] for i in range(0, len(clean) - (len(clean) % 3), 3)]
        if not codons:
            return None, mutation_record

        # 随机选 1~3 个位置做同义替换
        num_mut = random.randint(1, min(IterationController.MAX_MUTATIONS_PER_SEQ, len(codons)))
        positions = random.sample(range(len(codons)), num_mut)

        new_codons = list(codons)
        for pos in positions:
            old_codon = codons[pos]
            aa = CODON_TO_AA.get(old_codon)
            if aa is None or aa == "*":   # 跳过终止密码子和未知
                continue
            alternatives = [c for c in AA_TO_CODONS.get(aa, []) if c != old_codon]
            if alternatives:
                new_codon = random.choice(alternatives)
                new_codons[pos] = new_codon
                mutation_record["mutations"].append({
                    "codon_index": pos,
                    "old_codon": old_codon,
                    "new_codon": new_codon,
                    "amino_acid": aa,
                })

        mutation_record["num_mutations"] = len(mutation_record["mutations"])
        if mutation_record["num_mutations"] == 0:
            return None, mutation_record

        # 2026-08-22: 返回空格分隔格式, 与新生成候选/CAI 优化输出保持一致
        # (此前返回无空格串, 导致评估与预检的序列格式不对称)
        return " ".join(new_codons), mutation_record

    # ------------------------------------------------------------------
    #  CAI 定向优化: 用宿主密码子表替换低适应性密码子
    # ------------------------------------------------------------------

    def _apply_cai_optimization_batch(self, candidates: List[str],
                                       ratio: float, gen_meta: Dict,
                                       source_tag: str,
                                       intensity: float = None) -> List[str]:
        """对一批候选序列应用 CAI 优化

        Args:
            candidates: 候选序列列表
            ratio: 应用 CAI 优化的比例 (0.0-1.0)
            gen_meta: 生成元数据 (记录哪些被优化了)
            source_tag: 来源标记 ("new_generation" / "elite_mutation")

        Returns:
            优化后的候选序列列表 (长度不变)
        """
        if not self.codon_table or ratio <= 0 or not candidates:
            return candidates

        num_to_optimize = max(1, int(len(candidates) * ratio))
        # 随机选择要优化的索引
        optimize_indices = set(random.sample(range(len(candidates)), min(num_to_optimize, len(candidates))))

        optimized = list(candidates)
        cai_count = 0
        for i in optimize_indices:
            optimized_seq = self._cai_optimize(candidates[i], intensity=intensity)
            if optimized_seq != candidates[i]:
                optimized[i] = optimized_seq
                cai_count += 1

        if cai_count > 0:
            print(f"    (CAI 优化: {cai_count}/{len(candidates)} 条候选已优化)")

        return optimized

    def _cai_optimize(self, sequence: str,
                      intensity: float = None) -> str:
        """CAI 定向优化 — 将低适应性密码子替换为宿主偏好的高适应性同义密码子

        原理:
          NUWA MLM 的 logits 不响应 LLM 文本反馈 "提高 CAI"，
          因此需要一个确定性的后处理步骤来直接优化 CAI。

        策略:
          1. 按密码子拆分序列
          2. 计算每个密码子的相对适应性 (weight / max_weight_for_aa)
          3. 找出适应性最低的 intensity 比例的密码子
          4. 将它们替换为该氨基酸对应的最优密码子
          5. 保留原始序列格式 (空格分隔)

        Args:
            sequence: mRNA 序列 (可能含空格)
            intensity: 替换比例 (None 时用类常量 CAI_OPTIMIZE_INTENSITY)

        Returns:
            优化后的 mRNA 序列
        """
        if not self.codon_table:
            return sequence

        if intensity is None:
            intensity = self.CAI_OPTIMIZE_INTENSITY

        clean = sequence.replace(" ", "").upper().replace("T", "U")
        if len(clean) < 3:
            return sequence

        codons = [clean[i:i+3] for i in range(0, len(clean) - (len(clean) % 3), 3)]
        if not codons:
            return sequence

        # 构建 AA → (codon, weight) 排序表 (DNA 格式权重, RNA 格式密码子)
        aa_best = {}  # {aa: [(rna_codon, weight), ...]} 降序
        for aa, codon_list in AA_TO_CODONS.items():
            if aa == "*":
                continue
            weighted = []
            for c in codon_list:
                dna_c = c.replace("U", "T")
                w = self.codon_table.get(dna_c, 0)
                weighted.append((c, w))
            weighted.sort(key=lambda x: -x[1])
            aa_best[aa] = weighted

        # 计算每个密码子的相对适应性
        codon_scores = []
        for i, c in enumerate(codons):
            aa = CODON_TO_AA.get(c)
            if aa is None or aa == "*" or aa not in aa_best:
                codon_scores.append((i, 1.0))  # 不优化
                continue
            best_list = aa_best[aa]
            max_w = best_list[0][1] if best_list else 1
            dna_c = c.replace("U", "T")
            w = self.codon_table.get(dna_c, 0)
            rel = w / max_w if max_w > 0 else 0
            codon_scores.append((i, rel))

        # 按相对适应性升序排列, 优化最低的 intensity 比例
        codon_scores.sort(key=lambda x: x[1])
        num_to_optimize = max(1, int(len(codons) * intensity))
        positions_to_optimize = set(idx for idx, _ in codon_scores[:num_to_optimize])

        new_codons = list(codons)
        replaced = 0
        for i in positions_to_optimize:
            aa = CODON_TO_AA.get(codons[i])
            if aa is None or aa == "*" or aa not in aa_best:
                continue
            best_list = aa_best[aa]
            if best_list and best_list[0][0] != codons[i] and best_list[0][1] > 0:
                new_codons[i] = best_list[0][0]
                replaced += 1

        if replaced == 0:
            return sequence

        # 保持原始格式 (如果有空格则输出空格分隔)
        had_spaces = " " in sequence
        result = " ".join(new_codons) if had_spaces else "".join(new_codons)
        return result
