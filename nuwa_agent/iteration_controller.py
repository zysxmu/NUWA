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
from round_deliberation import RoundDeliberation
from model_registry import registry
from config import (MAX_ITERATION_ROUNDS, MIN_ITERATION_ROUNDS,
                    HV_CONVERGENCE_THRESHOLD, HV_CONVERGENCE_PATIENCE,
                    HV_BEST_GAP_TOLERANCE, NUM_CANDIDATES, PARETO_TOP_K,
                    MULTIAGENT_ENABLED)

import random
import math
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
                 codon_table: dict = None,
                 deliberator: Optional[RoundDeliberation] = None):
        self.selected_model = selected_model
        self.constraint_bounds = constraint_bounds
        self.evaluator = evaluator
        self.class_id = class_id
        self.codon_table = codon_table    # 宿主参考密码子表 (用于 CAI 优化)
        self.checker = ConstraintChecker(constraint_bounds)
        self.selector = ParetoSelector()
        self.analyzer = FeedbackAnalyzer()
        self.deliberator = deliberator if deliberator is not None else RoundDeliberation()

        self.history = []
        self.prev_hv = 0.0
        self._elite_sequences: list = []  # 上轮 Pareto 前沿序列 (RNA 字符串)
        self._elite_solutions: list = []  # 上轮 Pareto 前沿解 (含分数, 供 focus 偏置)
        self._last_directives: dict = None  # 上一轮中央 LLM 的结构化决策
        self._best_hv: float = 0.0        # 历史最高 HV
        self._best_solutions: list = []   # HV 最高轮的 Pareto 解
        self._best_round: int = 0         # HV 最高轮的轮次
        self._stable_rounds: int = 0      # 接近历史最优且变化很小的连续轮数
        self._consecutive_hv_declines: int = 0
        self._elite_source: str = "none"
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
        self.history = []
        self.prev_hv = 0.0
        self._best_hv = 0.0
        self._best_solutions = []
        self._best_round = 0
        self._stable_rounds = 0
        self._consecutive_hv_declines = 0
        self._elite_source = "none"
        self._good_sequences_queue = []
        self._good_seq_seen = set()
        self._elite_sequences = []   # 每轮更新
        self._elite_solutions = []
        self._last_directives = None

        for round_num in range(1, MAX_ITERATION_ROUNDS + 1):
            print(f"\n[Round {round_num}/{MAX_ITERATION_ROUNDS}]")

            # 第 1 轮使用预注册基线；之后执行上一轮中央 LLM 的决策。
            # 无可用父本时强制全新生成，并记录实际应用值。
            requested = self._last_directives or self._default_decision(round_num)
            directives = self._sanitize_directives(requested, round_num)
            if round_num == 1 or not self._elite_sequences:
                directives["mutate_fraction"] = 0.0
            temperature = directives["temperature"]

            # Step A: 生成候选 (新生成 + 精英突变), 指令随 feedback 动态调整
            candidates, gen_meta = self._generate_candidates(
                protein_seq, round_num, temperature, feedback, directives=directives
            )
            self._validate_candidate_translations(candidates, protein_seq)

            # Step B: 评估 (外部生物信息学工具)
            print("  Evaluating with external bioinformatics tools...")
            results = self.evaluator.evaluate_batch(candidates, protein_seq)

            # Step C: 约束 + Pareto
            results = self.checker.check_batch(results)
            if not results:
                raise RuntimeError("本轮没有可评估候选，无法继续优化")
            feasible_count = sum(1 for r in results if r.is_feasible)
            print(f"  Feasible: {feasible_count}/{len(results)}")

            # 保留全部可行解完成非支配排序与 HV。
            pareto_solutions = self.selector.select(results, top_k=None)

            hv = self.selector.compute_hypervolume(pareto_solutions)
            hv_improvement = hv - self.prev_hv
            front_size = len([s for s in pareto_solutions if s.rank == 0])

            # 追踪历史最优 HV 和对应解 (修复 P0: 收敛时返回最优轮而非当前轮)
            if hv > self._best_hv:
                self._best_hv = hv
                self._best_solutions = [s for s in pareto_solutions if s.rank == 0]
                self._best_round = round_num
                print(f"  📈 新最优 HV: {hv:.4f} (Round {round_num})")

            best_gap = max(0.0, self._best_hv - hv)
            if hv_improvement < 0:
                self._consecutive_hv_declines += 1
            else:
                self._consecutive_hv_declines = 0
            if (best_gap <= HV_BEST_GAP_TOLERANCE
                    and abs(hv_improvement) < HV_CONVERGENCE_THRESHOLD):
                self._stable_rounds += 1
            else:
                self._stable_rounds = 0

            # Do not let a regressed generation replace the parent archive and
            # recursively amplify its own degradation. Once the current front
            # falls materially below the best, mutate the historical best front.
            current_elites = [s for s in pareto_solutions if s.rank == 0][:PARETO_TOP_K]
            if self._best_solutions and best_gap > HV_BEST_GAP_TOLERANCE:
                elite_parents = self._best_solutions[:PARETO_TOP_K]
                self._elite_source = f"historical_best_round_{self._best_round}"
            else:
                elite_parents = current_elites
                self._elite_source = f"current_round_{round_num}"
            if not elite_parents and pareto_solutions:
                elite_parents = [s for s in pareto_solutions if s.rank == 999][:PARETO_TOP_K]
                self._elite_source = f"recovery_round_{round_num}"
            self._elite_sequences = [s.result.sequence for s in elite_parents]
            self._elite_solutions = elite_parents

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
                          f"| Feasible={r.is_feasible}")

            # 打印次优前沿 (rank=1) 摘要
            rank1 = [s for s in pareto_solutions if s.rank == 1]
            if rank1:
                best_r1 = max(rank1, key=lambda s: s.result.expression_score)
                r = best_r1.result
                print(f"  Rank-1 best: TE={r.te_score:.3f} Stab={r.stability_score:.3f} "
                      f"Expr={r.expression_score:.3f}")

            print(f"  HV: {hv:.4f} (Δ{hv_improvement:+.4f})")
            print(f"  Pareto front: {front_size} solutions")

            # Step D: 数值证据先计算，再由专家往返讨论和中央 LLM 裁决。
            # 相关性只描述本轮样本关联，不赋予 Agent 修改分数/约束的权限。
            score_corr = (self.analyzer._compute_score_correlations(results)
                          if len(results) >= 5 else {})
            round_summary = self._build_round_summary(
                round_num, protein_seq, host_organism, results, pareto_solutions,
                hv, hv_improvement, directives, score_corr, gen_meta,
            )
            converged_now = self._has_converged(round_num, feasible_count, hv)
            discussion = {"decision": None, "transcript": [],
                          "fallback_used": False, "error": None,
                          "skip_reason": None}
            if round_num < MAX_ITERATION_ROUNDS and not converged_now:
                fallback = self._default_decision(round_num + 1)
                if MULTIAGENT_ENABLED:
                    try:
                        discussion = self.deliberator.decide(round_summary, fallback)
                    except Exception as exc:
                        discussion = {"decision": fallback, "transcript": [],
                                      "fallback_used": True,
                                      "error": f"deliberation: {type(exc).__name__}"}
                else:
                    discussion = {"decision": fallback, "transcript": [],
                                  "fallback_used": True,
                                  "error": "multi-agent discussion disabled"}
                try:
                    proposed = self._sanitize_directives(
                        discussion.get("decision"), round_num + 1
                    )
                    discussion["requested_decision"] = dict(proposed)
                    self._last_directives, guardrail = self._guard_next_decision(
                        proposed, directives, round_summary
                    )
                    self._last_directives["rationale"] = self._execution_summary(
                        proposed, self._last_directives, guardrail
                    )
                    discussion["guardrail"] = guardrail
                    discussion["adjusted_decision"] = dict(self._last_directives)
                    discussion["decision"] = dict(self._last_directives)
                except ValueError as exc:
                    self._last_directives = self._sanitize_directives(fallback, round_num + 1)
                    discussion["requested_decision"] = None
                    discussion["adjusted_decision"] = dict(self._last_directives)
                    discussion["decision"] = dict(self._last_directives)
                    discussion["fallback_used"] = True
                    discussion["error"] = f"controller validation: {type(exc).__name__}"
                    discussion["guardrail"] = {"applied": False, "adjustments": []}
                feedback = self._last_directives.get("rationale", "")
                print(f"  Central LLM next-round decision: {self._last_directives}")
            elif round_num >= MAX_ITERATION_ROUNDS:
                discussion["skip_reason"] = "final_round_has_no_next_round"
            elif converged_now:
                discussion["skip_reason"] = "optimization_converged"

            central_decision = discussion.get("decision")
            analysis = {
                "feedback": (central_decision or {}).get("rationale", ""),
                "weaknesses": [],
                "suggestions": [],
                "overall_assessment": "no feasible candidates" if not feasible_count else "",
                "generation_directives": central_decision or {},
            }

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
                "round_summary": round_summary,
                "score_correlations": score_corr,
                "discussion_transcript": discussion.get("transcript", []),
                "decision_for_round": (round_num + 1 if central_decision else None),
                "central_requested_decision": discussion.get("requested_decision"),
                "guardrail_adjusted_decision": discussion.get("adjusted_decision"),
                "central_decision": central_decision,
                "applied_decision": {k: directives.get(k) for k in
                                     ("temperature", "mutate_fraction",
                                      "substitutions_per_candidate", "parent_focus",
                                      "evidence_ids")},
                "applied_decision_origin_round": (round_num - 1 if round_num > 1 else None),
                "applied_decision_source": (
                    "previous_round_guardrail_adjusted_decision"
                    if round_num > 1 else "preregistered_round_1_baseline"
                ),
                "discussion_fallback_used": bool(discussion.get("fallback_used", False)),
                "discussion_error": discussion.get("error"),
                "discussion_decision_source": discussion.get("decision_source"),
                "decision_guardrail": discussion.get("guardrail", {}),
                "discussion_skipped": not bool(discussion.get("transcript")),
                "discussion_skip_reason": discussion.get("skip_reason"),
                # 完整反馈 (不截断)
                "feedback": analysis["feedback"],
                "next_round_execution_summary": analysis["feedback"],
                "weaknesses": analysis["weaknesses"],
                "suggestions": analysis["suggestions"],
                "overall_assessment": analysis.get("overall_assessment", ""),
                "generation_directives": analysis.get("generation_directives", {}),
                # 生成元数据
                "generation_meta": gen_meta,
                # 本轮全部候选打分
                "candidate_scores": candidate_scores,
                # 本轮 Pareto 前沿
                "pareto_front_details": pareto_front_details,
                # 精英序列 (供下一轮突变参考)
                "elite_sequences": list(self._elite_sequences),
            })

            # 收敛判定：需要连续稳定，并且当前 HV 仍接近历史最优。
            # 这防止“大幅退化后的一次小反弹”被误报为收敛。
            if round_num <= MIN_ITERATION_ROUNDS:
                print(f"  ⏳ 最少 {MIN_ITERATION_ROUNDS} 轮保护中 (当前第 {round_num} 轮)，跳过收敛检查")
            elif converged_now:
                print(f"  Converged! {self._stable_rounds} stable rounds, "
                      f"best gap={best_gap:.4f}")
                # 安全回退: 如果历史最优为空 (所有轮都无可行解), 返回当前轮的解
                recovery = [s for s in pareto_solutions if s.rank == 999][:PARETO_TOP_K]
                return_solutions = self._best_solutions if self._best_solutions else recovery
                return_hv = self._best_hv if self._best_solutions else hv
                return {
                    "best_solutions": return_solutions,
                    "history": self.history,
                    "good_sequences_queue": self._good_sequences_queue,
                    "total_rounds": round_num,
                    "final_hv": return_hv,
                    "best_hv": self._best_hv,
                    "last_round_hv": hv,
                    "best_round": self._best_round,
                    "converged": True,
                }
            elif feasible_count == 0:
                print("  ⚠️ 本轮无可行解，HV=0 不作为收敛证据；下一轮进入恢复搜索")
            elif hv_improvement < 0:
                print(f"  ⚠️ HV 下降 ({hv_improvement:.4f}), 继续迭代寻找更优解...")

            self.prev_hv = hv

        print(f"  Reached max rounds ({MAX_ITERATION_ROUNDS})")
        recovery = [s for s in pareto_solutions if s.rank == 999][:PARETO_TOP_K]
        return_solutions = self._best_solutions if self._best_solutions else recovery
        return_hv = self._best_hv if self._best_solutions else hv
        return {
            "best_solutions": return_solutions,
            "history": self.history,
            "good_sequences_queue": self._good_sequences_queue,
            "total_rounds": MAX_ITERATION_ROUNDS,
            "final_hv": return_hv,
            "best_hv": self._best_hv,
            "last_round_hv": hv,
            "best_round": self._best_round,
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
            "planned_mutated_count": 0,
            "mutation_shortfall": 0,
            "sources": {},
            "mutation_details": [],
            "cai_edit_details": [],
            "elite_parents": list(self._elite_sequences),
            "elite_parent_source": self._elite_source,
        }

        # 中央 LLM 仅控制温度、生成/突变配比、同义替换次数及父本侧重。
        # CAI 后处理保持预注册的确定性日程，避免额外混杂因素。
        directives = directives or self._default_decision(round_num)
        mutate_fraction = directives["mutate_fraction"]
        focus_objective = directives["parent_focus"]
        substitutions = directives["substitutions_per_candidate"]
        cai_ratio = min(0.6, self.CAI_OPTIMIZE_RATIO + 0.1 * (round_num - 1))
        # These are the parameters that generated this round. Keep the legacy
        # key for compatibility, but label the authoritative provenance clearly.
        gen_meta["requested_decision"] = dict(directives)
        gen_meta["applied_decision"] = dict(directives)
        gen_meta["cai_application_fraction"] = cai_ratio
        gen_meta["cai_intensity"] = self.CAI_OPTIMIZE_INTENSITY

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
                    candidates, cai_ratio, gen_meta, "new_generation",
                    intensity=self.CAI_OPTIMIZE_INTENSITY
                )
            for i in range(len(candidates)):
                gen_meta["sources"][i] = "new_generation"
            if len(candidates) != num_total:
                raise RuntimeError("NUWA did not generate the requested candidate budget")
            gen_meta["actual_mutate_fraction"] = 0.0
            return candidates, gen_meta

        # 第 2 轮起: 混合策略
        num_mutate = int(round(num_total * mutate_fraction))
        num_new = num_total - num_mutate
        gen_meta["planned_mutated_count"] = num_mutate

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
                new_candidates = self._apply_cai_optimization_batch(
                    new_candidates, cai_ratio, gen_meta, "new_generation",
                    intensity=self.CAI_OPTIMIZE_INTENSITY
                )

        # 精英突变 (按 focus_objective 偏置选择精英)
        mutated_candidates, mutation_details = self._mutate_elites(
            protein_seq, num_mutate, focus_objective=focus_objective,
            substitutions_per_candidate=substitutions,
        )

        # 对精英突变也应用 CAI 优化 (突破 CAI 停滞)
        if self.codon_table and mutated_candidates:
            mutated_candidates = self._apply_cai_optimization_batch(
                mutated_candidates, cai_ratio, gen_meta, "elite_mutation",
                intensity=self.CAI_OPTIMIZE_INTENSITY
            )
        for detail, final_sequence in zip(mutation_details, mutated_candidates):
            detail["post_cai_sequence"] = final_sequence
            detail["evaluated_sequence"] = final_sequence
            cai_edits = self._codon_edit_diff(
                detail["mutated_sequence"], final_sequence
            )
            detail["cai_postprocessing"] = {
                "applied": bool(cai_edits),
                "num_edits": len(cai_edits),
                "edits": cai_edits,
            }
            detail["provenance_chain"] = [
                "elite_parent", "synonymous_mutation",
                "cai_postprocessing" if cai_edits else "cai_postprocessing_no_change",
                "evaluated_sequence",
            ]

        initial_new_count = len(new_candidates)
        # 预检可能拒绝部分同义子代；用同温度 NUWA 生成补齐，保证每轮
        # 的候选和评估预算固定，并记录实际突变比例。
        shortfall = num_total - len(new_candidates) - len(mutated_candidates)
        if shortfall > 0:
            gen_meta["mutation_shortfall"] = shortfall
            extra = registry.generate(
                self.selected_model, protein_seq, num=shortfall,
                feedback=feedback, temperature=temperature, class_id=self.class_id,
            )
            if len(extra) != shortfall:
                raise RuntimeError("NUWA did not replenish the candidate budget")
            if self.codon_table:
                extra = self._apply_cai_optimization_batch(
                    extra, cai_ratio, gen_meta, "new_generation_replenishment",
                    intensity=self.CAI_OPTIMIZE_INTENSITY,
                )
            new_candidates.extend(extra)

        # 合并 + 标记来源
        all_candidates = new_candidates + mutated_candidates
        gen_meta["new_count"] = len(new_candidates)
        gen_meta["mutated_count"] = len(mutated_candidates)
        gen_meta["actual_mutate_fraction"] = (
            len(mutated_candidates) / len(all_candidates) if all_candidates else 0.0
        )
        for i in range(len(new_candidates)):
            gen_meta["sources"][i] = (
                "new_generation" if i < initial_new_count
                else "new_generation_replenishment"
            )
        for j, detail in enumerate(mutation_details):
            idx = len(new_candidates) + j
            gen_meta["sources"][idx] = f"elite_mutation:{detail.get('elite_index', '?')}"
        gen_meta["mutation_details"] = mutation_details

        if len(all_candidates) != num_total:
            raise RuntimeError("candidate budget changed after generation and mutation")

        return all_candidates, gen_meta

    @staticmethod
    def _default_decision(round_num: int) -> dict:
        """预注册回退策略：前 40% 轮次由 T=1.0 退火至 0.3。"""
        decay_rounds = max(2, int(math.ceil(MAX_ITERATION_ROUNDS * 0.4)))
        progress = min(1.0, (round_num - 1) / (decay_rounds - 1))
        return {
            "temperature": round(1.0 - 0.7 * progress, 6),
            "mutate_fraction": 0.0 if round_num == 1 else 0.5,
            "substitutions_per_candidate": 2,
            "parent_focus": "balanced",
            "evidence_ids": [],
            "rationale": "Prespecified temperature schedule and 50% elite mutation fallback.",
        }

    @staticmethod
    def _sanitize_directives(d: Optional[dict], round_num: int) -> dict:
        """控制器再次校验中央决策；越界值不得进入生成/突变模块。"""
        if not isinstance(d, dict):
            raise ValueError("central decision must be a JSON object")
        try:
            temperature = float(d["temperature"])
            mutate_fraction = float(d["mutate_fraction"])
            substitutions = d["substitutions_per_candidate"]
            parent_focus = d["parent_focus"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid central decision: {exc}") from exc
        if not math.isfinite(temperature) or not 0.3 <= temperature <= 1.0:
            raise ValueError("temperature must be finite and in [0.3, 1.0]")
        if not math.isfinite(mutate_fraction) or not 0.0 <= mutate_fraction <= 0.75:
            raise ValueError("mutate_fraction must be finite and in [0, 0.75]")
        if type(substitutions) is not int or not 1 <= substitutions <= 3:
            raise ValueError("substitutions_per_candidate must be an integer 1..3")
        if parent_focus not in ("balanced", "te", "stability", "expression"):
            raise ValueError("invalid parent_focus")
        evidence_ids = d.get("evidence_ids", [])
        if not isinstance(evidence_ids, list) or not all(isinstance(x, str) for x in evidence_ids):
            raise ValueError("evidence_ids must be a list of strings")
        rationale = d.get("rationale", "")
        if not isinstance(rationale, str):
            raise ValueError("rationale must be a string")
        return {
            "temperature": temperature,
            "mutate_fraction": 0.0 if round_num == 1 else mutate_fraction,
            "substitutions_per_candidate": substitutions,
            "parent_focus": parent_focus,
            "evidence_ids": evidence_ids,
            "rationale": rationale,
        }

    def _guard_next_decision(self, proposed: dict, applied: dict,
                             round_summary: dict) -> Tuple[dict, dict]:
        """Bound strategy jumps and force conservative recovery after regression.

        The LLM still selects the strategy inside the advertised ranges. This
        controller-side policy only prevents abrupt, self-reinforcing changes
        unsupported by the observed optimization trajectory.
        """
        guarded = dict(proposed)
        adjustments = []

        def clamp_step(key: str, max_step: float) -> None:
            old = float(applied[key])
            requested = float(guarded[key])
            bounded = min(old + max_step, max(old - max_step, requested))
            bounded = round(bounded, 6)
            if bounded != requested:
                adjustments.append({"field": key, "requested": requested,
                                    "applied": bounded, "reason": "step_limit"})
                guarded[key] = bounded

        clamp_step("temperature", 0.2)
        clamp_step("mutate_fraction", 0.15)

        best_gap = max(0.0, float(round_summary.get("best_hv", 0.0))
                       - float(round_summary.get("hv", 0.0)))
        severe_regression = (self._consecutive_hv_declines >= 2
                             or best_gap > HV_BEST_GAP_TOLERANCE)
        if severe_regression:
            recovery_caps = {
                "temperature": min(float(guarded["temperature"]),
                                   float(applied["temperature"])),
                "mutate_fraction": min(float(guarded["mutate_fraction"]), 0.5),
                "substitutions_per_candidate": min(
                    int(guarded["substitutions_per_candidate"]), 2
                ),
                "parent_focus": "balanced",
            }
            for key, value in recovery_caps.items():
                if guarded[key] != value:
                    adjustments.append({"field": key, "requested": guarded[key],
                                        "applied": value,
                                        "reason": "regression_recovery"})
                    guarded[key] = value

        return guarded, {"applied": bool(adjustments),
                         "adjustments": adjustments,
                         "consecutive_hv_declines": self._consecutive_hv_declines,
                         "best_hv_gap": round(best_gap, 6)}

    @staticmethod
    def _execution_summary(requested: dict, adjusted: dict, guardrail: dict) -> str:
        """Describe executable values only; keep free-form LLM prose in requested_decision."""
        summary = (
            "Next round executable decision: "
            f"temperature={adjusted['temperature']}, "
            f"mutate_fraction={adjusted['mutate_fraction']}, "
            f"substitutions_per_candidate={adjusted['substitutions_per_candidate']}, "
            f"parent_focus='{adjusted['parent_focus']}'."
        )
        changes = guardrail.get("adjustments", []) if isinstance(guardrail, dict) else []
        if not changes:
            return summary + " Central request accepted without controller adjustment."
        rendered = []
        for item in changes:
            rendered.append(
                f"{item.get('field')} {item.get('requested')}→{item.get('applied')} "
                f"({item.get('reason')})"
            )
        return summary + " Controller adjustments: " + "; ".join(rendered) + "."

    def _has_converged(self, round_num: int, feasible_count: int, hv: float) -> bool:
        """Require a sustained plateau near the best, not a post-crash blip."""
        return (round_num > MIN_ITERATION_ROUNDS
                and feasible_count > 0
                and max(0.0, self._best_hv - hv) <= HV_BEST_GAP_TOLERANCE
                and self._stable_rounds >= HV_CONVERGENCE_PATIENCE)

    def _build_round_summary(self, round_num: int, protein_seq: str,
                             host_organism: str, results: list,
                             pareto_solutions: list, hv: float,
                             hv_improvement: float, applied_decision: dict,
                             score_correlations: dict, generation_meta: dict) -> dict:
        """给专家共享同一份数值证据；不发送完整候选序列。"""
        feasible_count = sum(bool(r.is_feasible) for r in results)
        violations = {}
        for result in results:
            for violation in result.constraint_violations:
                name = violation.split("=", 1)[0].strip()
                violations[name] = violations.get(name, 0) + 1
        front = []
        for index, solution in enumerate(pareto_solutions):
            r = solution.result
            if solution.rank not in (0, 999):
                continue
            clean_len = len(r.sequence.replace(" ", ""))
            front.append({
                "candidate_id": index,
                "rank": solution.rank,
                "te": round(r.te_score, 6),
                "stability": round(r.stability_score, 6),
                "expression": round(r.expression_score, 6),
                "cai": round(r.cai, 6),
                "gc": round(r.gc_content, 6),
                "mfe_per_nt": round(r.mfe / clean_len, 6) if clean_len else None,
                "stem": r.max_stem_len,
                "homopolymer": r.max_homopolymer,
                "violations": list(r.constraint_violations),
            })
        effective_bounds = dict(self.constraint_bounds)
        inactive_bounds = {}
        if "mfe_per_nt_min" in effective_bounds and "mfe_per_nt_max" in effective_bounds:
            for key in ("mfe_min", "mfe_max"):
                if key in effective_bounds:
                    inactive_bounds[key] = effective_bounds.pop(key)
        return {
            "round": round_num,
            "next_round": round_num + 1,
            "host": host_organism,
            "protein_length_aa": len(protein_seq.rstrip("*")),
            "population_size": len(results),
            "unique_sequences": len({r.sequence.replace(" ", "").upper() for r in results}),
            "feasible_count": feasible_count,
            "feasible_rate": round(feasible_count / len(results), 6),
            "recovery_mode": feasible_count == 0,
            "constraint_violations": violations,
            "hv": round(hv, 6),
            "hv_delta": round(hv_improvement, 6),
            "previous_hv": round(self.prev_hv, 6),
            "best_hv": round(self._best_hv, 6),
            "best_hv_gap": round(max(0.0, self._best_hv - hv), 6),
            "best_round": self._best_round,
            "consecutive_hv_declines": self._consecutive_hv_declines,
            "stable_rounds": self._stable_rounds,
            "recent_rounds": [
                {
                    "round": item["round"],
                    "hv": item["hv"],
                    "hv_delta": item["hv_improvement"],
                    "feasible_rate": round(
                        item["feasible_count"] / max(1, item["num_candidates"]), 6
                    ),
                    "applied_decision": item.get("applied_decision", {}),
                }
                for item in self.history[-3:]
            ],
            "pareto_or_recovery_candidates": front,
            "score_correlations": score_correlations,
            "correlation_sample_size": len(results),
            "applied_decision": dict(applied_decision),
            "actual_new_count": generation_meta.get("new_count", 0),
            "actual_mutated_count": generation_meta.get("mutated_count", 0),
            "elite_parent_source": self._elite_source,
            "host_codon_table_available": bool(self.codon_table),
            "constraint_bounds": effective_bounds,
            "inactive_constraint_bounds": inactive_bounds,
            "constraint_modes": {
                "mfe": "per_nt" if "mfe_per_nt_min" in effective_bounds else "absolute"
            },
            "operator_semantics": {
                "mutate_fraction": "share replaced by edits of parent sequences; does not repair infeasible samples",
                "substitutions_per_candidate": "random synonymous positions, not homopolymer-targeted",
                "parent_focus": "biases selection toward parents weak on that objective; no guaranteed improvement",
            },
        }

    @staticmethod
    def _codon_edit_diff(before_sequence: str, after_sequence: str) -> List[Dict]:
        """Return the exact codon edits between two sequence-processing stages."""
        before = before_sequence.replace(" ", "").upper().replace("T", "U")
        after = after_sequence.replace(" ", "").upper().replace("T", "U")
        return [
            {"codon_index": pos // 3, "old_codon": before[pos:pos + 3],
             "new_codon": after[pos:pos + 3]}
            for pos in range(0, min(len(before), len(after)), 3)
            if before[pos:pos + 3] != after[pos:pos + 3]
        ]

    @staticmethod
    def _validate_candidate_translations(candidates: List[str], protein_seq: str) -> None:
        """防止编码错误或模型长度截断的序列进入评分。"""
        protein = protein_seq.strip().upper().rstrip("*")
        if not protein or not candidates:
            raise ValueError("target protein and candidate population must be non-empty")
        for index, sequence in enumerate(candidates):
            clean = sequence.replace(" ", "").upper().replace("T", "U")
            if len(clean) != (len(protein) + 1) * 3:
                raise ValueError(f"candidate {index} has incorrect CDS length")
            codons = [clean[i:i + 3] for i in range(0, len(clean), 3)]
            if CODON_TO_AA.get(codons[-1]) != "*":
                raise ValueError(f"candidate {index} has no terminal stop codon")
            translated = "".join(CODON_TO_AA.get(c, "?") for c in codons[:-1])
            if translated != protein:
                raise ValueError(f"candidate {index} does not encode the target protein")

    def _mutate_elites(self, protein_seq: str, num_needed: int,
                      focus_objective: str = "balanced",
                      substitutions_per_candidate: int = 2) -> Tuple[List[str], List[Dict]]:
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
            mutated_seq, mutation_record = self._synonymous_substitute(
                parent_seq, protein_seq, substitutions_per_candidate
            )
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
        跳过: CAI (需要密码子表)
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

        # MFE (ViennaRNA): 与正式 ConstraintChecker 使用同一量纲。
        mfe = self.evaluator._compute_mfe(sequence)
        if "mfe_per_nt_min" in bounds and "mfe_per_nt_max" in bounds:
            nt_length = len("".join(sequence.split()))
            if nt_length == 0:
                return False
            mfe_nt = mfe / nt_length
            if not bounds["mfe_per_nt_min"] <= mfe_nt <= bounds["mfe_per_nt_max"]:
                return False
        else:
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
    def _synonymous_substitute(seq: str, protein_seq: str,
                              substitutions_per_candidate: int = 2) -> Tuple[Optional[str], Dict]:
        """按中央决策做指定次数的同义替换，保持目标蛋白不变。

        Returns: (mutated_sequence_or_None, mutation_record)
          mutation_record = {"num_mutations": N, "mutations": [{"codon_index":..., "old_codon":..., "new_codon":..., "amino_acid":...}]}
        """
        mutation_record: Dict[str, Any] = {"num_mutations": 0, "mutations": []}

        # 清理序列: 去除空格, 转为 RNA (U 代替 T)
        clean = seq.replace(" ", "").upper().replace("T", "U")
        if len(clean) < 3:
            return None, mutation_record

        # 按密码子拆分
        codons = [clean[i:i+3] for i in range(0, len(clean) - (len(clean) % 3), 3)]
        protein = protein_seq.strip().upper().rstrip("*")
        if not codons or len(codons) != len(protein) + 1:
            return None, mutation_record
        if CODON_TO_AA.get(codons[-1]) != "*":
            return None, mutation_record
        if type(substitutions_per_candidate) is not int or not 1 <= substitutions_per_candidate <= 3:
            return None, mutation_record

        # 仅从与输入蛋白一致、且确有替代同义密码子的编码位点中选取。
        # 起始/终止密码子和无同义替代的 Met、Trp 不会被错误计入次数。
        eligible = []
        for pos, aa in enumerate(protein):
            if CODON_TO_AA.get(codons[pos]) != aa:
                return None, mutation_record
            if len(AA_TO_CODONS.get(aa, [])) > 1:
                eligible.append(pos)
        if len(eligible) < substitutions_per_candidate:
            return None, mutation_record
        positions = random.sample(eligible, substitutions_per_candidate)

        new_codons = list(codons)
        for pos in positions:
            old_codon = codons[pos]
            aa = CODON_TO_AA.get(old_codon)
            alternatives = [c for c in AA_TO_CODONS.get(aa, []) if c != old_codon]
            new_codon = random.choice(alternatives)
            new_codons[pos] = new_codon
            mutation_record["mutations"].append({
                "codon_index": pos,
                "old_codon": old_codon,
                "new_codon": new_codon,
                "amino_acid": aa,
            })

        mutation_record["num_mutations"] = len(mutation_record["mutations"])
        if mutation_record["num_mutations"] != substitutions_per_candidate:
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
        batch_id = len(gen_meta.setdefault("cai_batches", []))
        gen_meta["cai_batches"].append({
            "batch_id": batch_id,
            "source": source_tag,
            "candidate_count": len(candidates),
            "application_fraction": ratio,
            "intensity": intensity,
        })
        # 随机选择要优化的索引
        optimize_indices = set(random.sample(range(len(candidates)), min(num_to_optimize, len(candidates))))

        optimized = list(candidates)
        cai_count = 0
        for i in optimize_indices:
            optimized_seq = self._cai_optimize(candidates[i], intensity=intensity)
            if optimized_seq != candidates[i]:
                optimized[i] = optimized_seq
                cai_count += 1
                before = candidates[i].replace(" ", "").upper().replace("T", "U")
                after = optimized_seq.replace(" ", "").upper().replace("T", "U")
                edits = [
                    {"codon_index": pos // 3, "old_codon": before[pos:pos + 3],
                     "new_codon": after[pos:pos + 3]}
                    for pos in range(0, min(len(before), len(after)), 3)
                    if before[pos:pos + 3] != after[pos:pos + 3]
                ]
                gen_meta.setdefault("cai_edit_details", []).append({
                    "batch_id": batch_id,
                    "source": source_tag,
                    "candidate_index_in_batch": i,
                    "num_edits": len(edits),
                    "edits": edits,
                })

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
