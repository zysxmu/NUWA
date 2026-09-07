"""NUWA-Agent — Entry Point

完整流程:
  1. SpeciesResolver (内嵌于 Orchestrator): 物种名 → (domain, class_id, confidence)
  2. Orchestrator: LLM CoT 三层决策 → 选模型 + class_id + 约束 (详细推理链 400+ 字)
  3. IterationController: 生成→评估→Pareto→反馈→迭代 (最多10轮, 最少5轮)
  4. 输出: Pareto最优序列 + 好序列队列 + 完整推理链 + 全部迭代历史

温度衰减: 前40%轮次内从1.0线性衰减到0.3, 之后保持0.3
收敛: 0 ≤ ΔHV < 1% 且 round > 5 → 停止
"""

import json
import os
from datetime import datetime
from orchestrator import OrchestratorAgent
from evaluator import MultiObjectiveEvaluator
from iteration_controller import IterationController
from config import OUTPUT_DIR, CODON_TABLES_DIR, HV_CONVERGENCE_THRESHOLD, MAX_ITERATION_ROUNDS, MIN_ITERATION_ROUNDS


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


def main():
    print("=" * 60)
    print("  NUWA-Agent — CoT 三层决策选模型 + 迭代多目标优化")
    print("  Phase 1: SpeciesResolver + CoT 三层决策 → 选模型 + class_id + 约束")
    print(f"  Phase 2: 迭代优化 (生成→评估→Pareto→反馈) — 最多{MAX_ITERATION_ROUNDS}轮")
    print("  Phase 3: Pareto 最优输出 + 好序列队列")
    print("=" * 60)

    # ====== 输入 ======
    protein_seq = input("\n请输入目标蛋白序列: ").strip()
    host_organism = input("请输入宿主物种 (如 Escherichia coli): ").strip()

    user_constraints = {}
    gc_input = input("GC 含量约束 (如 0.3-0.7, 回车跳过): ").strip()
    if gc_input:
        try:
            gc_min, gc_max = gc_input.split("-")
            user_constraints = {"gc_min": float(gc_min), "gc_max": float(gc_max)}
        except ValueError:
            print("GC 约束格式无效，跳过。")

    # ====== Phase 1: SpeciesResolver + CoT 三层决策 ======
    print("\n" + "=" * 60)
    print("[Phase 1] SpeciesResolver + Orchestrator Agent — CoT 三层决策...")

    agent = OrchestratorAgent()
    decision = agent.run(protein_seq, host_organism, user_constraints or None)

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
          f"收敛阈值 ΔHV < {HV_CONVERGENCE_THRESHOLD})")
    # 加载宿主参考密码子表
    codon_table = load_codon_table(host_organism)

    # 创建评估器 (外部生物信息学工具 + 微调 NUWA 回归模型)
    # domain 和 class_id 传入以匹配微调模型的 token_type_ids
    evaluator = MultiObjectiveEvaluator(
        codon_table=codon_table,
        host_organism=host_organism,
        use_mhcflurry=True,     # MHCflurry (免疫原性)
        domain=selected_model,  # "bacteria" / "eukaryote" / "archaea"
        class_id=class_id,      # 来自 SpeciesResolver + CoT 决策
    )

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
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
        "optimization_result": {
            "total_rounds": result["total_rounds"],
            "converged": result["converged"],
            "final_hv": result["final_hv"],
            "final_temperature": max(0.3, 1.0 - 0.7 * (result["total_rounds"] - 1) / max(1, int(MAX_ITERATION_ROUNDS * 0.4)))
                               if result["total_rounds"] > 0 else 1.0,
        },
        # 每轮收集的好序列队列 (可行 Pareto 前沿, 去重)
        "good_sequences_queue": result.get("good_sequences_queue", []),
        # 完整序列, 不截断
        "pareto_solutions": [
            {
                "rank": s.rank,
                "crowding_distance": round(s.crowding_distance, 6),
                "sequence": s.result.sequence,
                "sequence_length": len(s.result.sequence.replace(" ", "")),
                "objectives": {
                    "TE": round(s.result.te_score, 6),
                    "Stability": round(s.result.stability_score, 6),
                    "Expression": round(s.result.expression_score, 6),
                },
                "constraints": {
                    "CAI": round(s.result.cai, 6),
                    "GC%": round(s.result.gc_content, 6),
                    "MFE": round(s.result.mfe, 3),
                    "max_stem_len": s.result.max_stem_len,
                    "max_homopolymer": s.result.max_homopolymer,
                    "immunogenicity_risk": round(s.result.immunogenicity_risk, 6),
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
                "feedback_raw": h["feedback_raw"],
                # 生成元数据
                "generation_meta": h["generation_meta"],
                # 本轮全部候选打分
                "candidate_scores": h["candidate_scores"],
                # 本轮 Pareto 前沿
                "pareto_front_details": h["pareto_front_details"],
                # 精英序列
                "elite_sequences": h["elite_sequences"],
            }
            for h in result["history"]
        ],
        "full_llm_log": decision.get("full_log", []),
    }

    output_path = os.path.join(OUTPUT_DIR, f"nuwa_agent_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ====== 保存推理链 Markdown ======
    _save_chain_markdown(decision, result, output, timestamp)

    # 摘要
    print(f"\nModel: {decision['selected_model']} (class_id={class_id}, confidence={class_id_confidence})")
    print(f"Rounds: {result['total_rounds']}, Converged: {result['converged']}")
    print(f"Final HV: {result['final_hv']:.4f}")
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
    print("=" * 60)


def _save_chain_markdown(decision: dict, result: dict, output: dict, timestamp: str):
    """保存完整推理链 + 优化过程为可读 Markdown"""
    md_path = os.path.join(OUTPUT_DIR, f"nuwa_agent_chain_{timestamp}.md")

    species_info = decision.get("species_info", {})

    lines = [
        f"# NUWA-Agent 完整推理链 & 优化过程报告",
        f"",
        f"**时间**: {timestamp}",
        f"**选定模型**: {decision.get('selected_model', 'N/A')} ({decision.get('model_name', '')})",
        f"**Class ID**: {decision.get('class_id', 'N/A')} (confidence: {decision.get('class_id_confidence', 'N/A')})",
        f"**选模理由**: {decision.get('rationale', '')}",
        f"**总轮数**: {result['total_rounds']}, **收敛**: {result['converged']}, **最终HV**: {result['final_hv']:.6f}",
        f"",
        f"---",
        f"",
        f"# Phase 0: SpeciesResolver — 物种解析",
        f"",
    ]

    # SpeciesResolver 结果
    if species_info:
        lines.append(f"- **物种匹配**: {species_info.get('matched_name', 'N/A')}")
        lines.append(f"- **域**: {species_info.get('domain', 'N/A')}")
        lines.append(f"- **置信度**: {species_info.get('confidence', 'N/A')}")
        lines.append(f"- **是否解析成功**: {species_info.get('is_resolved', False)}")
        lines.append(f"- **理由**: {species_info.get('reason', '')}")

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
    lines.append(f"```json")
    lines.append(json.dumps(decision.get("constraint_bounds", {}), ensure_ascii=False, indent=2))
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
        lines.append(f"## Round {round_num} (温度: {h.get('temperature', 1.0):.1f})")
        lines.append("")
        lines.append(f"- **候选总数**: {h['num_candidates']}")
        lines.append(f"- **可行解**: {h['feasible_count']}/{h['num_candidates']}")
        lines.append(f"- **Pareto 前沿大小**: {h['pareto_front_size']}")
        lines.append(f"- **HV**: {h['hv']:.6f} (Δ{h['hv_improvement']:+.6f})")
        lines.append("")

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
            lines.append("| Rank | TE | Stability | Expression | CAI | GC% | MFE | Stem | Homo | Immuno | Feasible |")
            lines.append("|------|-----|-----------|------------|-----|-----|-----|------|------|--------|----------|")
            for pf in pareto_front:
                o = pf["objectives"]
                c = pf["constraints"]
                lines.append(f"| {pf['rank']} | {o['TE']:.4f} | {o['Stability']:.4f} | {o['Expression']:.4f} "
                            f"| {c['CAI']:.4f} | {c['GC%']:.1%} | {c['MFE']:.1f} "
                            f"| {c['max_stem_len']} | {c['max_homopolymer']} "
                            f"| {c['immunogenicity_risk']:.4f} | {pf['is_feasible']} |")
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
        if feedback:
            lines.append(f"### LLM 优化反馈")
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
        lines.append(f"- **immunogenicity_risk**: {c['immunogenicity_risk']:.6f}")
        lines.append(f"- **Feasible**: {s['is_feasible']}")
        lines.append("")
        lines.append(f"### 完整序列")
        lines.append(f"```")
        lines.append(s['sequence'])
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
        lines.append("| # | 轮次 | TE | Stability | Expression | CAI | GC% | MFE | Stem | Homo | Immuno |")
        lines.append("|---|------|-----|-----------|------------|-----|-----|-----|------|------|--------|")
        for i, gq in enumerate(good_queue):
            lines.append(f"| {i+1} | R{gq['round']} | {gq['te_score']:.4f} | {gq['stability_score']:.4f} "
                        f"| {gq['expression_score']:.4f} | {gq['cai']:.4f} | {gq['gc_content']:.1%} "
                        f"| {gq['mfe']:.1f} | {gq['max_stem_len']} | {gq['max_homopolymer']} "
                        f"| {gq['immunogenicity_risk']:.4f} |")
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
                lines.append(gq['sequence'])
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
    lines.append(f"- **最终 HV**: {result['final_hv']:.6f}")
    lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
