"""LLM Feedback Analyzer: 分析 Pareto 结果 → 生成改进反馈

支持相关性驱动反馈: 每轮计算可解释指标
(CAI/GC/MFE per nt/Stem/Homo)
与三个微调模型分数 (TE/Stability/Expression) 的 Spearman 相关系数，
让 LLM 知道哪些可解释指标真正驱动了打分上涨。
"""

import json
import time
import numpy as np
from openai import OpenAI
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from pareto_selector import ParetoSolution

# scipy 用于 Spearman 相关 (延迟导入, 避免硬依赖)
_SCIPY_AVAILABLE = None


def _has_scipy():
    global _SCIPY_AVAILABLE
    if _SCIPY_AVAILABLE is None:
        try:
            import scipy.stats  # noqa: F401
            _SCIPY_AVAILABLE = True
        except ImportError:
            _SCIPY_AVAILABLE = False
    return _SCIPY_AVAILABLE


class FeedbackAnalyzer:
    """LLM 分析评估结果 → 生成下一轮改进反馈"""

    def __init__(self):
        self.client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        self._prev_round_scores = {}  # 上轮分数 (用于跨轮对比)
        self._feedback_history: list = []  # 历史反馈 (用于避免重复建议)

    def analyze(self, pareto_solutions: list, round_num: int,
                host_organism: str, constraint_bounds: dict,
                all_results: list = None) -> dict:
        """
        分析当前 Pareto 前沿, 生成改进反馈

        Args:
            pareto_solutions: Pareto 前沿解列表
            round_num: 当前轮次
            host_organism: 宿主物种名
            constraint_bounds: 约束边界
            all_results: 本轮全部候选的评估结果 (EvaluationResult 列表),
                        用于计算可解释指标与模型分数的相关性。
                        None 时退化为纯 Pareto 分析。

        Returns:
            {"feedback": str, "weaknesses": list, "suggestions": list,
             "overall_assessment": str, "score_correlations": dict}
        """
        front_summary = self._summarize_pareto_front(pareto_solutions)

        feasible = [s for s in pareto_solutions if s.result.is_feasible]
        infeasible = [s for s in pareto_solutions if not s.result.is_feasible]

        te_scores = [s.result.te_score for s in pareto_solutions]
        stab_scores = [s.result.stability_score for s in pareto_solutions]
        expr_scores = [s.result.expression_score for s in pareto_solutions]

        min_scores = {"TE": min(te_scores), "Stability": min(stab_scores), "Expression": min(expr_scores)}
        weakest_obj = min(min_scores, key=min_scores.get)

        violation_counts = {}
        for s in infeasible:
            for v in s.result.constraint_violations:
                key = v.split("=")[0] if "=" in v else v[:10]
                violation_counts[key] = violation_counts.get(key, 0) + 1

        # —— 计算可解释指标 ↔ 模型分数的相关性 ——
        correlation_insights = ""
        score_correlations = {}
        if all_results and len(all_results) >= 5:
            score_correlations = self._compute_score_correlations(all_results)
            correlation_insights = self._format_correlation_insights(score_correlations)

        # —— 构建历史反馈摘要 (避免重复建议) ——
        history_hint = ""
        if self._feedback_history:
            history_lines = []
            for i, fb in enumerate(self._feedback_history[-2:]):  # 最近2轮
                round_n = round_num - len(self._feedback_history) + i
                history_lines.append(f"  第{round_n}轮反馈: {fb[:150]}{'...' if len(fb) > 150 else ''}")
            history_hint = f"""
## 前几轮反馈历史 (⚠️ 避免重复!)
{chr(10).join(history_lines)}

**重要**: 上述反馈已经给过, 不要重复相同的建议。如果某个问题仍未改善, 
请提出**不同的**具体策略, 或者分析为什么之前的策略没有效果。
"""

        prompt = f"""你是一位 mRNA 设计优化专家。分析当前迭代结果并给出改进反馈。

## 背景
- 宿主物种: {host_organism}
- 迭代轮次: {round_num}
- 约束边界: {json.dumps(constraint_bounds, ensure_ascii=False)}

## 当前 Pareto 前沿 (TE/Stability/Expression 由 NUWA 微调模型打分)
{front_summary}

## 统计
- 可行解: {len(feasible)}/{len(pareto_solutions)}
- 最弱目标: {weakest_obj} (min={min_scores[weakest_obj]:.3f})
- 目标范围: TE [{min(te_scores):.3f}, {max(te_scores):.3f}], Stability [{min(stab_scores):.3f}, {max(stab_scores):.3f}], Expression [{min(expr_scores):.3f}, {max(expr_scores):.3f}]
- 约束违反: {json.dumps(violation_counts, ensure_ascii=False)}

## 打分驱动因子分析 (Spearman 相关系数)
{correlation_insights if correlation_insights else "(无相关性数据)"}

⚠️ 重要: 上表告诉你哪些可解释指标与微调模型打分正相关/负相关。
例如 "TE ~ CAI: r=+0.62" 意味着提高 CAI 很可能提高 TE 打分。
"Stability ~ MFE: r=-0.45" 意味着 MFE 更负 (结构更稳定) 反而可能降低 Stability 打分。
{history_hint}
## 你的任务
1. 结合相关性数据，识别当前候选的薄弱环节
2. 给出**具体且可操作**的改进建议 — 优先推荐与目标分数正相关的操作，避开负相关的操作
3. 聚焦最弱目标的同时保持其他目标
4. 如果某指标在前几轮已建议改善但未改善，分析原因并提出新策略

## 输出格式 (严格 JSON)
```json
{{
  "weaknesses": ["薄弱环节1", "薄弱环节2"],
  "suggestions": ["改进建议1", "改进建议2"],
  "feedback": "对下一轮生成的简明改进指导。优先引用相关性数据指导方向。",
  "overall_assessment": "简要评价",
  "generation_directives": {{
    "focus_objective": "stability",
    "cai_intensity": 0.35,
    "mutate_fraction": 0.6,
    "explore": 0.3
  }}
}}
```
其中 generation_directives 是给控制器的机器指令 (人类无需阅读, 仅用于调节下一轮生成):
- focus_objective: 下一轮重点优化目标 ("te" | "stability" | "expression" | "balanced")
- cai_intensity: 密码子优化强度 0.1–0.5 (越大越激进替换为宿主偏好密码子)
- mutate_fraction: 精英突变占本轮候选比例 0.0–1.0 (越大越利用已知好序列)
- explore: 探索度 0.0–1.0 (越大温度越高、越广撒网; 越小越利用)"""

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                timeout_val = 300 + attempt * 60
                if attempt > 0:
                    print(f"  [Feedback] 🔄 第 {attempt + 1}/{max_retries + 1} 次尝试...")

                response = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": "你是一位 mRNA 设计优化专家。给出具体可操作的反馈。始终以纯 JSON 格式输出。"},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                    timeout=timeout_val,
                )

                raw_content = response.choices[0].message.content.strip()
                content = raw_content
                # 递归剥离多层 markdown 代码块标记
                # LLM 可能输出 ```json\n```json\n...\n```\n``` 嵌套结构
                while True:
                    if content.startswith("```json"):
                        content = content[7:]
                    elif content.startswith("```"):
                        content = content[3:]
                    else:
                        break
                    # 去除末尾的 ```
                    end_idx = content.rfind("```")
                    if end_idx >= 0:
                        content = content[:end_idx]
                    content = content.strip()
                    # 如果剥离后还以 ``` 开头，继续循环
                    if not (content.startswith("```json") or content.startswith("```")):
                        break

                result = json.loads(content)
                result["raw_response"] = raw_content
                # 2026-08-24: 保存发送给 LLM 的 prompt, 否则反馈环节提示词无法复现/审计
                result["prompt"] = prompt
                result["score_correlations"] = score_correlations
                result["generation_directives"] = result.get("generation_directives", {}) or {}
                # 记录到历史 (用于下一轮避免重复建议)
                self._feedback_history.append(result.get("feedback", ""))
                print(f"  [Feedback] ✅ LLM 调用成功 ({len(raw_content)} 字符)")
                return result

            except (json.JSONDecodeError, Exception) as e:
                error_str = str(e).lower()
                is_timeout = any(kw in error_str for kw in
                    ["timeout", "timed out", "timed_out", "connection", "reset"])
                if attempt < max_retries:
                    if is_timeout:
                        print(f"  [Feedback] ⚠️ 超时, 立即重试...")
                    else:
                        time.sleep(2 ** attempt)
                        print(f"  [Feedback] ⚠️ {type(e).__name__}, 重试...")
                else:
                    print(f"  [Feedback] ❌ 全部尝试失败: {e}")

        # 所有重试耗尽 → fallback
        fallback_result = {
            "weaknesses": [f"{weakest_obj} 是最弱目标"],
            "suggestions": [f"聚焦改进 {weakest_obj}"],
            "feedback": f"改进 {weakest_obj} 同时保持其他目标。当前 min {weakest_obj}={min_scores[weakest_obj]:.3f}。",
            "overall_assessment": f"第 {round_num} 轮, {len(feasible)} 个可行解。",
            "raw_response": "LLM call failed after retries",
            "prompt": prompt,
            "score_correlations": score_correlations,
            "generation_directives": {"focus_objective": "balanced", "cai_intensity": 0.25,
                                     "mutate_fraction": 0.5, "explore": 1.0},
        }
        self._feedback_history.append(fallback_result.get("feedback", ""))
        return fallback_result

    @staticmethod
    def _summarize_pareto_front(solutions: list) -> str:
        lines = []
        for i, s in enumerate(solutions[:5]):
            r = s.result
            lines.append(
                f"  #{i+1} (rank={s.rank}): "
                f"TE={r.te_score:.3f}, Stab={r.stability_score:.3f}, "
                f"Expr={r.expression_score:.3f}, "
                f"CAI={r.cai:.3f}, GC={r.gc_content:.1%}, "
                f"MFE={r.mfe:.1f}, "
                f"Feasible={r.is_feasible}"
            )
        if len(solutions) > 5:
            lines.append(f"  ... 以及 {len(solutions) - 5} 个更多解")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    #  相关性分析: 可解释指标 ↔ 微调模型分数
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_score_correlations(all_results: list) -> dict:
        """计算可解释指标与微调模型打分之间的 Spearman 相关系数。

        可解释指标 (LLM 能理解并操作的):
          - CAI (密码子适应指数)
          - GC% (GC 含量)
          - MFE/nt (长度归一化最小自由能)
          - Stem (最大茎区长度)
          - Homo (最大同聚物长度)

        模型打分 (黑盒, LLM 不知道内部逻辑):
          - TE (翻译效率, finetuned_model_TE)
          - Stability (稳定性, finetuned_model_fungal)
          - Expression (表达量, finetuned_model_fungal_euk)

        Returns:
            {
                "TE": {"cai": 0.35, "gc": 0.12, "mfe_nt": -0.08, ...},
                "Stability": {"cai": 0.21, "gc": 0.45, "mfe_nt": -0.72, ...},
                "Expression": {"cai": 0.52, "gc": 0.18, "mfe_nt": -0.33, ...},
            }
        """
        if not all_results or len(all_results) < 5:
            return {}

        # 提取可解释指标向量
        cai_vals = np.array([r.cai for r in all_results])
        gc_vals = np.array([r.gc_content for r in all_results])
        mfe_vals = np.array([
            r.mfe / max(1, len(r.sequence.replace(" ", "")))
            for r in all_results
        ])
        stem_vals = np.array([r.max_stem_len for r in all_results], dtype=float)
        homo_vals = np.array([r.max_homopolymer for r in all_results], dtype=float)

        # 提取模型打分向量
        te_vals = np.array([r.te_score for r in all_results])
        stab_vals = np.array([r.stability_score for r in all_results])
        expr_vals = np.array([r.expression_score for r in all_results])

        interpretable = {
            "cai": cai_vals,
            "gc": gc_vals,
            "mfe_nt": mfe_vals,
            "stem": stem_vals,
            "homo": homo_vals,
        }

        score_targets = {
            "TE": te_vals,
            "Stability": stab_vals,
            "Expression": expr_vals,
        }

        # 计算真正的 Spearman：即使 scipy 不可用，也对秩而非原值求相关。
        if _has_scipy():
            from scipy.stats import spearmanr
            corr_fn = lambda x, y: spearmanr(x, y)[0]
        else:
            def rankdata(values):
                order = np.argsort(values, kind="mergesort")
                ranks = np.empty(len(values), dtype=float)
                start = 0
                while start < len(values):
                    end = start + 1
                    while end < len(values) and values[order[end]] == values[order[start]]:
                        end += 1
                    ranks[order[start:end]] = (start + end - 1) / 2.0
                    start = end
                return ranks

            corr_fn = lambda x, y: float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])

        correlations = {}
        for score_name, score_vec in score_targets.items():
            correlations[score_name] = {}
            for metric_name, metric_vec in interpretable.items():
                # 跳过常数向量 (如所有 Homo 都是 4)
                if np.std(metric_vec) < 1e-9 or np.std(score_vec) < 1e-9:
                    correlations[score_name][metric_name] = 0.0
                else:
                    value = float(corr_fn(metric_vec, score_vec))
                    correlations[score_name][metric_name] = (
                        round(value, 4) if np.isfinite(value) else 0.0
                    )

        return correlations

    @staticmethod
    def _format_correlation_insights(correlations: dict) -> str:
        """将相关系数字典格式化为 LLM 可读的文本。

        输出示例:
          ## 打分驱动因子 (Spearman r)
          | 模型目标    | CAI   | GC%   | MFE/nt | Stem  | Homo  |
          |------------|-------|-------|--------|-------|-------|
          | TE         | +0.35 | +0.12 | -0.08  | +0.01 | 0.00  |
          | Stability  | +0.21 | +0.45 | -0.72  | -0.15 | -0.02 |
          | Expression | +0.52 | +0.18 | -0.33  | -0.08 | -0.01 |

          解读:
          - TE 主要受 CAI 正向驱动 (r=+0.35)，提升 CAI 可能提升 TE 打分
          - Stability 与 MFE 强负相关 (r=-0.72) → MFE 更负 (结构更稳定) = Stability 打分更低
          - Expression 与 CAI 正相关 (r=+0.52)，是主要驱动力
        """
        if not correlations:
            return "(无相关性数据)"

        # 取第一个目标的所有指标名 (列名)
        first_target = next(iter(correlations.values()))
        metric_names = list(first_target.keys())

        lines = []
        lines.append("## 打分驱动因子 (Spearman r)")
        lines.append("")

        # 表头
        header = "| 模型目标    |"
        sep = "|------------|"
        for m in metric_names:
            header += f" {m:^7s} |"
            sep += "---------|"
        lines.append(header)
        lines.append(sep)

        # 数据行
        for score_name, corrs in correlations.items():
            row = f"| {score_name:<11s} |"
            for m in metric_names:
                val = corrs.get(m, 0.0)
                if val >= 0:
                    row += f" +{val:<6.2f} |"
                else:
                    row += f" {val:<7.2f} |"
            lines.append(row)

        lines.append("")
        lines.append("解读 (按 r 绝对值 >0.2 视为显著):")
        lines.append("")

        for score_name, corrs in correlations.items():
            significant = [(m, v) for m, v in corrs.items() if abs(v) > 0.2]
            if significant:
                significant.sort(key=lambda x: abs(x[1]), reverse=True)
                parts = []
                for m, v in significant:
                    direction = "正向" if v > 0 else "负向"
                    parts.append(f"{m} ({direction}, r={v:+.2f})")
                lines.append(
                    f"- **{score_name}**: {', '.join(parts)}"
                )
            else:
                lines.append(f"- **{score_name}**: 无可解释指标显著相关 (|r|<0.2)")

        return "\n".join(lines)
