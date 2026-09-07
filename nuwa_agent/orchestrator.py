"""Orchestrator Agent: SpeciesResolver + LLM CoT 三层决策 → 选模型 + class_id + 制定约束

三层决策:
  1. 域级: 该物种属于哪个域?
  2. 物种级: 精确匹配直接用; 找不到 → LLM 讨论候选项 (近亲物种); 全无 → class_id=0
  3. 跨域级: 是否用其他域模型更好? (极端GC/嗜热等场景)

核心: 当 SpeciesResolver 找不到精确匹配时，把候选项列表交给 LLM，
     让 LLM 用领域知识讨论「该用哪个近亲的 class_id」还是「class_id=0 更安全」。
"""

import json
import time
from typing import Optional, List
from openai import OpenAI
from model_registry import registry
from species_resolver import SpeciesResolver, SpeciesInfo, CandidateSpecies
from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS,
    NUWA_MODELS, COT_VERBOSE, DEFAULT_CONSTRAINTS,
)


class OrchestratorAgent:
    """Orchestrator: SpeciesResolver + LLM CoT 三层决策 → 选模型 + class_id + 制定约束"""

    def __init__(self):
        self.client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        self.resolver = SpeciesResolver()
        self.full_log = []

    def run(self, protein_seq: str, host_organism: str,
            user_constraints: dict = None) -> dict:
        """
        Phase 1: SpeciesResolver + LLM CoT 三层决策

        Returns:
            {
                "selected_model": "eukaryote",
                "model_name": "NUWA-Eukaryote",
                "class_id": 1234,
                "class_id_confidence": "exact",
                "reasoning_chain": "...",
                "rationale": "...",
                "constraint_bounds": { ... },
                "host_analysis": { ... },
                "model_discussion": { ... },
                "species_info": { ... },
                "chosen_candidate": { ... } | None,   # LLM 选的近亲物种
            }
        """
        self.full_log = []

        constraints = {**DEFAULT_CONSTRAINTS}
        if user_constraints:
            constraints.update(user_constraints)

        # Step 1: SpeciesResolver 查找
        species_info = self.resolver.resolve(host_organism)
        print(f"\n[Orchestrator] Species resolved: domain={species_info.domain}, "
              f"class_id={species_info.class_id}, confidence={species_info.confidence}")
        if species_info.is_resolved:
            print(f"  ✅ Exact match: {species_info.matched_name}")
        elif species_info.needs_cot_discussion:
            print(f"  ⚠️  Not found exactly → {len(species_info.candidates)} candidates for CoT discussion")
            for c in species_info.candidates[:3]:
                print(f"     - {c.name} (domain={c.domain}, class_id={c.class_id})")
        else:
            print(f"  ❌ Not found at all → LLM must determine domain, class_id=0")
        print(f"  Reason: {species_info.reason}")

        # Step 2: LLM CoT 三层决策
        print("\n[CoT] ====== 三层决策推理 ======")
        result = self._cot_three_level_decision(
            protein_seq, host_organism, species_info, constraints
        )
        print("[CoT] ====== 三层决策完成 ======\n")

        # 安全检查
        selected_model = result.get("selected_model", "")
        if selected_model not in NUWA_MODELS:
            selected_model = self._fallback_domain_match(host_organism)
            result["selected_model"] = selected_model

        # 确保 class_id 信息存在
        if "class_id" not in result or result["class_id"] is None:
            result["class_id"] = species_info.class_id
        if "class_id_confidence" not in result:
            result["class_id_confidence"] = species_info.confidence

        # 用 LLM 建议覆盖默认约束
        if "constraint_bounds" in result:
            for k, v in result["constraint_bounds"].items():
                if k in constraints and v is not None:
                    constraints[k] = v

        # 2026-08-22: 同聚物阈值钳制 — 无论 LLM 提议什么, 强制 >= 10
        # (此前 LLM 常提议 4, 而真实序列同聚物多为 4-9, 阈值 4 会让绝大多数候选不可行)
        if constraints.get("max_homopolymer", 10) < 10:
            constraints["max_homopolymer"] = 10

        result["constraint_bounds"] = constraints
        result["model_name"] = NUWA_MODELS[selected_model]["name"]
        result["species_info"] = {
            "domain": species_info.domain,
            "class_id": species_info.class_id,
            "confidence": species_info.confidence,
            "matched_name": species_info.matched_name,
            "reason": species_info.reason,
            "is_resolved": species_info.is_resolved,
        }

        return result

    def _cot_three_level_decision(self, protein_seq: str, host_organism: str,
                                   species_info: SpeciesInfo,
                                   constraints: dict) -> dict:
        """LLM Chain-of-Thought 三层决策: 域级 → 物种级(讨论近亲) → 跨域级"""

        # 构建模型信息
        models_summary = ""
        for key in registry.list_models():
            info = registry.get_model_info(key)
            models_summary += f"""
### {info['name']} (key: {info['key']})
- Domain: {info['domain']}
- Supported species: {NUWA_MODELS[key].get('num_species', '?')}
- Description: {info['description']}
- Strengths: {', '.join(info['strengths'])}
- Limitations: {', '.join(info['limitations'])}
"""

        protein_len = len(protein_seq)
        defaults_str = json.dumps(constraints, ensure_ascii=False)

        # ---- 构建 SpeciesResolver 结果 prompt (含候选项) ----
        species_hint = self._build_species_hint(host_organism, species_info)

        prompt = f"""You are an mRNA design expert. Select the optimal NUWA model, determine class_id, and set constraint bounds for the following target.

## Task
- Target protein: {protein_seq[:80]}{'...' if len(protein_seq) > 80 else ''} (length: {protein_len} aa)
- Host organism: {host_organism}
- Default constraints: {defaults_str}

{species_hint}

## Available Models
{models_summary}

## Decision Process

### Level 1: DOMAIN — Which domain does this organism belong to?
Verify domain classification with brief taxonomic evidence. If relatives found, explain why they support this domain. Discuss GC content range and codon preferences for this domain.

### Level 2: SPECIES — What class_id to use?
If exact match: confirm and note key codon usage traits. If relatives found: briefly compare candidates, pick the closest relative (or class_id=0), justify with taxonomy. If nothing found: class_id=0, note implications.

### Level 3: CROSS-DOMAIN — Should we use a different domain model?
Check: extreme GC? Extremophile? Any reason to prefer another domain? Even if no, briefly explain why staying in-domain is correct.

## Output (strict JSON, keep each text field to 2-3 sentences)
```json
{{
  "host_analysis": {{
    "domain": "Eukaryote/Bacteria/Archaea",
    "codon_bias": "2-3 sentences on codon usage patterns.",
    "gc_tendency": "2-3 sentences on GC content and GC3.",
    "special_features": "2-3 sentences on notable features."
  }},
  "level1_domain": {{
    "domain": "bacteria/eukaryote/archaea",
    "confidence": "high/medium/low",
    "reason": "2-3 sentences of taxonomic justification."
  }},
  "level2_species": {{
    "class_id": <int>,
    "class_id_confidence": "exact/genus_match/fuzzy_match/default",
    "reason": "2-3 sentences explaining class_id choice."
  }},
  "chosen_candidate": {{
    "name": "species name or null",
    "class_id": <int or null>,
    "domain": "string or null",
    "why": "2-3 sentences justifying this candidate, or null if not applicable."
  }},
  "level3_cross_domain": {{
    "consider_cross_domain": true/false,
    "alternative_model": "bacteria/eukaryote/archaea/null",
    "reason": "2-3 sentences of cross-domain analysis."
  }},
  "model_discussion": {{
    "bacteria": {{"suitability": "High/Medium/Low", "reason": "1-2 sentences.", "risk": "1 sentence."}},
    "eukaryote": {{"suitability": "High/Medium/Low", "reason": "1-2 sentences.", "risk": "1 sentence."}},
    "archaea": {{"suitability": "High/Medium/Low", "reason": "1-2 sentences.", "risk": "1 sentence."}}
  }},
  "selected_model": "bacteria/eukaryote/archaea",
  "class_id": <int>,
  "class_id_confidence": "exact/genus_match/fuzzy_match/default",
  "rationale": "3-4 sentences synthesizing the decision.",
  "reasoning_chain": "150-200 words covering Level 1, 2, and 3 in sequence.",
  "constraint_bounds": {{
    "cai_min": null,
    "gc_min": null,
    "gc_max": null,
    "mfe_max": null,
    "mfe_min": null,
    "max_stem_length": null,
    "max_homopolymer": 10,
    "safety_threshold": null
  }}
}}
```

IMPORTANT: Keep text fields CONCISE (2-3 sentences each). Output ONLY valid JSON. Do not write long essays.
NOTE: For constraint_bounds, set max_homopolymer to 10 (nt) — do not lower it below 10. Leave other fields null to accept the defaults unless the host biology clearly demands tighter bounds."""

        result = self._call_llm(
            prompt,
            system_role="You are an mRNA design expert specializing in model selection, species resolution, and constraint formulation. Always output valid JSON.",
            step_name="CoT_三层决策",
            fallback=self._build_decision_fallback(host_organism, species_info),
        )

        return result

    def _build_species_hint(self, host_organism: str,
                             species_info: SpeciesInfo) -> str:
        """根据 SpeciesResolver 结果构建 CoT 提示文本"""

        if species_info.confidence == "exact":
            return f"""
SPECIES RESOLUTION: ✅ EXACT MATCH
- "{species_info.matched_name}" found in {species_info.domain} mapping table
- class_id = {species_info.class_id}
- Trust this result directly — no discussion needed.
"""

        elif species_info.confidence in ("genus_match", "fuzzy_match") \
                and species_info.candidates:
            # ---- 候选项列表，交给 LLM 讨论 ----
            match_type = "SAME GENUS" if species_info.confidence == "genus_match" else "FUZZY SIMILARITY"
            candidates_str = "\n".join(
                f"  {i+1}. {c.name} — domain={c.domain}, class_id={c.class_id}"
                for i, c in enumerate(species_info.candidates)
            )
            return f"""
SPECIES RESOLUTION: ⚠️ NOT IN DATABASE — {match_type} RELATIVES FOUND
- "{host_organism}" was NOT found exactly in any species mapping table.
- However, the following RELATED species ARE in the database:

CANDIDATE RELATIVES (in {species_info.domain}):
{candidates_str}

- Total {len(species_info.candidates)} candidates returned ({match_type}).
- **YOUR TASK**: In Level 2, discuss these candidates. Which one is the CLOSEST relative 
  to "{host_organism}"? Consider taxonomy, ecology, codon usage patterns.
  - If a candidate is clearly the best match → choose its class_id
  - If no candidate is close enough → set class_id=0 (no species signal) and explain why
  - If multiple are equally good → pick the most well-studied one and note the ambiguity
"""

        else:  # default
            return f"""
SPECIES RESOLUTION: ❌ NOT FOUND AT ALL
- "{host_organism}" was not found in any mapping table, and no close relatives were found.
- Suggested fallback: domain=unknown, class_id=0
- YOU MUST determine the domain based on your biological knowledge.
- class_id=0 means "no specific species signal" — the model will use general codon patterns.
- Consider: is this organism's codon usage similar to any well-known species?
- If you can think of a known relative, suggest class_id=0 but note the potential relative.
"""

    @staticmethod
    def _fallback_domain_match(host_organism: str) -> str:
        """降级: 关键词域匹配 (仅在 SpeciesResolver 和 LLM 都失败时使用)"""
        eukaryote_kw = ["human", "homo", "mouse", "mus", "rat", "rattus",
                         "yeast", "saccharomyces", "drosophila", "arabidopsis",
                         "人", "小鼠", "大鼠", "酵母", "果蝇", "拟南芥"]
        bacteria_kw = ["ecoli", "escherichia", "bacillus", "pseudomonas",
                        "staphylococcus", "大肠杆菌", "枯草", "假单胞"]
        archaea_kw = ["halo", "thermo", "sulfo", "methano", "pyro",
                       "嗜盐", "嗜热", "硫化", "产甲烷"]

        host_lower = host_organism.lower()
        for kw in eukaryote_kw:
            if kw in host_lower:
                return "eukaryote"
        for kw in bacteria_kw:
            if kw in host_lower:
                return "bacteria"
        for kw in archaea_kw:
            if kw in host_lower:
                return "archaea"
        return "eukaryote"

    def _build_decision_fallback(self, host_organism: str,
                                  species_info: SpeciesInfo) -> dict:
        """构建 LLM 解析失败时的降级决策"""
        fallback_model = self._fallback_domain_match(host_organism)
        if species_info.domain != "unknown":
            fallback_model = species_info.domain
        return {
            "host_analysis": {"domain": species_info.domain},
            "level1_domain": {"domain": fallback_model, "confidence": "low", "reason": "LLM parse failed, fallback"},
            "level2_species": {"class_id": species_info.class_id, "class_id_confidence": species_info.confidence, "reason": "LLM parse failed, fallback"},
            "chosen_candidate": None,
            "level3_cross_domain": {"consider_cross_domain": False, "alternative_model": None, "reason": "Fallback"},
            "model_discussion": {},
            "selected_model": fallback_model,
            "class_id": species_info.class_id,
            "class_id_confidence": species_info.confidence,
            "rationale": "LLM parse failed, using SpeciesResolver fallback",
            "reasoning_chain": f"Fallback: SpeciesResolver → {species_info.domain} (class_id={species_info.class_id}, {species_info.confidence})",
            "constraint_bounds": {},
        }

    # ------------------------------------------------------------------
    #  工具方法
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, system_role: str = "mRNA design expert",
                   fallback: dict = None, step_name: str = "",
                   max_retries: int = 3) -> dict:
        """调用 LLM API，含自动重试和指数退避。

        重试策略:
          - 首次 timeout=300s, 随后每次 +60s (300→360→420→480)
          - timeout 错误: 立即重试 (不等待)
          - 其他错误: 指数退避 1s→2s→4s→8s
          - 空内容检测: 如果 content 为空但 token 被消耗, 直接降级
          - JSON 模式: 使用 response_format 强制 JSON 输出
          - 最多 max_retries 次重试 (含首次共 max_retries+1 次)
        """
        raw_content = ""
        response = None
        use_json_mode = True  # 首次尝试 JSON 模式, 失败则降级

        for attempt in range(max_retries + 1):
            try:
                timeout_val = 300 + attempt * 60
                prefix = "  [CoT]" if attempt == 0 else f"  [CoT] 🔄 第 {attempt + 1}/{max_retries + 1} 次尝试"
                print(f"{prefix} 调用 LLM (timeout={timeout_val}s)...")

                api_kwargs = {
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": system_role},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": LLM_TEMPERATURE,
                    "max_tokens": LLM_MAX_TOKENS,
                    "timeout": timeout_val,
                }
                if use_json_mode:
                    api_kwargs["response_format"] = {"type": "json_object"}

                response = self.client.chat.completions.create(**api_kwargs)

                # ===== 成功: 解析响应 =====
                content = response.choices[0].message.content
                content = content.strip() if content else ""
                raw_content = content

                completion_tokens = response.usage.completion_tokens if response.usage else 0
                prompt_tokens = response.usage.prompt_tokens if response.usage else 0

                print(f"  [CoT] ✅ LLM 调用成功! 返回 {len(content)} 字符 "
                      f"(tokens: prompt={prompt_tokens}, completion={completion_tokens})")

                finish_reason = response.choices[0].finish_reason
                if finish_reason != "stop":
                    print(f"  [CoT] 警告: 输出被截断 (finish_reason={finish_reason})")

                # ===== 空内容检测 =====
                # GLM 在 finish_reason=length 时可能返回空 content，
                # 即使 completion_tokens > 0
                if not content:
                    print(f"  [CoT] ⚠️ 空内容返回 (completion_tokens={completion_tokens}), "
                          f"可能因 token 耗尽未生成 JSON")

                    if attempt < max_retries:
                        wait = 2 ** attempt
                        print(f"  [CoT] {wait}s 后重试...")
                        time.sleep(wait)
                        continue
                    else:
                        print(f"  [CoT] ❌ 多次重试仍返回空内容, 使用 fallback")
                        if fallback:
                            self.full_log.append({
                                "step_name": step_name,
                                "system_role": system_role,
                                "prompt": prompt,
                                "raw_response": raw_content,
                                "parsed_result": None,
                                "error": f"Empty content after {max_retries + 1} attempts",
                            })
                            return fallback
                        return {"error": "LLM returned empty content"}

                # 递归剥离多层 markdown 代码块 (处理 ```json\n```json\n... 嵌套)
                content = raw_content
                while True:
                    if content.startswith("```json"):
                        content = content[7:]
                    elif content.startswith("```"):
                        content = content[3:]
                    else:
                        break
                    end_idx = content.rfind("```")
                    if end_idx >= 0:
                        content = content[:end_idx]
                    content = content.strip()
                    if not (content.startswith("```json") or content.startswith("```")):
                        break

                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    repaired = self._repair_json(content)
                    if repaired is not None:
                        parsed = repaired
                    else:
                        raise

                self.full_log.append({
                    "step_name": step_name,
                    "system_role": system_role,
                    "prompt": prompt,
                    "raw_response": raw_content,
                    "parsed_result": parsed,
                })

                return parsed

            except json.JSONDecodeError as e:
                # JSON 解析失败 (非 API 错误)
                if attempt < max_retries:
                    wait = 2 ** attempt
                    print(f"  [CoT] ⚠️ JSONDecodeError: {str(e)[:80]}, "
                          f"{wait}s 后重试 ({attempt + 1}/{max_retries + 1})...")
                    time.sleep(wait)
                else:
                    print(f"  [CoT] ❌ JSON 解析失败 {max_retries + 1} 次: {e}")
                    self.full_log.append({
                        "step_name": step_name,
                        "system_role": system_role,
                        "prompt": prompt,
                        "raw_response": raw_content,
                        "parsed_result": None,
                        "error": f"JSONDecodeError: {str(e)[:200]}",
                    })
                    if fallback:
                        return fallback
                    return {"error": "JSON parse failed"}

            except Exception as e:
                error_str = str(e).lower()
                is_timeout = any(kw in error_str for kw in
                    ["timeout", "timed out", "timed_out", "connection", "reset"])

                # 检测 response_format 不支持的错误, 降级到普通模式
                if use_json_mode and any(kw in error_str for kw in
                    ["response_format", "response type", "unsupported", "invalid parameter"]):
                    print(f"  [CoT] ⚠️ response_format 不支持, 降级到普通模式")
                    use_json_mode = False
                    continue  # 立即重试, 不等待

                if attempt < max_retries:
                    if is_timeout:
                        print(f"  [CoT] ⚠️ 超时, 立即重试 ({attempt + 1}/{max_retries + 1})...")
                    else:
                        wait = 2 ** attempt
                        print(f"  [CoT] ⚠️ {type(e).__name__}: {str(e)[:80]}, "
                              f"{wait}s 后重试 ({attempt + 1}/{max_retries + 1})...")
                        time.sleep(wait)
                else:
                    print(f"  [CoT] ❌ 全部 {max_retries + 1} 次尝试均失败: {e}")

        # ===== 所有重试耗尽 → 返回 fallback =====
        print(f"  [CoT] LLM 调用/解析失败, 使用 SpeciesResolver fallback")
        self.full_log.append({
            "step_name": step_name,
            "system_role": system_role,
            "prompt": prompt,
            "raw_response": raw_content,
            "parsed_result": None,
            "error": "All retries exhausted",
        })
        if fallback:
            return fallback
        return {"error": "LLM call failed after all retries"}

    def _repair_json(self, raw: str) -> Optional[dict]:
        """修复被截断的 JSON"""
        try:
            s = raw.rstrip()
            for _ in range(10):
                try:
                    return json.loads(s)
                except json.JSONDecodeError:
                    for closer, counterpart in [('}', '{'), (']', '[')]:
                        opens = s.count(counterpart) - s.count(closer)
                        if opens > 0:
                            s = s.rstrip(',').rstrip() + closer * opens
                            break
                    else:
                        last_comma = s.rfind(',')
                        last_brace = max(s.rfind('}'), s.rfind(']'))
                        if last_comma > last_brace and last_comma > 0:
                            s = s[:last_comma].rstrip()
                            continue
                        break
            return None
        except Exception:
            return None
