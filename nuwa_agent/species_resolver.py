"""SpeciesResolver: 物种名 → (domain, class_id, confidence, candidates)

NUWA BERT 使用 token_type_ids=class_id 嵌入物种信息:
  - NUWA-Bacteria: class_id ∈ [0, 19675]  (19676 个物种)
  - NUWA-Eukaryote: class_id ∈ [0, 4687]  (4688 个物种)
  - NUWA-Archaea: class_id ∈ [0, 701]      (702 个物种)

用户输入物种名 → 四级查找, 找不到精确匹配时返回候选项供 LLM CoT 讨论:
  1. 精确匹配 → confidence=exact, candidates=[]
  2. 属级匹配 → confidence=genus_match, candidates=[同属物种...]
  3. 模糊匹配 → confidence=fuzzy_match, candidates=[相似物种...]
  4. 全部失败 → confidence=default, class_id=0, candidates=[]
"""

import json
import os
from typing import Optional, List, Dict
from difflib import get_close_matches
from dataclasses import dataclass, field
from config import SPECIES_MAPS_DIR


@dataclass
class CandidateSpecies:
    """候选近亲物种"""
    name: str           # 物种名
    class_id: int       # 对应 class_id
    domain: str         # 所属域

    def to_dict(self) -> dict:
        return {"name": self.name, "class_id": self.class_id, "domain": self.domain}


@dataclass
class SpeciesInfo:
    """物种解析结果"""
    domain: str                               # "bacteria" | "eukaryote" | "archaea" | "unknown"
    class_id: int                             # 建议的 class_id (exact=确定的, 其他=建议值)
    confidence: str                           # "exact" | "genus_match" | "fuzzy_match" | "default"
    matched_name: str = None                  # 精确匹配到的物种名
    reason: str = ""                          # 解释
    candidates: List[CandidateSpecies] = field(default_factory=list)  # 候选项供 LLM 讨论

    @property
    def is_resolved(self) -> bool:
        """是否精确解析到具体物种"""
        return self.confidence == "exact"

    @property
    def needs_cot_discussion(self) -> bool:
        """是否需要 LLM 在 CoT 中讨论选择哪个候选"""
        return self.confidence in ("genus_match", "fuzzy_match") and len(self.candidates) > 0


class SpeciesResolver:
    """物种名 → (domain, class_id) 解析器

    当找不到精确匹配时，收集候选项（同属/模糊匹配）返回给 LLM CoT 讨论。
    不做程序化自动选择——选择权交给 LLM 的领域知识。
    """

    MAX_CANDIDATES = 10  # 最多返回的候选项数量

    def __init__(self, maps_dir: str = SPECIES_MAPS_DIR):
        self.maps = {}
        self.domain_files = {
            "bacteria": os.path.join(maps_dir, "bacteria_species_mapping.json"),
            "eukaryote": os.path.join(maps_dir, "eukaryote_species_mapping.json"),
            "archaea": os.path.join(maps_dir, "archaea_species_mapping.json"),
        }

        for domain, path in self.domain_files.items():
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        raw = json.load(f)
                    if isinstance(raw, dict) and "species_to_id" in raw:
                        mapping = raw["species_to_id"]
                    else:
                        mapping = raw
                    self.maps[domain] = mapping
                    print(f"[SpeciesResolver] Loaded {domain}: {len(mapping)} species "
                          f"from {os.path.basename(path)}")
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"[SpeciesResolver] ⚠️ Failed to load {path}: {e}")
                    self.maps[domain] = {}
            else:
                print(f"[SpeciesResolver] ⚠️ Missing: {path}")
                self.maps[domain] = {}

    def resolve(self, species_name: str) -> SpeciesInfo:
        """
        解析物种名 → (domain, class_id, confidence, candidates)

        找不到精确匹配时，返回候选项列表让 LLM 自己选。
        """
        normalized = species_name.lower().strip()

        # ---- Step 1: 精确匹配 ----
        for domain, mapping in self.maps.items():
            lower_map = {k.lower(): (k, v) for k, v in mapping.items()}
            if normalized in lower_map:
                orig_name, cid = lower_map[normalized]
                return SpeciesInfo(
                    domain=domain,
                    class_id=cid,
                    confidence="exact",
                    matched_name=orig_name,
                    reason=f"Exact match in {domain}: '{orig_name}' → class_id={cid}",
                    candidates=[],
                )

        # ---- Step 2: 属级匹配 —— 收集所有同属物种为候选项 ----
        genus = normalized.split()[0] if " " in normalized else normalized
        # 提取物种名部分 (第二个词，用于同种优先排序)
        species_part = normalized.split()[1] if len(normalized.split()) >= 2 else ""
        for domain, mapping in self.maps.items():
            same_genus_candidates: List[CandidateSpecies] = []
            for k, v in mapping.items():
                k_lower = k.lower()
                if k_lower.startswith(genus + " ") or k_lower == genus:
                    same_genus_candidates.append(
                        CandidateSpecies(name=k, class_id=v, domain=domain)
                    )

            if same_genus_candidates:
                # 排序策略: 同种 (species 词相同) 优先 → 再按 class_id
                # 例如 "Escherichia coli" → E. coli K-12/O157:H7 排在 E. albertii 前面
                def _sort_key(c: CandidateSpecies) -> tuple:
                    c_lower = c.name.lower()
                    parts = c_lower.split()
                    c_species = parts[1] if len(parts) >= 2 else ""
                    # 同种匹配 (第二个词相同) 排在最前面 (key=0)
                    same_species = 0 if (species_part and c_species == species_part) else 1
                    return (same_species, c.class_id)

                same_genus_candidates.sort(key=_sort_key)
                candidates = same_genus_candidates[:self.MAX_CANDIDATES]

                return SpeciesInfo(
                    domain=domain,
                    class_id=candidates[0].class_id,  # 建议值，LLM 可覆盖
                    confidence="genus_match",
                    matched_name=candidates[0].name,
                    reason=f"No exact match for '{species_name}', "
                           f"found {len(same_genus_candidates)} same-genus species in {domain}",
                    candidates=candidates,
                )

        # ---- Step 3: 模糊匹配 —— 收集所有相似物种为候选项 ----
        for domain, mapping in self.maps.items():
            all_names = list(mapping.keys())
            matches = get_close_matches(normalized, all_names, n=self.MAX_CANDIDATES, cutoff=0.6)
            if matches:
                candidates = [
                    CandidateSpecies(name=m, class_id=mapping[m], domain=domain)
                    for m in matches
                ]
                return SpeciesInfo(
                    domain=domain,
                    class_id=candidates[0].class_id,  # 建议值，LLM 可覆盖
                    confidence="fuzzy_match",
                    matched_name=candidates[0].name,
                    reason=f"Fuzzy match: '{species_name}' → top {len(candidates)} similar "
                           f"species in {domain}",
                    candidates=candidates,
                )

        # ---- Step 4: 全部失败 → LLM 全权判断 ----
        return SpeciesInfo(
            domain="unknown",
            class_id=0,
            confidence="default",
            matched_name=None,
            reason=f"Species '{species_name}' not found in any mapping table. "
                   f"LLM must determine domain and decide if class_id=0 is appropriate.",
            candidates=[],
        )

    def get_all_species(self, domain: str) -> list:
        """获取某个域的所有已知物种名"""
        return list(self.maps.get(domain, {}).keys())

    def get_class_id(self, domain: str, species_name: str) -> Optional[int]:
        """直接查询某个域的 class_id"""
        mapping = self.maps.get(domain, {})
        normalized = species_name.lower().strip()
        for k, v in mapping.items():
            if k.lower() == normalized:
                return v
        return None


# 全局单例
resolver = SpeciesResolver()
