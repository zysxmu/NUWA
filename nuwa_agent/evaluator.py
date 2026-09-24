"""Multi-Objective Evaluator: 三个微调 NUWA 回归模型 + 外部生物信息学工具

Pareto 软目标 (全部由微调模型驱动):
- TE: finetuned_model_TE (BertForRegressionHF, Spearman≈0.48)
- Stability: finetuned_model_fungal (BertForRegressionHF, Spearman≈0.76)
- Expression: finetuned_model_fungal_euk (BertForRegressionHF, Spearman≈0.77)
正式实验默认要求这些模型与工具全部可用；仅 demo 可显式启用启发式 fallback。

硬约束 (外部工具):
- CAI: cai2 / GC: 自定义 / MFE/Stem: ViennaRNA
- Homopolymer: 自定义
"""

import os
import numpy as np
from dataclasses import dataclass, field
from config import (
    FINETUNED_TE_MODEL, FINETUNED_STABILITY_MODEL, FINETUNED_EXPR_MODEL,
    STRICT_EVALUATION,
)


# ============ NUWA 微调回归模型封装 ============

class NUWAScorer:
    """封装 BertForRegressionHF 微调模型，用于对 mRNA 序列打分。

    输入: mRNA 序列（含空格分隔密码子，如 "AUG UAA ..."）
    输出: float 标量预测值（TE / Stability / Expression）

    架构: BertModel(NUWA backbone) + Linear(768, 1)
    推理: [CLS] pooler_output → regressor → scalar

    三个 scorer 实例分别对应三个 Pareto 软目标:
    - _te_scorer: finetuned_model_TE → TE
    - _stability_scorer: finetuned_model_fungal → Stability
    - _expr_scorer: finetuned_model_fungal_euk → Expression
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._load_error = None

    def _lazy_load(self):
        """延迟加载，首次调用时加载模型"""
        if self._loaded:
            return
        try:
            import torch
            from transformers import BertConfig, BertModel, PreTrainedModel
            from transformers import PreTrainedTokenizerFast
            import torch.nn as nn
            from safetensors.torch import load_file as load_safetensors

            # --- 构建 BertForRegressionHF（与 finetune_Reg.py 一致）---
            class _BertRegressor(PreTrainedModel):
                config_class = BertConfig
                def __init__(self, config):
                    super().__init__(config)
                    self.bert = BertModel(config)
                    self.regressor = nn.Linear(config.hidden_size, 1)
                    self.post_init()
                def forward(self, input_ids=None, attention_mask=None,
                            token_type_ids=None, **kwargs):
                    out = self.bert(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)
                    return self.regressor(out.pooler_output).squeeze(-1)

            config = BertConfig.from_pretrained(self.model_path)
            model = _BertRegressor(config)

            # 加载 safetensors 权重
            safetensors_path = os.path.join(self.model_path, "model.safetensors")
            state_dict = load_safetensors(safetensors_path, device=self.device)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            model.eval()
            model.to(self.device)
            self._model = model

            # 加载 tokenizer（BPE/WordLevel，从 checkpoint 目录读取）
            tokenizer_path = self.model_path
            if os.path.exists(os.path.join(tokenizer_path, "tokenizer.json")):
                self._tokenizer = PreTrainedTokenizerFast.from_pretrained(
                    tokenizer_path, local_files_only=True
                )
            else:
                # fallback: 用 model_registry 的 codon tokenizer
                from model_registry import build_codon_tokenizer
                self._tokenizer = build_codon_tokenizer()

            self._loaded = True
        except Exception as e:
            self._load_error = str(e)
            self._loaded = True  # 标记已尝试加载，避免重复

    def predict(self, sequence: str, class_id: int = 0) -> float:
        """预测单条 mRNA 序列的分数。

        Args:
            sequence: mRNA 序列（可含空格分隔密码子，也可无空格）
            class_id: 物种 class_id，对应 token_type_ids

        Returns:
            float 分数（原始预测值，不归一化）；加载失败返回 None
        """
        self._lazy_load()
        if self._load_error or self._model is None:
            return None
        try:
            import torch
            # 将序列按密码子分词：去除空格后每3个字符一组
            clean = sequence.replace(" ", "").upper()
            # 构造密码子 token 序列（保持与训练时一致的格式）
            codons = [clean[i:i+3] for i in range(0, len(clean) - len(clean) % 3, 3)]
            codon_str = " ".join(codons)  # "AUG CUG ..."

            enc = self._tokenizer(
                codon_str,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False,
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            seq_len = input_ids.shape[1]
            type_vocab_size = self._model.config.type_vocab_size
            safe_class_id = min(int(class_id), type_vocab_size - 1)
            token_type_ids = torch.full((1, seq_len), safe_class_id,
                                        dtype=torch.long, device=self.device)

            with torch.no_grad():
                score = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                ).item()
            return float(score)
        except Exception:
            return None

    @property
    def available(self) -> bool:
        self._lazy_load()
        return self._load_error is None and self._model is not None


@dataclass
class EvaluationResult:
    """单条序列的评估结果"""
    sequence: str
    # Objectives (maximize)
    te_score: float = 0.0
    stability_score: float = 0.0
    expression_score: float = 0.0
    # Constraint metrics
    cai: float = 0.0
    gc_content: float = 0.0
    mfe: float = 0.0
    max_stem_len: int = 0
    max_homopolymer: int = 0
    # Feasibility
    is_feasible: bool = False
    constraint_violations: list = field(default_factory=list)


class MultiObjectiveEvaluator:
    """
    多目标评估器 — 全部外部生物信息学工具

    3 objectives + 5 sequence-level constraints (cai2 + ViennaRNA + 自定义)
    """

    def __init__(self, codon_table: dict = None,
                 host_organism: str = "Homo sapiens",
                 domain: str = "bacteria",
                 class_id: int = 0,
                 strict: bool = STRICT_EVALUATION):
        self.codon_table = codon_table
        self.host_organism = host_organism
        self.domain = domain          # "bacteria" / "eukaryote" / "archaea"
        self.class_id = class_id      # 物种 class_id，传给 token_type_ids
        self.strict = bool(strict)

        # 延迟加载
        self._backend_usage = {
            "te": None, "stability": None, "expression": None,
            "cai": None, "folding": None,
        }

        # 微调 NUWA 回归模型 — 三个 Pareto 软目标，全部延迟加载
        self._te_scorer = NUWAScorer(FINETUNED_TE_MODEL)
        self._stability_scorer = NUWAScorer(FINETUNED_STABILITY_MODEL)
        self._expr_scorer = NUWAScorer(FINETUNED_EXPR_MODEL)

        # 打印工具状态（首次初始化时检查）
        self._print_scorer_status()

    def validate_backends(self) -> dict:
        """Fail before a formal run if a declared quantitative backend is unavailable."""
        failures = []
        for name, scorer in (
            ("TE", self._te_scorer), ("Stability", self._stability_scorer),
            ("Expression", self._expr_scorer),
        ):
            if not scorer.available:
                failures.append(f"{name} model: {scorer._load_error or 'unavailable'}")
        try:
            import RNA  # noqa: F401
        except Exception as exc:
            failures.append(f"ViennaRNA: {type(exc).__name__}")
        if not self.codon_table:
            failures.append("host codon table: unavailable")
        try:
            from cai2 import CAI  # noqa: F401
        except Exception as exc:
            failures.append(f"cai2: {type(exc).__name__}")
        try:
            import pymoo  # noqa: F401
        except Exception as exc:
            failures.append(f"pymoo: {type(exc).__name__}")
        if failures and self.strict:
            raise RuntimeError(
                "Formal evaluation cannot start because required backends are unavailable:\n- "
                + "\n- ".join(failures)
            )
        return {"strict": self.strict, "failures": failures}

    def backend_usage(self) -> dict:
        """Return the actual scorer/tool path used in this run."""
        return dict(self._backend_usage)

    def _print_scorer_status(self):
        """打印三个微调模型路径（仅首次初始化时调用）"""
        print(f"[Evaluator] domain={self.domain}, class_id={self.class_id}")
        print(f"[Evaluator] TE    scorer: {FINETUNED_TE_MODEL}")
        print(f"[Evaluator] Stab  scorer: {FINETUNED_STABILITY_MODEL}")
        print(f"[Evaluator] Expr  scorer: {FINETUNED_EXPR_MODEL}")
        print("[Evaluator] (模型将在首次评估时延迟加载)")

    def evaluate_batch(self, sequences: list, protein_seq: str) -> list:
        """批量评估"""
        return [self.evaluate_one(seq, protein_seq) for seq in sequences]

    def evaluate_one(self, sequence: str, protein_seq: str) -> EvaluationResult:
        """评估单条序列"""

        # ---- Objectives (外部工具) ----
        te = self._predict_te(sequence)
        stability = self._predict_stability(sequence)
        expression = self._predict_expression(sequence, protein_seq)

        # ---- Constraints (外部工具 + 自定义) ----
        cai = self._compute_cai(sequence)
        gc = self._compute_gc(sequence)
        mfe = self._compute_mfe(sequence)
        max_stem = self._compute_max_stem(sequence)
        max_homo = self._compute_max_homopolymer(sequence)

        return EvaluationResult(
            sequence=sequence,
            te_score=te,
            stability_score=stability,
            expression_score=expression,
            cai=cai,
            gc_content=gc,
            mfe=mfe,
            max_stem_len=max_stem,
            max_homopolymer=max_homo,
        )

    # ============ Objectives ============

    def _predict_te(self, sequence: str) -> float:
        """翻译效率 — 优先使用微调 NUWA TE 回归模型；失败时退回启发式

        微调模型: finetuned_model_TE (Spearman≈0.484, bacteria backbone)
        原始预测值为 log-scale TE，通过 sigmoid 归一化到 (0, 1)
        Fallback: 基于 GC + 长度的启发式估算
        """
        raw = self._te_scorer.predict(sequence, class_id=self.class_id)
        if raw is not None:
            self._backend_usage["te"] = "NUWA regression model"
            # 用 sigmoid 将原始回归值映射到 (0, 1)
            # TE 训练数据范围约 [-2, 4]，sigmoid(0)=0.5 对应中性预测
            import math
            return 1.0 / (1.0 + math.exp(-raw * 0.5))

        if self.strict:
            raise RuntimeError("TE regression model failed during strict evaluation")
        self._backend_usage["te"] = "heuristic fallback"
        # Fallback: 基于 GC + 长度的启发式估算
        clean = self._clean_seq(sequence)
        gc = self._compute_gc(sequence)
        length_score = min(1.0, 500 / max(len(clean), 1))
        return 0.5 + 0.3 * (1 - abs(gc - 0.55)) + 0.2 * length_score

    def _predict_stability(self, sequence: str) -> float:
        """mRNA 稳定性 — 优先使用微调 NUWA Stability 回归模型；失败时退回 ViennaRNA

        微调模型: finetuned_model_fungal (Spearman≈0.758, bacteria backbone)
        原始预测值通过 sigmoid 归一化到 (0, 1)
        Fallback: ViennaRNA MFE 归一化
        """
        raw = self._stability_scorer.predict(sequence, class_id=self.class_id)
        if raw is not None:
            self._backend_usage["stability"] = "NUWA regression model"
            import math
            return 1.0 / (1.0 + math.exp(-raw * 0.5))

        if self.strict:
            raise RuntimeError("Stability regression model failed during strict evaluation")
        self._backend_usage["stability"] = "ViennaRNA heuristic fallback"
        # Fallback: ViennaRNA MFE 归一化
        mfe = self._compute_mfe(sequence)
        clean = self._clean_seq(sequence)
        seq_len = len(clean) if clean else 1
        mfe_norm_factor = 0.45
        return min(1.0, abs(mfe) / (seq_len * mfe_norm_factor))

    def _predict_expression(self, sequence: str, protein_seq: str) -> float:
        """蛋白表达量 — 优先使用微调 NUWA Expression 回归模型；失败时退回启发式

        微调模型: finetuned_model_fungal_euk (Spearman≈0.768, eukaryote backbone, type_vocab=4688)
        原始预测值通过 sigmoid 归一化到 (0, 1)
        Fallback: TE + CAI 综合估算
        """
        raw = self._expr_scorer.predict(sequence, class_id=self.class_id)
        if raw is not None:
            self._backend_usage["expression"] = "NUWA regression model"
            import math
            return 1.0 / (1.0 + math.exp(-raw * 0.5))

        if self.strict:
            raise RuntimeError("Expression regression model failed during strict evaluation")
        self._backend_usage["expression"] = "TE+CAI heuristic fallback"
        # Fallback: TE + CAI 综合估算
        te = self._predict_te(sequence)
        cai = self._compute_cai(sequence)
        return 0.6 * te + 0.4 * cai

    # ============ Constraints ============

    def _compute_cai(self, sequence: str) -> float:
        """CAI — cai2 (Python lib)

        需要宿主参考密码子表，从 CODON_TABLES_DIR 加载。
        NUWA 生成 mRNA (含 U)，密码子表用 DNA 格式 (含 T)，
        传入 cai2 前自动 U→T 转换。
        """
        try:
            from cai2 import CAI
            if self.codon_table:
                # 去除空格, 转换 mRNA (U) → DNA (T)
                dna_seq = self._clean_seq(sequence).replace("U", "T")
                # 确保序列长度是 3 的倍数
                if len(dna_seq) % 3 != 0:
                    dna_seq = dna_seq[:len(dna_seq) - (len(dna_seq) % 3)]
                if len(dna_seq) >= 3:
                    value = CAI(dna_seq, weights=self.codon_table)
                    self._backend_usage["cai"] = "cai2 with host codon table"
                    return value
        except Exception as exc:
            if self.strict:
                raise RuntimeError("CAI computation failed during strict evaluation") from exc

        if self.strict:
            raise RuntimeError("CAI requires cai2 and a host codon table in strict evaluation")
        self._backend_usage["cai"] = "GC heuristic fallback"
        # Fallback: 基于 GC 含量的粗略估算
        gc = self._compute_gc(sequence)
        return 0.5 + 0.3 * (1 - abs(gc - 0.55))

    def _compute_gc(self, sequence: str) -> float:
        """GC 含量 — O(n) 自定义

        自动去除空格 (NUWA 生成的 mRNA 含空格分隔密码子)
        """
        clean = sequence.replace(" ", "").upper()
        if not clean:
            return 0.0
        return sum(1 for c in clean if c in "GC") / len(clean)

    def _compute_mfe(self, sequence: str) -> float:
        """MFE — ViennaRNA Python 绑定"""
        try:
            import RNA
            clean = self._clean_seq(sequence)
            _, mfe = RNA.fold(clean)
            self._backend_usage["folding"] = "ViennaRNA"
            return mfe
        except Exception as exc:
            if self.strict:
                raise RuntimeError("ViennaRNA MFE computation failed") from exc

        self._backend_usage["folding"] = "length heuristic fallback"
        # Fallback: 经验公式 (每 nt 约 -0.5 kcal/mol)
        clean = self._clean_seq(sequence)
        return -0.5 * len(clean) if clean else 0.0

    def _compute_max_stem(self, sequence: str) -> int:
        """最大茎区长度 — ViennaRNA 点括号结构解析"""
        try:
            import RNA
            clean = self._clean_seq(sequence)
            structure, _ = RNA.fold(clean)
            self._backend_usage["folding"] = "ViennaRNA"
            max_run = current = 0
            for c in structure:
                if c in "()":
                    current += 1
                    max_run = max(max_run, current)
                else:
                    current = 0
            return max_run // 2
        except Exception as exc:
            if self.strict:
                raise RuntimeError("ViennaRNA structure computation failed") from exc
        self._backend_usage["folding"] = "constant stem fallback"
        return 10  # fallback

    def _compute_max_homopolymer(self, sequence: str) -> int:
        """最大同聚物长度 — O(n) 自定义

        修复 (2026-08-22): 先去除空格再计算连续 run。
        NUWA 生成的 mRNA 为空格分隔密码子格式 ("AUG GUU ..."), 此前直接在
        原始字符串上计算, 空格把连续 run 截断为最多 3, 导致该约束恒通过、
        从未真正生效。现与 _compute_gc/_compute_mfe/_compute_max_stem 对齐。
        """
        clean = sequence.replace(" ", "").upper()
        if not clean:
            return 0
        max_run = current = 1
        for i in range(1, len(clean)):
            if clean[i] == clean[i - 1]:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 1
        return max_run

    # ============ 辅助 ============

    @staticmethod
    def _clean_seq(sequence: str) -> str:
        """去除空格，统一大写 (NUWA 生成 mRNA 含空格分隔)"""
        return sequence.replace(" ", "").upper()


# ============ 工具可用性检查 ============

def check_tool_availability():
    """检查所有外部工具是否可用 (供调试用)"""
    tools = {}

    # ViennaRNA
    try:
        import RNA  # noqa: F401
        tools["ViennaRNA"] = "available"
    except ImportError:
        tools["ViennaRNA"] = "not installed (conda install -c bioconda viennarna)"

    # cai2
    try:
        from cai2 import CAI  # noqa: F401
        tools["cai2"] = "available"
    except ImportError:
        tools["cai2"] = "not installed (pip install cai2)"

    # pymoo
    try:
        import pymoo  # noqa: F401
        tools["pymoo"] = "available"
    except ImportError:
        tools["pymoo"] = "not installed (pip install pymoo, optional for precise HV)"

    return tools
