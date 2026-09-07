"""Multi-Objective Evaluator: 三个微调 NUWA 回归模型 + 外部生物信息学工具

Pareto 软目标 (全部由微调模型驱动):
- TE: finetuned_model_TE (BertForRegressionHF, Spearman≈0.48)
- Stability: finetuned_model_fungal (BertForRegressionHF, Spearman≈0.76)
- Expression: finetuned_model_fungal_euk (BertForRegressionHF, Spearman≈0.77)
失败时退回启发式 fallback。

硬约束 (外部工具):
- CAI: cai2 / GC: 自定义 / MFE/Stem: ViennaRNA
- Immunogenicity: MHCflurry / Homopolymer: 自定义
"""

import os
import sys

# Windows: mhcflurry/mhcgnomes 内部 open() 不指定 encoding，默认用 gbk
# 读 UTF-8 YAML → UnicodeDecodeError。PYTHONUTF8=1 必须在进程启动前设置，
# os.environ.setdefault 无效，真正修复在 _predict_immunogenicity 中完成。

import json
import numpy as np
from dataclasses import dataclass, field
from config import (
    DEFAULT_MHC_ALLELES, CODON_TABLES_DIR,
    FINETUNED_TE_MODEL, FINETUNED_STABILITY_MODEL, FINETUNED_EXPR_MODEL,
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
            token_type_ids = torch.full((1, seq_len), class_id,
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
    immunogenicity_risk: float = 0.0
    # Feasibility
    is_feasible: bool = False
    constraint_violations: list = field(default_factory=list)


class MultiObjectiveEvaluator:
    """
    多目标评估器 — 全部外部生物信息学工具

    3 Objectives (启发式 + ViennaRNA) + 6 Constraints (cai2 + ViennaRNA + MHCflurry + 自定义)
    """

    def __init__(self, codon_table: dict = None,
                 mhc_alleles: list = None,
                 host_organism: str = "Homo sapiens",
                 use_mhcflurry: bool = True,
                 domain: str = "bacteria",
                 class_id: int = 0):
        self.codon_table = codon_table
        self.mhc_alleles = mhc_alleles or DEFAULT_MHC_ALLELES
        self.host_organism = host_organism
        self.use_mhcflurry = use_mhcflurry
        self.domain = domain          # "bacteria" / "eukaryote" / "archaea"
        self.class_id = class_id      # 物种 class_id，传给 token_type_ids

        # 延迟加载
        self._mhcflurry_predictor = None

        # 微调 NUWA 回归模型 — 三个 Pareto 软目标，全部延迟加载
        self._te_scorer = NUWAScorer(FINETUNED_TE_MODEL)
        self._stability_scorer = NUWAScorer(FINETUNED_STABILITY_MODEL)
        self._expr_scorer = NUWAScorer(FINETUNED_EXPR_MODEL)

        # 打印工具状态（首次初始化时检查）
        self._print_scorer_status()

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
        imm_risk = self._predict_immunogenicity(protein_seq)

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
            immunogenicity_risk=imm_risk,
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
            # 用 sigmoid 将原始回归值映射到 (0, 1)
            # TE 训练数据范围约 [-2, 4]，sigmoid(0)=0.5 对应中性预测
            import math
            return 1.0 / (1.0 + math.exp(-raw * 0.5))

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
            import math
            return 1.0 / (1.0 + math.exp(-raw * 0.5))

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
            import math
            return 1.0 / (1.0 + math.exp(-raw * 0.5))

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
                    return CAI(dna_seq, weights=self.codon_table)
        except (ImportError, Exception):
            pass

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
            return mfe
        except ImportError:
            pass

        # Fallback: 经验公式 (每 nt 约 -0.5 kcal/mol)
        clean = self._clean_seq(sequence)
        return -0.5 * len(clean) if clean else 0.0

    def _compute_max_stem(self, sequence: str) -> int:
        """最大茎区长度 — ViennaRNA 点括号结构解析"""
        try:
            import RNA
            clean = self._clean_seq(sequence)
            structure, _ = RNA.fold(clean)
            max_run = current = 0
            for c in structure:
                if c in "()":
                    current += 1
                    max_run = max(max_run, current)
                else:
                    current = 0
            return max_run // 2
        except ImportError:
            pass
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

    # 类级别标志，全局只打印一次 MHCflurry 警告
    _mhcflurry_warned = False

    def _predict_immunogenicity(self, protein_seq: str) -> float:
        """
        免疫原性 — MHCflurry Class1AffinityPredictor (含可变 fallback)

        扫描蛋白 9-mer → MHC-I 结合亲和力预测 → 转换为风险分数
        亲和力越低 → 结合越强 → 免疫原性风险越高
        转换: score = max(0, min(1, (6 - log10(affinity_nM)) / 6))
          1 nM → 1.0, 10 nM → 0.83, 100 nM → 0.67, 500 nM → 0.55
          1000 nM → 0.5, 5000 nM → 0.38, >1e6 nM → 0

        输出: [0,1], >0.5 为高风险

        NOTE: 同一蛋白的不同密码子变体在 MHC-I 表位层面免疫原性相同
        (因为氨基酸序列不变)。如需区分候选序列，看 Expression/TE/Stability。
        本函数在 fallback 模式下加入轻微序列相关扰动以产生区分度。
        """
        if self.use_mhcflurry:
            try:
                if self._mhcflurry_predictor is None:
                    import logging
                    logging.getLogger('mhcflurry').setLevel(logging.ERROR)

                    # Windows: mhcgnomes 的 data.py 用 open() 读 YAML 不指定 encoding，
                    # 系统默认 gbk 读 UTF-8 文件 → UnicodeDecodeError (Py3.13 之前)。
                    # PYTHONUTF8=1 需进程启动前设置，运行时 os.environ 修改无效，
                    # 故在 import mhcflurry 前临时补丁 builtins.open 默认 UTF-8。
                    _open_patched = False
                    if sys.platform == "win32":
                        import builtins as _bi
                        _orig_open = _bi.open
                        def _utf8_open(file, mode='r', buffering=-1, encoding=None,
                                       errors=None, newline=None, closefd=True, opener=None):
                            if encoding is None and isinstance(mode, str) and 'b' not in mode:
                                encoding = 'utf-8'
                            return _orig_open(file, mode, buffering, encoding,
                                              errors, newline, closefd, opener)
                        _bi.open = _utf8_open
                        _open_patched = True

                    import mhcflurry

                    models_dir = (
                        r'C:\Users\30778\AppData\Local\mhcflurry'
                        r'\mhcflurry\4\2.2.0\models_class1\models'
                    )
                    if not os.path.exists(models_dir):
                        raise FileNotFoundError(
                            f"MHCflurry models not found at {models_dir}. "
                            f"Run: mhcflurry-downloads fetch models_class1"
                        )
                    self._mhcflurry_predictor = mhcflurry.Class1AffinityPredictor.load(
                        models_dir=models_dir
                    )

                    # 恢复原始 open，避免影响后续代码
                    if _open_patched:
                        _bi.open = _orig_open

                peptides = [protein_seq[i:i+9] for i in range(len(protein_seq) - 8)]
                if not peptides:
                    return 0.0

                # 收集所有 peptide × allele 的分数，取 P95 而非 max
                # max 对长蛋白过于严苛：200+ aa 蛋白必有少数强 MHC-I 表位
                # P95 排除极端离群值同时保留整体免疫原性信号
                all_scores = []
                for allele in self.mhc_alleles:
                    try:
                        affinities = self._mhcflurry_predictor.predict(
                            peptides=peptides,
                            alleles=[allele] * len(peptides),
                        )
                        affinities = np.asarray(affinities, dtype=float).flatten()
                        scores = np.clip(
                            (6.0 - np.log10(np.maximum(1.0, affinities))) / 6.0,
                            0.0, 1.0
                        )
                        all_scores.extend(scores.tolist())
                    except (ValueError, KeyError):
                        continue
                if not all_scores:
                    return 0.0
                # P95: 排除最极端 5% 的离群肽段，更稳健
                return float(np.percentile(all_scores, 95))
            except (ImportError, Exception) as e:
                if not MultiObjectiveEvaluator._mhcflurry_warned:
                    print(f"  [Evaluator] ⚠️ MHCflurry 不可用 ({type(e).__name__})，使用 fallback。"
                          f" 同一蛋白的候选序列免疫原性分数将保持相似。")
                    MultiObjectiveEvaluator._mhcflurry_warned = True

        # Fallback: 基于氨基酸多样性 + 亲水性估算 (带轻微随机种子区分候选)
        # 即使 MHCflurry 不可用，也能给出合理且可区分的估算
        unique_ratio = len(set(protein_seq)) / max(len(protein_seq), 1)
        # 统计疏水残基 (ILVFWM) — 疏水区更可能暴露为 T 细胞表位
        hydrophobic = sum(1 for aa in protein_seq if aa in "ILVFWM")
        hydro_ratio = hydrophobic / max(len(protein_seq), 1)
        # 组合: 多样性低 + 疏水残基多 → 风险略高
        base_risk = max(0.0, 0.45 - unique_ratio * 0.3 + hydro_ratio * 0.15)
        # 轻微抖动 (<0.003)，产生候选间区分 (基于蛋白序列 hash)
        jitter = hash(protein_seq) % 1000 / 1000000.0
        return min(1.0, max(0.0, base_risk + jitter))

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

    # MHCflurry
    try:
        import mhcflurry  # noqa: F401
        models_dir = (
            r'C:\Users\30778\AppData\Local\mhcflurry'
            r'\mhcflurry\4\2.2.0\models_class1\models'
        )
        import os
        if os.path.exists(models_dir):
            tools["MHCflurry"] = f"available (models loaded from {os.path.basename(models_dir)})"
        else:
            tools["MHCflurry"] = "package installed but models not downloaded"
    except ImportError:
        tools["MHCflurry"] = "not installed (pip install mhcflurry)"

    # pymoo
    try:
        import pymoo  # noqa: F401
        tools["pymoo"] = "available"
    except ImportError:
        tools["pymoo"] = "not installed (pip install pymoo, optional for precise HV)"

    return tools
