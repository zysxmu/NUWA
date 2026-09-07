"""NUWA 模型注册表：加载 + 生成

复用项目现有的 BertForMaskedLM 模型加载和 entropy-guide 生成逻辑。
每个域 (bacteria/eukaryote/archaea) 对应一个独立训练的 NUWA 模型检查点。

注意: NUWA 没有感知头 — 所有评测分数来自外部生物信息学工具 (evaluator.py)
"""

import os
import random
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertForMaskedLM, PreTrainedTokenizerFast, BertConfig
from safetensors.torch import load_file as _load_safetensors

from config import (
    NUWA_MODELS, NUM_CANDIDATES,
    GENERATION_TOP_K, GENERATION_TOP_P,
)

# --- Codon <-> Amino Acid 常量 ---

CODON_TO_AA = {
    'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
    'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M',
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
    'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
    'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*',
    'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W',
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

AA_TO_CODONS: Dict[str, List[str]] = {}
for _codon, _aa in CODON_TO_AA.items():
    if _aa not in AA_TO_CODONS:
        AA_TO_CODONS[_aa] = []
    AA_TO_CODONS[_aa].append(_codon)

STOP_CODONS = ["UAA", "UAG", "UGA"]


# --- Tokenizer ---

def build_codon_tokenizer(model_max_length: int = 512) -> PreTrainedTokenizerFast:
    """构建 64 密码子 WordLevel tokenizer（与原项目一致）"""
    lst_ele = list('AUGC')
    lst_voc = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    for a1 in lst_ele:
        for a2 in lst_ele:
            for a3 in lst_ele:
                lst_voc.append(f'{a1}{a2}{a3}')

    dic_voc = dict(zip(lst_voc, range(len(lst_voc))))

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import BertProcessing

    tokenizer_obj = Tokenizer(WordLevel(vocab=dic_voc, unk_token="[UNK]"))
    tokenizer_obj.add_special_tokens(['[PAD]', '[CLS]', '[UNK]', '[SEP]', '[MASK]'])
    tokenizer_obj.pre_tokenizer = Whitespace()
    tokenizer_obj.post_processor = BertProcessing(
        ("[SEP]", dic_voc['[SEP]']),
        ("[CLS]", dic_voc['[CLS]']),
    )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        do_lower_case=False,
        clean_text=False,
        tokenize_chinese_chars=False,
        strip_accents=False,
        unk_token='[UNK]',
        sep_token='[SEP]',
        pad_token='[PAD]',
        cls_token='[CLS]',
        mask_token='[MASK]',
        model_max_length=model_max_length
    )
    print(f"[ModelRegistry] Codon tokenizer built. Vocab size: {len(dic_voc)}")
    return tokenizer


# --- Registry ---

class NUWAModelRegistry:
    """管理 3 个 NUWA BERT 模型：加载 + 生成"""

    def __init__(self):
        self.models: dict = {}
        self.tokenizers: dict = {}
        self._loaded: dict = {}

    def load(self, model_key: str, device: str = "auto"):
        """加载指定模型到 GPU

        支持两种格式:
          - HuggingFace safetensors 检查点 (model.safetensors + config.json)
          - 旧版 .pt 检查点 (BertForMaskedLM.from_pretrained)
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        target_device = torch.device(device)

        if model_key in self._loaded:
            cached = self._loaded[model_key]
            if cached is not None and cached.device != target_device:
                cached = cached.to(target_device)
                self._loaded[model_key] = cached
            return cached

        if model_key not in NUWA_MODELS:
            raise ValueError(f"Unknown model key: {model_key}. Available: {list(NUWA_MODELS.keys())}")

        cfg = NUWA_MODELS[model_key]
        model_path = cfg["path"]

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[ModelRegistry] 模型路径不存在: {model_path}\n"
                f"  请检查 config.py 中 NUWA_MODELS['{model_key}']['path'] 配置"
            )

        print(f"[ModelRegistry] 加载 {cfg['name']} from {model_path}")

        # 构建 codon tokenizer (所有域共用 69 vocab size)
        max_pos = cfg.get("max_position_embeddings", 1024)
        tokenizer = build_codon_tokenizer(model_max_length=max_pos)
        self.tokenizers[model_key] = tokenizer

        # 加载模型权重
        safetensors_file = os.path.join(model_path, "model.safetensors")
        if os.path.exists(safetensors_file):
            # safetensors 格式 (HuggingFace Trainer 保存)
            # 注意: 训练时 tie_word_embeddings=True, decoder 权重与 embedding 共享
            print(f"[ModelRegistry]   检测到 safetensors 格式, 权重大小: {os.path.getsize(safetensors_file) / 1e9:.2f} GB")
            config = BertConfig.from_pretrained(model_path)
            model = BertForMaskedLM(config)
            state_dict = _load_safetensors(safetensors_file)
            # strict=False: decoder 权重在 tie_word_embeddings 模式下不单独存储
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"[ModelRegistry]   缺失 keys (已自动跳过): {missing_keys}")
            if unexpected_keys:
                print(f"[ModelRegistry]   多余 keys (已忽略): {unexpected_keys}")
            # 绑定 decoder 权重到 embedding (tie_word_embeddings)
            model.cls.predictions.decoder.weight = model.get_input_embeddings().weight
            model.cls.predictions.decoder.bias = model.cls.predictions.bias
        else:
            # 标准 HuggingFace 格式 (.bin) 或 .pt
            print(f"[ModelRegistry]   使用 from_pretrained 加载")
            model = BertForMaskedLM.from_pretrained(model_path)

        model.to(target_device)
        model.eval()
        self._loaded[model_key] = model

        print(f"[ModelRegistry]   {cfg['name']} 加载完成, 设备: {target_device}")
        return model

    def generate(self, model_key: str, protein_seq: str, num: int = NUM_CANDIDATES,
                 temperature: float = 0.8, top_p: float = GENERATION_TOP_P,
                 feedback: Optional[str] = None,
                 class_id: Optional[int] = None,    # 可选覆盖 class_id
                 batch_size: int = 64, device: str = "auto") -> List[str]:
        """用指定模型生成候选 mRNA 序列

        Args:
            model_key: 模型 key (bacteria/eukaryote/archaea)
            protein_seq: 目标蛋白序列
            num: 生成数量
            temperature: 生成温度
            top_p: nucleus sampling 阈值
            feedback: 上一轮 LLM 反馈
            class_id: 物种 ID (对应 --class_id), None=使用域默认值
            batch_size: 批量大小
            device: 计算设备
        """
        model = self.load(model_key, device=device)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_device = torch.device(device)

        tokenizer = self.tokenizers[model_key]
        cfg = NUWA_MODELS[model_key]

        if feedback:
            print(f"[ModelRegistry] 接收到反馈: {feedback[:80]}...")

        # class_id: 优先用传入的物种 ID，否则用域默认值
        if class_id is None:
            domain_to_class_id = {"Bacteria": 0, "Eukaryote": 1, "Archaea": 2}
            class_id = domain_to_class_id.get(cfg["domain"], 0)
        print(f"[ModelRegistry] 使用 class_id={class_id} (domain={cfg['domain']})")

        all_candidates = []
        remaining = num

        while remaining > 0:
            current_batch = min(batch_size, remaining)
            batch_proteins = [protein_seq] * current_batch

            generated_batch = self._generate_protein_batch_vectorized(
                target_protein_seqs=batch_proteins,
                model=model,
                tokenizer=tokenizer,
                device=torch_device,
                temperature=temperature,
                top_p=top_p,
                class_id=class_id,
            )

            all_candidates.extend(generated_batch)
            remaining -= len(generated_batch)
            print(f"[ModelRegistry] 已生成 {len(all_candidates)}/{num} 条候选序列")

        return all_candidates[:num]

    def get_model_info(self, model_key: str) -> dict:
        """获取模型描述信息 (供 LLM 讨论参考)"""
        cfg = NUWA_MODELS[model_key]
        return {
            "key": model_key,
            "name": cfg["name"],
            "domain": cfg["domain"],
            "description": cfg["description"],
            "strengths": cfg["strengths"],
            "limitations": cfg["limitations"],
        }

    def list_models(self) -> List[str]:
        """列出所有可用模型 key"""
        return list(NUWA_MODELS.keys())

    def _generate_protein_batch_vectorized(
        self,
        target_protein_seqs: List[str],
        model,
        tokenizer: PreTrainedTokenizerFast,
        device: torch.device,
        temperature: float = 1.0,
        top_p: float = 0.9,
        class_id: int = 0,
    ) -> List[str]:
        """蛋白质约束的向量化生成 (复用 entropy_guide_mRNA_generation.py 逻辑)"""
        model.eval()
        batch_size = len(target_protein_seqs)

        original_lengths = [len(s) for s in target_protein_seqs]
        max_len = max(original_lengths)

        effective_max_len = min(max_len, tokenizer.model_max_length)
        if max_len > tokenizer.model_max_length:
            original_lengths = [min(l, effective_max_len) for l in original_lengths]
            target_protein_seqs = [s[:effective_max_len] for s in target_protein_seqs]
            max_len = effective_max_len

        protein_pad_char = 'X'
        if protein_pad_char not in AA_TO_CODONS:
            AA_TO_CODONS[protein_pad_char] = []

        padded_protein_seqs = [s.ljust(max_len, protein_pad_char) for s in target_protein_seqs]

        unique_aas = sorted(list(AA_TO_CODONS.keys()))
        aa_to_idx = {aa: i for i, aa in enumerate(unique_aas)}

        aa_constraint_matrix = torch.zeros(
            (len(unique_aas), tokenizer.vocab_size), dtype=torch.float, device=device
        )
        for aa, codons in AA_TO_CODONS.items():
            aa_idx = aa_to_idx.get(aa)
            if aa_idx is None:
                continue
            codon_ids = [tokenizer.convert_tokens_to_ids(c) for c in codons if c in tokenizer.vocab]
            if codon_ids:
                aa_constraint_matrix[aa_idx].index_fill_(
                    0, torch.tensor(codon_ids, device=device, dtype=torch.long), 1.0
                )

        protein_aa_indices = torch.tensor(
            [[aa_to_idx.get(aa, aa_to_idx[protein_pad_char]) for aa in protein_seq]
             for protein_seq in padded_protein_seqs],
            dtype=torch.long, device=device,
        )

        batch_constraint_mask = aa_constraint_matrix[protein_aa_indices]

        batch_input_ids = torch.full(
            (batch_size, max_len), tokenizer.mask_token_id, dtype=torch.long, device=device
        )
        attention_mask = torch.arange(max_len, device=device)[None, :] < \
            torch.tensor(original_lengths, device=device)[:, None]

        stop_codon_ids = [tokenizer.convert_tokens_to_ids(c) for c in STOP_CODONS if c in tokenizer.vocab]
        for i in range(batch_size):
            length = original_lengths[i]
            batch_input_ids[i, 0] = tokenizer.convert_tokens_to_ids("AUG")
            if length > 1:
                batch_input_ids[i, length - 1] = random.choice(stop_codon_ids)

        forbidden_stop_mask = torch.zeros(tokenizer.vocab_size, dtype=torch.float, device=device)
        for fid in stop_codon_ids:
            if fid is not None:
                forbidden_stop_mask[fid] = -float('inf')

        num_iterations = max_len - 2
        for step in tqdm(range(num_iterations), desc="Generating", leave=False):
            token_type_ids = torch.full_like(batch_input_ids, fill_value=class_id)

            with torch.no_grad():
                outputs = model(
                    input_ids=batch_input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                batch_logits = outputs.logits

            batch_logits.masked_fill_(batch_constraint_mask == 0, -float('inf'))

            batch_probs = torch.softmax(batch_logits / temperature, dim=-1)
            entropies = -torch.sum(batch_probs * torch.log(batch_probs + 1e-8), dim=-1)

            is_mask = (batch_input_ids == tokenizer.mask_token_id)
            selectable_mask = is_mask & attention_mask
            masked_entropies = entropies.masked_fill(~selectable_mask, -1.0)

            selected_positions = torch.argmax(masked_entropies, dim=1)
            active_mask = (masked_entropies.max(dim=1).values > -1.0)

            if not active_mask.any():
                break
            active_indices = active_mask.nonzero(as_tuple=True)[0]

            active_batch_logits = batch_logits[active_indices]
            active_selected_positions = selected_positions[active_indices]

            idx_tensor = active_selected_positions.view(-1, 1, 1).expand(
                -1, -1, active_batch_logits.shape[-1]
            )
            selected_logits = torch.gather(active_batch_logits, 1, idx_tensor).squeeze(1)

            selected_logits += forbidden_stop_mask

            sorted_logits, sorted_indices = torch.sort(selected_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = torch.zeros_like(sorted_indices_to_remove).scatter_(
                1, sorted_indices, sorted_indices_to_remove
            )
            selected_logits.masked_fill_(indices_to_remove, -float('inf'))

            final_probs = F.softmax(selected_logits, dim=-1)
            rows_with_zeros = final_probs.sum(dim=-1) == 0
            if rows_with_zeros.any():
                active_protein_indices = protein_aa_indices[active_indices]
                selected_aa_indices = torch.gather(
                    active_protein_indices, 1, active_selected_positions.unsqueeze(1)
                ).squeeze(1)
                fallback_constraint = aa_constraint_matrix[selected_aa_indices]
                fallback_constraint.masked_fill_(forbidden_stop_mask == -float('inf'), 0.0)
                uniform_fallback = fallback_constraint / (fallback_constraint.sum(dim=-1, keepdim=True) + 1e-8)
                final_probs[rows_with_zeros] = uniform_fallback[rows_with_zeros]

            sampled_token_ids = torch.multinomial(final_probs, 1)

            current_active_ids = batch_input_ids[active_indices]
            updated_active_ids = current_active_ids.scatter_(
                1, active_selected_positions.unsqueeze(1), sampled_token_ids
            )
            batch_input_ids[active_indices] = updated_active_ids

        output_sequences = []
        for i in range(batch_size):
            valid_ids = batch_input_ids[i, :original_lengths[i]]
            output_sequences.append(tokenizer.decode(valid_ids, skip_special_tokens=True))

        return output_sequences


# 全局单例
registry = NUWAModelRegistry()
