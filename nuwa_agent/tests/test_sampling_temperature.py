"""Focused checks for temperature-controlled, protein-constrained codon sampling."""

import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

os.environ.setdefault("NUWA_API_KEY", "test-only")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_registry import CODON_TO_AA, NUWAModelRegistry, build_codon_tokenizer


class FixedLogitModel:
    def __init__(self, tokenizer):
        self.vocab_size = tokenizer.vocab_size
        self.uuu = tokenizer.convert_tokens_to_ids("UUU")
        self.uuc = tokenizer.convert_tokens_to_ids("UUC")

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask, token_type_ids):
        logits = torch.zeros((*input_ids.shape, self.vocab_size))
        logits[..., self.uuu] = 2.0
        logits[..., self.uuc] = 1.0
        return SimpleNamespace(logits=logits)


@pytest.fixture
def generator():
    tokenizer = build_codon_tokenizer(model_max_length=8)
    return NUWAModelRegistry(), FixedLogitModel(tokenizer), tokenizer


def _sample(generator, protein, temperature, count=1):
    registry, model, tokenizer = generator
    return registry._generate_protein_batch_vectorized(
        [protein] * count, model, tokenizer, torch.device("cpu"),
        temperature=temperature, top_p=1.0,
    )


def test_temperature_changes_actual_codon_sampling(generator):
    random.seed(7)
    torch.manual_seed(7)
    cold = _sample(generator, "MF", 0.2, count=512)
    random.seed(7)
    torch.manual_seed(7)
    hot = _sample(generator, "MF", 2.0, count=512)

    cold_uuu = sum(seq.split()[1] == "UUU" for seq in cold)
    hot_uuu = sum(seq.split()[1] == "UUU" for seq in hot)
    assert cold_uuu > hot_uuu + 100
    assert all([CODON_TO_AA[c] for c in seq.split()] == ["M", "F", "*"]
               for seq in cold + hot)


@pytest.mark.parametrize("protein", ["FM", "FM*"])
def test_preserves_first_and_last_amino_acids_and_adds_one_stop(generator, protein):
    for seq in _sample(generator, protein, 1.0, count=16):
        codons = seq.split()
        assert len(codons) == 3
        assert [CODON_TO_AA[c] for c in codons] == ["F", "M", "*"]


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf"), True])
def test_rejects_invalid_temperature_before_loading_model(temperature):
    with pytest.raises(ValueError, match="temperature must be a finite positive number"):
        NUWAModelRegistry().generate("unavailable", "MF", temperature=temperature)


def test_rejects_length_that_requires_truncating_protein(generator):
    registry, model, tokenizer = generator
    tokenizer.model_max_length = 2
    with pytest.raises(ValueError, match="exceeding model_max_length"):
        _sample((registry, model, tokenizer), "MF", 1.0)
