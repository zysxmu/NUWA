#!/usr/bin/env python3
"""
Script for pre-training a CodonBERT model using a combined loss:
1. Masked Language Model (MLM) Loss
2. Supervised Contrastive (SupCon) Loss based on species ID.

Uses an N-K sampling strategy for batch creation and supports multi-GPU
training with DDP.
"""

import os
import argparse
import json
import random
from functools import partial
from itertools import islice
from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import accuracy_score

from datasets import Dataset, IterableDataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import BertProcessing

from transformers import (
    TrainingArguments,
    Trainer,
    TrainerCallback,
    PreTrainedTokenizerFast,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
)
from transformers.file_utils import ModelOutput

# Determine DDP (Distributed Data Parallel) environment
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))

# Pre-defined model architectures
MODEL_CONFIGS = {
    "base": {
        "hidden_size": 768,
        "inter_size": 3072,
        "num_heads": 12,
        "num_layers": 12,
    }
}


## --- Utility Functions --- ##

def is_main_process():
    """Check if the current process is the main one (rank 0)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def compute_metrics(eval_pred):
    """Compute MLM accuracy for evaluation."""
    logits, labels = eval_pred
    mlm_logits = logits[0]  # Model output can be a tuple
    mlm_labels = labels

    mlm_preds = np.argmax(mlm_logits, axis=-1)

    # Create a mask to ignore padding tokens (labeled -100)
    mask = mlm_labels != -100
    mlm_acc = accuracy_score(mlm_labels[mask].flatten(), mlm_preds[mask].flatten())

    return {
        "mlm_acc": mlm_acc,
    }


## --- Tokenizer Definition --- ##

def build_codon_tokenizer() -> PreTrainedTokenizerFast:
    """Builds a WordLevel tokenizer for 64 codons."""
    lst_ele = list('AUGC')
    # Start with special tokens
    lst_voc = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    
    # Add all 64 codons
    for a1 in lst_ele:
        for a2 in lst_ele:
            for a3 in lst_ele:
                lst_voc.append(f'{a1}{a2}{a3}')

    dic_voc = {token: i for i, token in enumerate(lst_voc)}
    
    tokenizer = Tokenizer(WordLevel(vocab=dic_voc, unk_token="[UNK]"))
    tokenizer.add_special_tokens(['[PAD]', '[CLS]', '[UNK]', '[SEP]', '[MASK]'])
    tokenizer.pre_tokenizer = Whitespace()  # Assumes codons are space-separated
    tokenizer.post_processor = BertProcessing(
        ("[SEP]", dic_voc['[SEP]']),
        ("[CLS]", dic_voc['[CLS]']),
    )

    bert_tokenizer_fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        do_lower_case=False,
        clean_text=False,
        tokenize_chinese_chars=False,
        strip_accents=False,
        unk_token='[UNK]',
        sep_token='[SEP]',
        pad_token='[PAD]',
        cls_token='[CLS]',
        mask_token='[MASK]'
    )
    return bert_tokenizer_fast


## --- Data Loading & Sampling (N-K Strategy) --- ##

def nk_sample_generator(class_seq_dict, class2id, N, K, max_batches):
    """
    Core N-K sampling generator.
    Yields samples by first picking N classes, then K samples from each.
    """
    all_classes = list(class_seq_dict.keys())
    if not all_classes:
        print(f"Warning: class_seq_dict is empty. Generator will yield nothing.")
        return

    for _ in range(max_batches):
        try:
            sampled_classes = random.sample(all_classes, N)
        except ValueError:
            # Not enough classes to sample N, sample with replacement
            if not all_classes: continue # Safeguard
            sampled_classes = random.choices(all_classes, k=N)
        
        batch = []
        for cls in sampled_classes:
            seqs = class_seq_dict[cls]
            if len(seqs) < K:
                # Sample with replacement if not enough sequences
                selected_seqs = random.choices(seqs, k=K)
            else:
                selected_seqs = random.sample(seqs, K)
            
            for seq in selected_seqs:
                batch.append({
                    "seq": seq,
                    "species_id": class2id[cls],  # Used for SupCon Loss
                    "species_class": class2id[cls]  # Used for token_type_ids
                })

        if len(batch) == N * K:
            # Flatten to yield sample-wise
            for item in batch:
                yield item


def simple_nk_generator(class_seq_dict, class2id, N, K, max_batches):
    """Wrapper for the N-K generator."""
    yield from nk_sample_generator(class_seq_dict, class2id, N, K, max_batches)


def generator_fn(rank, world_size, class_seq_dict, class2id, N, K, max_batches):
    """
    DDP-aware generator function.
    Calculates the number of batches for this specific rank.
    """
    # Calculate batches per rank to handle uneven distribution
    batches_per_rank = max_batches // world_size
    remainder = max_batches % world_size
    if rank < remainder:
        batches_per_rank += 1
        
    if is_main_process():
        print(f"Total batches (max_steps): {max_batches}. Batches for rank {rank}: {batches_per_rank}.")

    # Generate only the data for this rank
    yield from nk_sample_generator(
        class_seq_dict=class_seq_dict,
        class2id=class2id,
        N=N,
        K=K,
        max_batches=batches_per_rank  # Only generate batches for this rank
    )


def encode_string(example, tokenizer, max_length):
    """Tokenize a single sequence string and format for the model."""
    encodings = tokenizer(
        example['seq'],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_special_tokens_mask=True
    )
    # Use species_class as token_type_ids
    encodings['token_type_ids'] = [example['species_class']] * max_length
    encodings['species_id'] = example['species_id']
    return encodings


def load_and_split_data(data_path, val_ratio, test_ratio, min_samples):
    """Loads the class_seq JSON and splits it into train/val/test dicts."""
    print(f"Loading data from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        class_seq_dict = json.load(f)
    
    # Build class2id mapping
    class2id = {cls: i for i, cls in enumerate(sorted(class_seq_dict.keys()))}
    
    train_dict, val_dict, test_dict = {}, {}, {}
    total_train, total_val, total_test = 0, 0, 0

    for cls, seqs in class_seq_dict.items():
        seqs = seqs.copy()
        random.shuffle(seqs)
        n = len(seqs)

        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        n_train = n - n_val - n_test

        # Filter: ensure val/test sets have at least min_samples (K)
        if n_val < min_samples or n_test < min_samples or n_train < min_samples:
            continue
        
        train_dict[cls] = seqs[:n_train]
        val_dict[cls] = seqs[n_train : n_train + n_val]
        test_dict[cls] = seqs[n_train + n_val :]
        
        total_train += n_train
        total_val += n_val
        total_test += n_test

    print(f"Data loaded. Total classes: {len(class_seq_dict)}")
    print(f"Filtered classes (with >= {min_samples} val/test samples): {len(train_dict)}")
    print(f"Total Train Samples: {total_train}")
    print(f"Total Val Samples:   {total_val}")
    print(f"Total Test Samples:  {total_test}")

    return train_dict, val_dict, test_dict, class2id, total_train


## --- Custom Model (BERT + SupCon Loss) --- ##

def supervised_contrastive_loss(features, labels, temperature=0.07):
    """Implementation of the Supervised Contrastive Loss."""
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    batch_size = features.shape[0]
    device = features.device

    # Create a boolean mask for positive pairs (samples with the same label)
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).to(device)
    
    # Calculate cosine similarity (dot product of normalized features)
    contrast = torch.div(torch.matmul(features, features.T), temperature)

    # Mask to remove self-comparison (diagonal)
    logits_mask = torch.ones_like(mask, device=device) - torch.eye(batch_size, device=device)

    # Mask for positive pairs (excluding self)
    mask = mask * logits_mask
    
    # Compute log-probability
    exp_contrast = torch.exp(contrast) * logits_mask
    log_prob = contrast - torch.log(exp_contrast.sum(dim=1, keepdim=True) + 1e-12)

    # Compute mean log-likelihood over positive pairs
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
    
    # Final loss is the negative mean of mean_log_prob_pos
    loss = -mean_log_prob_pos.mean()
    return loss


@dataclass
class CustomModelOutput(ModelOutput):
    """Custom output dataclass to hold individual loss components."""
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    mlm_loss: torch.FloatTensor = None
    supcon_loss: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None
    attentions: torch.FloatTensor = None


class BertWithSupConLoss(BertForMaskedLM):
    """
    BERT model modified to compute a combined loss:
    Total Loss = MLM Loss + supcon_weight * SupCon Loss
    """
    def __init__(self, config, supcon_weight=1.0, temperature=0.07):
        super().__init__(config)
        self.supcon_weight = supcon_weight
        self.temperature = temperature

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        species_id=None,  # shape: (batch,) - for SupCon
        labels=None,      # shape: (batch, seq) - for MLM
        **kwargs
    ):
        # 1. Run the standard BertForMaskedLM forward pass
        # We need hidden_states to get the [CLS] embedding
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_hidden_states=True,  # Crucial for getting [CLS] embedding
            **kwargs
        )

        # 2. Calculate Supervised Contrastive Loss
        supcon_loss = torch.tensor(0.0, device=outputs.loss.device)
        if species_id is not None:
            # Get [CLS] token embedding from the last hidden state
            # outputs.hidden_states[-1] shape: (batch, seq_len, hidden_size)
            cls_embeds = outputs.hidden_states[-1][:, 0, :]  # (batch, hidden_size)
            supcon_loss = supervised_contrastive_loss(
                cls_embeds, species_id, temperature=self.temperature
            )
        
        # 3. Combine losses
        mlm_loss = outputs.loss
        total_loss = mlm_loss + self.supcon_weight * supcon_loss

        # 4. Return custom output for logging
        return CustomModelOutput(
            loss=total_loss,
            logits=outputs.logits,
            mlm_loss=mlm_loss,
            supcon_loss=self.supcon_weight * supcon_loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


## --- Custom Trainer & Callbacks --- ##

class SupConLoggingTrainer(Trainer):
    """
    Custom Trainer to log mlm_loss and supcon_loss separately.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_logs = {}

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            outputs = model(**inputs)
            loss = outputs.loss
            
            # Store individual losses for logging
            self._loss_logs = {
                'mlm_loss': getattr(outputs, 'mlm_loss', torch.tensor(0.0)).detach().item(),
                'supcon_loss': getattr(outputs, 'supcon_loss', torch.tensor(0.0)).detach().item(),
                'total_loss': loss.detach().item()
            }

        self.accelerator.backward(loss)
        return loss.detach()

    def log(self, logs: Dict[str, float]) -> None:
        """Logs metrics, including custom loss components."""
        # Add our stored loss components to the logs
        logs.update(self._loss_logs)
        super().log(logs)


class DynamicMLMDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator with a dynamically adjustable mlm_probability.
    """
    def __init__(self, *args, initial_mlm_probability=0.15, max_mlm_probability=0.5, **kwargs):
        self.max_mlm_probability = max_mlm_probability
        self._mlm_probability = initial_mlm_probability
        super().__init__(*args, mlm_probability=initial_mlm_probability, **kwargs)

    @property
    def mlm_probability(self):
        return self._mlm_probability

    @mlm_probability.setter
    def mlm_probability(self, value):
        # Enforce the maximum probability limit
        self._mlm_probability = min(value, self.max_mlm_probability)
        # Update the parent class's property as well
        super(DynamicMLMDataCollator, self.__class__).mlm_probability.fset(self, self._mlm_probability)

    def set_mlm_probability(self, value):
        self.mlm_probability = value


class DynamicMLMProbabilityCallback(TrainerCallback):
    """
    A callback to linearly increase the MLM probability during training.
    """
    def __init__(self, data_collator, start=0.15, end=0.5, total_steps=10000):
        self.data_collator = data_collator
        self.start = start
        self.end = end
        self.total_steps = total_steps

    def on_step_end(self, args, state, control, **kwargs):
        # Linearly increase probability
        progress = min(state.global_step / self.total_steps, 1.0)
        new_prob = self.start + (self.end - self.start) * progress
        self.data_collator.set_mlm_probability(new_prob)
        
        # Optional: log the change
        # if state.global_step % 100 == 0 and is_main_process():
        #     print(f"[Step {state.global_step}] mlm_probability set to {self.data_collator.mlm_probability:.4f}")


def build_species_supcon_collate(data_collator):
    """
    Wrapper for the data collator to add 'species_id' to the batch.
    The MLM collator (parent) doesn't know about 'species_id'.
    """
    def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1. Extract 'species_id' before it's removed by the MLM collator
        species_ids = [f['species_id'] for f in features]
        
        # 2. Let the MLM collator do its work (masking, padding)
        # 'seq' and 'species_id' are not model inputs, so they are fine to be in `features`
        batch = data_collator(features)
        
        # 3. Add the 'species_id' tensor to the final batch
        batch['species_id'] = torch.tensor(species_ids, dtype=torch.long)
        return batch
    return collate_fn


## --- Main Execution --- ##

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="CodonBERT Pre-training with MLM and SupCon Loss")
    
    # Data and Model Paths
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Path to the training data (class_seq.json)")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory to save model checkpoints")

    # Model Configuration
    parser.add_argument("--model_config_name", type=str, default="base", choices=MODEL_CONFIGS.keys(),
                        help="Name of the model config to use (e.g., 'base'")
    parser.add_argument("--num_organisms", type=int, default=19673,
                        help="Total number of unique species (type_vocab_size)")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length")

    # N-K Sampling
    parser.add_argument("--num_class_per_batch", type=int, default=4,
                        help="N: Number of classes (species) per batch")
    parser.add_argument("--num_sample_per_class", type=int, default=4,
                        help="K: Number of samples per class")
    parser.add_argument("--num_class_per_batch_val", type=int, default=4,
                        help="N for validation")
    parser.add_argument("--num_sample_per_class_val", type=int, default=4,
                        help="K for validation")
    
    # Loss Configuration
    parser.add_argument("--supcon_weight", type=float, default=1.0,
                        help="Weight for the supervised contrastive loss")
    parser.add_argument("--supcon_temperature", type=float, default=1,
                        help="Temperature for the supervised contrastive loss")
    
    # Dynamic MLM
    parser.add_argument("--mlm_prob_start", type=float, default=0.15,
                        help="Initial MLM probability")
    parser.add_argument("--mlm_prob_end", type=float, default=0.5,
                        help="Final MLM probability (linearly scaled to this value)")

    # Training Hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=10000,
                        help="Warmup steps for learning rate scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--log_steps", type=int, default=20,
                        help="Log metrics every N steps")
    parser.add_argument("--eval_steps_factor", type=int, default=1,
                        help="Evaluate every N * log_steps")
    parser.add_argument("--save_steps_factor", type=int, default=0,
                        help="Save every N * max_steps (0 = never, 0.1 = 10% of total steps)")
    parser.add_argument("--eval_samples", type=int, default=2000,
                        help="Number of samples to use for validation/testing")

    # Data Splitting
    parser.add_argument("--val_ratio", type=float, default=0.02, help="Fraction of data for validation")
    parser.add_argument("--test_ratio", type=float, default=0.01, help="Fraction of data for testing")

    return parser.parse_args()


def main():
    """Main training and evaluation function."""
    args = parse_args()

    if is_main_process():
        os.makedirs(args.model_dir, exist_ok=True)
        print(f"World size: {WORLD_SIZE}, Rank: {RANK}")
        print(f"Model config: {args.model_config_name}")
        
    # 1. Build Tokenizer
    bert_tokenizer_fast = build_codon_tokenizer()
    vocab_size = bert_tokenizer_fast.vocab_size
    print(f"Tokenizer built. Vocab size: {vocab_size}")

    # 2. Load and Split Data
    bs_train = args.num_class_per_batch * args.num_sample_per_class
    bs_valtest = args.num_class_per_batch_val * args.num_sample_per_class_val
    
    # Min samples = K for val/test
    min_samples_k = max(args.num_sample_per_class, args.num_sample_per_class_val)

    train_dict, val_dict, test_dict, class2id, total_train_samples = load_and_split_data(
        args.train_data_path,
        args.val_ratio,
        args.test_ratio,
        min_samples_k
    )

    # 3. Calculate Training Steps
    total_iterations = (total_train_samples * args.num_train_epochs) / bs_train
    max_steps = int(total_iterations / WORLD_SIZE) # Steps per GPU
    
    if args.save_steps_factor > 0:
        save_steps = int(max_steps * args.save_steps_factor)
        save_strategy = "steps"
    else:
        save_steps = 0
        save_strategy = "no"

    if is_main_process():
        print(f"Total train samples: {total_train_samples}, Epochs: {args.num_train_epochs}")
        print(f"Batch size per GPU: {bs_train} (N={args.num_class_per_batch}, K={args.num_sample_per_class})")
        print(f"Total iterations: {total_iterations:.0f}, Max steps per GPU: {max_steps}")
        print(f"Save strategy: {save_strategy}, Save steps: {save_steps}")

    # 4. Create Datasets
    
    # Create the tokenization function with fixed tokenizer and max_length
    encode_fn = partial(encode_string, 
                        tokenizer=bert_tokenizer_fast, 
                        max_length=args.max_length)

    # A. Train Dataset (Streaming IterableDataset)
    train_generator = IterableDataset.from_generator(
        generator_fn,
        gen_kwargs={
            "rank": RANK,
            "world_size": WORLD_SIZE,
            "class_seq_dict": train_dict,
            "class2id": class2id,
            "N": args.num_class_per_batch,
            "K": args.num_sample_per_class,
            "max_batches": max_steps
        }
    )
    train_padded_generator = train_generator.map(encode_fn, batched=False)

    # B. Eval/Test Datasets (Fixed-size in-memory Datasets)
    # This matches the original logic of slicing the generator
    if is_main_process():
        print(f"Creating fixed val/test datasets with {args.eval_samples} samples each...")
        
    eval_samples = list(islice(simple_nk_generator(
        val_dict, class2id, args.num_class_per_batch_val, args.num_sample_per_class_val,
        max_batches=int(args.eval_samples / bs_valtest) + 1), args.eval_samples))
    eval_samples = [encode_fn(x) for x in eval_samples]
    eval_dataset = Dataset.from_list(eval_samples)

    test_samples = list(islice(simple_nk_generator(
        test_dict, class2id, args.num_class_per_batch_val, args.num_sample_per_class_val,
        max_batches=int(args.eval_samples / bs_valtest) + 1), args.eval_samples))
    test_samples = [encode_fn(x) for x in test_samples]
    test_dataset = Dataset.from_list(test_samples)

    if is_main_process():
        print(f"Created eval_dataset ({len(eval_dataset)} samples) and test_dataset ({len(test_dataset)} samples)")


    # 5. Define Model
    model_params = MODEL_CONFIGS[args.model_config_name]
    model_config = BertConfig(
        vocab_size=vocab_size,
        max_position_embeddings=args.max_length,
        num_hidden_layers=model_params["num_layers"],
        num_attention_heads=model_params["num_heads"],
        hidden_size=model_params["hidden_size"],
        intermediate_size=model_params["inter_size"],
        type_vocab_size=args.num_organisms
    )
    
    model = BertWithSupConLoss(
        model_config, 
        supcon_weight=args.supcon_weight, 
        temperature=args.supcon_temperature
    )
    if is_main_process():
        print(f"Model created with {model.num_parameters() / 1e6:.2f} M parameters.")

    # 6. Define Collator and Callback
    data_collator = DynamicMLMDataCollator(
        tokenizer=bert_tokenizer_fast,
        mlm=True,
        initial_mlm_probability=args.mlm_prob_start,
        max_mlm_probability=args.mlm_prob_end
    )

    collate_fn = build_species_supcon_collate(data_collator)

    dynamic_mlm_callback = DynamicMLMProbabilityCallback(
        data_collator=data_collator,
        start=args.mlm_prob_start,
        end=args.mlm_prob_end,
        total_steps=max_steps
    )

    # 7. Define Training Arguments
    training_args = TrainingArguments(
        optim='adamw_torch',
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_steps=max_steps,
        per_device_train_batch_size=bs_train,
        per_device_eval_batch_size=bs_valtest,
        eval_strategy="steps",
        eval_steps=args.log_steps * args.eval_steps_factor,
        logging_steps=args.log_steps,
        save_steps=save_steps,
        save_strategy=save_strategy,
        load_best_model_at_end=False,
        output_dir=args.model_dir,
        overwrite_output_dir=True,
        save_total_limit=10,
        fp16=True,
        dataloader_num_workers=0,  # IterableDataset often works best with 0
        ddp_backend='nccl' if WORLD_SIZE > 1 else None,
        local_rank=RANK if WORLD_SIZE > 1 else -1,
        eval_accumulation_steps=4,
        report_to="none", # Disable wandb/etc. unless specified
    )

    # 8. Initialize Trainer
    trainer = SupConLoggingTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_padded_generator,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[dynamic_mlm_callback]
    )

    # 9. Train and Evaluate
    print("Starting training...")
    trainer.train()
    
    if is_main_process():
        print("Training finished.")
        print("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_dataset)
        print("Test Set Metrics:")
        print(test_metrics)
        
        # Save final model
        trainer.save_model(os.path.join(args.model_dir, "final_model"))
        bert_tokenizer_fast.save_pretrained(os.path.join(args.model_dir, "final_model"))
        print("Final model saved.")


if __name__ == "__main__":
    main()


