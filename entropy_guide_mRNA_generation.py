#!/usr/bin/env python3
"""
Generates Coding DNA Sequences (CDS) using a pre-trained CodonBERT
(BertForMaskedLM) model using mask-predict strategies. Supports various
generation modes, including protein-constrained and batch vectorized generation.
"""

import os
import math
import random
import argparse
import numpy as np
from Bio import SeqIO
import json
from tqdm import tqdm
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import BertProcessing
from transformers import BertForMaskedLM, PreTrainedTokenizerFast

# --- Constants ---

# Standard Codon -> Amino Acid mapping
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

# Reverse mapping: Amino Acid -> List of Codons
AA_TO_CODONS = {}
for codon, aa in CODON_TO_AA.items():
    if aa not in AA_TO_CODONS:
        AA_TO_CODONS[aa] = []
    AA_TO_CODONS[aa].append(codon)

STOP_CODONS = ["UAA", "UAG", "UGA"]


# --- Tokenizer ---

def build_codon_tokenizer(model_max_length: int = 512) -> PreTrainedTokenizerFast:
    """Builds a WordLevel tokenizer for 64 codons."""
    lst_ele = list('AUGC')
    lst_voc = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    for a1 in lst_ele:
        for a2 in lst_ele:
            for a3 in lst_ele:
                lst_voc.append(f'{a1}{a2}{a3}')

    dic_voc = dict(zip(lst_voc, range(len(lst_voc))))

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
    print(f"Codon tokenizer built. Vocab size: {len(dic_voc)}")
    return tokenizer


# --- Utility Functions ---

def calculate_entropy(probs: torch.Tensor) -> float:
    """Calculates the entropy of a probability distribution."""
    # Add epsilon to avoid log(0)
    probs = probs + 1e-8
    entropy = -torch.sum(probs * torch.log(probs))
    return entropy.item()


def translate_cds(cds_seq_str: str) -> str:
    """Translates a space-separated CDS sequence string into a protein sequence."""
    codons = cds_seq_str.split(' ')
    aa_seq = ""
    for codon in codons:
        aa = CODON_TO_AA.get(codon, "X") # Use 'X' for unknown/invalid codons
        aa_seq += aa
    return aa_seq


# --- Generation Functions ---


def generate_cds_sequence_batch_vectorized(
    num_sequences: int,
    seq_len: int,
    model: BertForMaskedLM,
    tokenizer: PreTrainedTokenizerFast,
    device: torch.device = "cpu",
    temperature: float = 1.0,
    top_p: float = 0.9,
    class_id: int = 0
):
    """
    Generates a batch of CDS sequences WITHOUT protein constraints using CodonBERT.
    Uses fully vectorized PyTorch operations for maximum speed.
    """
    model.eval()
    print(f"\n--- (Vectorized No-Protein) Generating {num_sequences} sequences of length {seq_len} ---")

    # Initialize batch
    batch_input_ids = torch.full(
        (num_sequences, seq_len),
        fill_value=tokenizer.mask_token_id,
        dtype=torch.long,
        device=device
    )

    # Set start/stop codons
    batch_input_ids[:, 0] = tokenizer.convert_tokens_to_ids("AUG")
    stop_codon_ids = [tokenizer.convert_tokens_to_ids(c) for c in STOP_CODONS if c in tokenizer.vocab]
    for i in range(num_sequences):
        batch_input_ids[i, -1] = random.choice(stop_codon_ids)

    # Create mask to forbid stop codons during sampling
    forbidden_mask = torch.zeros(tokenizer.vocab_size, dtype=torch.float, device=device)
    for fid in stop_codon_ids:
        if fid is not None:
            forbidden_mask[fid] = -float('inf')

    num_iterations = seq_len - 2
    for step in tqdm(range(num_iterations), desc="Generating (Vectorized No-Protein)"):
        # Prepare batch input
        attention_mask = torch.ones_like(batch_input_ids)
        token_type_ids = torch.full_like(batch_input_ids, fill_value=class_id)

        # --- 1. Sync Inference ---
        with torch.no_grad():
            outputs = model(input_ids=batch_input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            batch_logits = outputs.logits # (B, S, V)

        # --- 2. Vectorized Entropy Calculation & Position Selection ---
        batch_probs = torch.softmax(batch_logits / temperature, dim=-1) # (B, S, V)
        entropies = -torch.sum(batch_probs * torch.log(batch_probs + 1e-8), dim=-1) # (B, S)

        # Mask entropies, keeping only [MASK] positions
        is_mask = (batch_input_ids == tokenizer.mask_token_id)
        masked_entropies = entropies.masked_fill(~is_mask, -1.0) # Set non-mask entropy to -1

        # Find position with max entropy for each sequence
        selected_positions = torch.argmax(masked_entropies, dim=1) # (B,)

        # Check if any sequences are already fully generated (all entropies are -1)
        # If max entropy is -1, it means no masks were left. We can skip updates for these.
        active_mask = (masked_entropies.max(dim=1).values > -1.0) # (B,) boolean tensor
        if not active_mask.any():
            print("All sequences completed generation.")
            break
            
        active_indices = active_mask.nonzero(as_tuple=True)[0]

        # --- 3. Vectorized Probability Extraction & Sampling (Only for active sequences) ---
        active_batch_logits = batch_logits[active_indices] # (ActiveB, S, V)
        active_selected_positions = selected_positions[active_indices] # (ActiveB,)

        # Gather logits for the selected positions
        # Index needs to be (ActiveB, 1, V) for gather
        idx_tensor = active_selected_positions.view(-1, 1, 1).expand(-1, -1, active_batch_logits.shape[-1])
        selected_logits = torch.gather(active_batch_logits, 1, idx_tensor).squeeze(1) # (ActiveB, V)

        # Apply stop codon constraint
        selected_logits += forbidden_mask

        # Top-p filtering (vectorized)
        sorted_logits, sorted_indices = torch.sort(selected_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Use scatter to apply the removal mask based on original indices
        indices_to_remove = torch.zeros_like(sorted_indices_to_remove).scatter_(
            1, sorted_indices, sorted_indices_to_remove
        )
        selected_logits.masked_fill_(indices_to_remove, -float('inf'))

        # Sample from the filtered distribution
        final_probs = F.softmax(selected_logits, dim=-1)
        # Handle cases where all valid probabilities become zero after filtering
        rows_with_zeros = final_probs.sum(dim=-1) == 0
        if rows_with_zeros.any():
             # Fallback: uniform distribution over non-forbidden tokens
             uniform_fallback = torch.ones_like(final_probs)
             uniform_fallback.masked_fill_(forbidden_mask == -float('inf'), 0.0)
             uniform_fallback /= uniform_fallback.sum(dim=-1, keepdim=True)
             final_probs[rows_with_zeros] = uniform_fallback[rows_with_zeros]

        sampled_token_ids = torch.multinomial(final_probs, 1) # (ActiveB, 1)

        # --- 4. Vectorized Update (Only for active sequences) ---
        # Use scatter_ to update only the active sequences at their selected positions
        batch_input_ids[active_indices] = batch_input_ids[active_indices].scatter_(
            1, active_selected_positions.unsqueeze(1), sampled_token_ids
        )

    # Final decoding
    output_sequences = [
        tokenizer.decode(ids, skip_special_tokens=True) for ids in batch_input_ids
    ]
    return output_sequences


def generate_cds_sequence_protein_batch_vectorized(
    target_protein_seqs: List[str],
    model: BertForMaskedLM,
    tokenizer: PreTrainedTokenizerFast,
    device: torch.device = "cpu",
    temperature: float = 1.0,
    top_p: float = 0.9,
    class_id: int = 0
):
    """
    (Vectorized) Generates CDS sequences corresponding to a batch of
    target protein sequences (can have variable lengths).
    Uses padding and attention masks.
    """
    model.eval()
    batch_size = len(target_protein_seqs)

    # --- 1. Handle variable lengths and padding ---
    original_lengths = [len(s) for s in target_protein_seqs]
    max_len = max(original_lengths)
    
    # Check max length against tokenizer limit
    effective_max_len = min(max_len, tokenizer.model_max_length)
    if max_len > tokenizer.model_max_length:
        print(f"Warning: Max protein length ({max_len}) exceeds tokenizer max length "
              f"({tokenizer.model_max_length}). Sequences will be truncated to {effective_max_len}.")
        original_lengths = [min(l, effective_max_len) for l in original_lengths]
        target_protein_seqs = [s[:effective_max_len] for s in target_protein_seqs]
        max_len = effective_max_len

    protein_pad_char = 'X' # Define a padding character for proteins
    if protein_pad_char not in AA_TO_CODONS:
        AA_TO_CODONS[protein_pad_char] = [] # Ensure it maps to no codons

    # Pad protein sequences to max_len
    padded_protein_seqs = [s.ljust(max_len, protein_pad_char) for s in target_protein_seqs]

    print(f"\n--- (Vectorized Protein Constraint) Generating {batch_size} sequences (Max Length: {max_len}) ---")

    # --- 2. Create batch-level protein constraint tensor ---
    unique_aas = sorted(list(AA_TO_CODONS.keys()))
    aa_to_idx = {aa: i for i, aa in enumerate(unique_aas)}

    # Matrix: rows=amino_acids, cols=vocab_ids. 1 if codon maps to AA, 0 otherwise.
    aa_constraint_matrix = torch.zeros((len(unique_aas), tokenizer.vocab_size), dtype=torch.float, device=device)
    for aa, codons in AA_TO_CODONS.items():
        aa_idx = aa_to_idx.get(aa)
        if aa_idx is None: continue
        codon_ids = [tokenizer.convert_tokens_to_ids(c) for c in codons if c in tokenizer.vocab]
        if codon_ids:
            # Use index_fill_ for potentially faster sparse updates
            aa_constraint_matrix[aa_idx].index_fill_(0, torch.tensor(codon_ids, device=device, dtype=torch.long), 1.0)


    # Tensor mapping each position in the batch to its required AA index
    protein_aa_indices = torch.tensor(
        [[aa_to_idx.get(aa, aa_to_idx[protein_pad_char]) for aa in protein_seq] # Use pad char index if AA not found
         for protein_seq in padded_protein_seqs],
        dtype=torch.long, device=device
    ) # Shape: (B, S)

    # Batch constraint mask: 1 if codon is valid for the AA at that position, 0 otherwise
    batch_constraint_mask = aa_constraint_matrix[protein_aa_indices] # Shape: (B, S, V)

    # --- 3. Initialize sequences and attention mask ---
    batch_input_ids = torch.full((batch_size, max_len), tokenizer.mask_token_id, dtype=torch.long, device=device)
    
    # Attention mask ignores padding positions
    attention_mask = torch.arange(max_len, device=device)[None, :] < torch.tensor(original_lengths, device=device)[:, None]

    # Set start/stop codons based on original lengths
    stop_codon_ids = [tokenizer.convert_tokens_to_ids(c) for c in STOP_CODONS if c in tokenizer.vocab]
    for i in range(batch_size):
        length = original_lengths[i]
        batch_input_ids[i, 0] = tokenizer.convert_tokens_to_ids("AUG")
        if length > 1:
            batch_input_ids[i, length - 1] = random.choice(stop_codon_ids)

    # Create mask to forbid stop codons during sampling (used for selected positions)
    forbidden_stop_mask = torch.zeros(tokenizer.vocab_size, dtype=torch.float, device=device)
    for fid in stop_codon_ids:
        if fid is not None:
            forbidden_stop_mask[fid] = -float('inf')


    # --- 4. Main Generation Loop ---
    num_iterations = max_len - 2
    for step in tqdm(range(num_iterations), desc="Generating (Vectorized Protein)"):
        # Prepare batch input, including the attention_mask
        token_type_ids = torch.full_like(batch_input_ids, fill_value=class_id)

        with torch.no_grad():
            outputs = model(input_ids=batch_input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            batch_logits = outputs.logits # (B, S, V)

        # --- 5. Vectorized Decision Making ---

        # Apply AA constraint: Invalid codons get -inf logit
        batch_logits.masked_fill_(batch_constraint_mask == 0, -float('inf'))

        # Calculate probabilities and entropy
        batch_probs = torch.softmax(batch_logits / temperature, dim=-1) # (B, S, V)
        entropies = -torch.sum(batch_probs * torch.log(batch_probs + 1e-8), dim=-1) # (B, S)

        # Mask entropies: only consider [MASK] tokens within the true sequence length
        is_mask = (batch_input_ids == tokenizer.mask_token_id)
        selectable_mask = is_mask & attention_mask # Must be mask AND within original length
        masked_entropies = entropies.masked_fill(~selectable_mask, -1.0)

        # Find position with max entropy for each sequence
        selected_positions = torch.argmax(masked_entropies, dim=1) # (B,)

        # Check if any sequences are active (have remaining masks)
        active_mask = (masked_entropies.max(dim=1).values > -1.0) # (B,) boolean
        if not active_mask.any():
            print("All sequences completed generation.")
            break
        active_indices = active_mask.nonzero(as_tuple=True)[0]

        # --- Process only active sequences ---
        active_batch_logits = batch_logits[active_indices] # (ActiveB, S, V)
        active_selected_positions = selected_positions[active_indices] # (ActiveB,)

        # Gather logits for the selected positions
        idx_tensor = active_selected_positions.view(-1, 1, 1).expand(-1, -1, active_batch_logits.shape[-1])
        selected_logits = torch.gather(active_batch_logits, 1, idx_tensor).squeeze(1) # (ActiveB, V)

        # Apply stop codon constraint for internal positions
        selected_logits += forbidden_stop_mask

        # Top-p filtering (vectorized)
        sorted_logits, sorted_indices = torch.sort(selected_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = torch.zeros_like(sorted_indices_to_remove).scatter_(
            1, sorted_indices, sorted_indices_to_remove
        )
        selected_logits.masked_fill_(indices_to_remove, -float('inf'))

        # Sample from the filtered distribution
        final_probs = F.softmax(selected_logits, dim=-1)
        # Handle cases where all valid probabilities become zero
        rows_with_zeros = final_probs.sum(dim=-1) == 0
        if rows_with_zeros.any():
             # Fallback: uniform over allowed codons for that AA at that position
             active_protein_indices = protein_aa_indices[active_indices] # (ActiveB, S)
             selected_aa_indices = torch.gather(active_protein_indices, 1, active_selected_positions.unsqueeze(1)).squeeze(1) # (ActiveB,)
             fallback_constraint = aa_constraint_matrix[selected_aa_indices] # (ActiveB, V)
             fallback_constraint.masked_fill_(forbidden_stop_mask == -float('inf'), 0.0) # Remove stop codons too
             
             uniform_fallback = fallback_constraint / (fallback_constraint.sum(dim=-1, keepdim=True) + 1e-8)
             final_probs[rows_with_zeros] = uniform_fallback[rows_with_zeros]

        sampled_token_ids = torch.multinomial(final_probs, 1) # (ActiveB, 1)

        # --- 6. Vectorized Update ---
        # Update only the active sequences
        current_active_ids = batch_input_ids[active_indices]
        updated_active_ids = current_active_ids.scatter_(
            1, active_selected_positions.unsqueeze(1), sampled_token_ids
        )
        batch_input_ids[active_indices] = updated_active_ids


    # --- 7. Final Decoding (using original lengths) ---
    output_sequences = []
    for i in range(batch_size):
        valid_ids = batch_input_ids[i, :original_lengths[i]] # Slice up to original length
        output_sequences.append(tokenizer.decode(valid_ids, skip_special_tokens=True))

    return output_sequences




# --- Argument Parsing ---

def parse_args():
    parser = argparse.ArgumentParser(description='CodonBERT CDS Generation Script')

    # Paths
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pre-trained BertForMaskedLM model directory')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to the output file to save generated CDS sequences')

    # Generation Mode
    parser.add_argument('--mode', type=str, required=True,
                        choices=[
                            'protein', 'noprotein',
                            'noprotein_batch', 'noprotein_vectorized',
                            'protein_vectorized'
                        ],
                        help='Generation strategy to use')

    # Protein Mode Arguments
    parser.add_argument('--protein_seq', type=str)
    parser.add_argument('--total_proteins', type=int, default=1000)
    

    # No-Protein Mode Arguments
    parser.add_argument('--seq_len', type=int, default=500,
                        help='Fixed length for sequences generated in noprotein modes')
    parser.add_argument('--num_sequences', type=int, default=100,
                        help='Total number of sequences to generate in noprotein modes')

    # Batching and Generation Parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for batch generation modes')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for sampling (higher = more random)')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p (nucleus) sampling threshold')
    parser.add_argument('--class_id', type=int, default=0,
                        help='Class ID for token_type_ids')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use ("cuda", "cpu", or None for auto-detect)')
    parser.add_argument('--model_max_length', type=int, default=512,
                         help='Maximum sequence length the tokenizer/model can handle')


    return parser.parse_args()


# --- Main Execution ---

def main(args):
    """Loads model, data, and runs the selected generation mode."""

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Tokenizer
    tokenizer = build_codon_tokenizer(model_max_length=args.model_max_length)

    # Load Model
    print(f"Loading model from: {args.model_path}")
    model = BertForMaskedLM.from_pretrained(args.model_path).to(device)
    model.eval() # Set model to evaluation mode

    # --- Execute selected generation mode ---

    if args.mode == 'protein_vectorized':
        protein_seq = args.protein_seq

        total_proteins = args.total_proteins
        generated_count = 0

        with open(args.output_file, 'w') as f: # Use 'w' to overwrite or 'a' to append
            for i in range(0, total_proteins, args.batch_size):
                batch_proteins = [protein_seq] * args.batch_size
                if not batch_proteins: continue

                print(f"\nProcessing batch {i // args.batch_size + 1}...")
                generated_batch = generate_cds_sequence_protein_batch_vectorized(
                    target_protein_seqs=batch_proteins,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    class_id=args.class_id
                )
                # Write results for this batch
                for prot, cds in zip(batch_proteins, generated_batch):
                    f.write(f">protein_len={len(prot)}\n") # Example header
                    f.write(f"{cds}\n")
                generated_count += len(generated_batch)
                print(f"Written {len(generated_batch)} sequences. Total generated: {generated_count}/{total_proteins}")

    elif args.mode == 'noprotein_vectorized':
        total_to_generate = args.num_sequences
        generated_count = 0
        
        with open(args.output_file, 'w') as f:
            print(f"Generating {total_to_generate} sequences using vectorized no-protein mode...")
            # Generate in batches if total is large
            while generated_count < total_to_generate:
                current_batch_size = min(args.batch_size, total_to_generate - generated_count)
                if current_batch_size <= 0: break
                
                generated_batch = generate_cds_sequence_batch_vectorized(
                    num_sequences=current_batch_size,
                    seq_len=args.seq_len,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    class_id=args.class_id
                )
                for cds in generated_batch:
                        f.write(f"{cds}\n")
                generated_count += len(generated_batch)
                print(f"Generated {len(generated_batch)} sequences. Total: {generated_count}/{total_to_generate}")

    print(f"\nGeneration complete. Results saved to {args.output_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args)


