#!/usr/bin/env python3
"""
Finetunes a pre-trained CodonBERT model for the 'mRNA Stability'
regression task using Hugging Face Trainer and PEFT (LoRA).
"""

import os
import math
import random
import argparse
import numpy as np
import pandas as pd
import json
from functools import partial

from Bio import SeqIO
from datasets import Dataset
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

import torch
from torch.utils.data import DataLoader

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import BertProcessing

from transformers import (
    TrainingArguments, 
    Trainer, 
    PreTrainedTokenizerFast, 
    BertForSequenceClassification
)

# --- Argument Processing ---
parser = argparse.ArgumentParser(description='CodonBERT Stability Finetuning')

# Paths and Task
parser.add_argument('--pretrain', '-p', default='codonbert_models/checkpoint-1/', 
                    type=str, help='Folder of the pretrained model')
parser.add_argument('--model_dir', type=str, default=None, 
                    help='Directory to save finetuned checkpoints')

# Training Hyperparameters
parser.add_argument('--lr', type=float, default=0.00005, 
                    help='Learning rate')
parser.add_argument('--batch', '-b', type=int, default=128, 
                    help='Batch size')
parser.add_argument('--epochs', '-e', type=int, default=200, 
                    help='Number of training epochs')
parser.add_argument('--eval_step', '-s', type=int, default=1, 
                    help='Number of training epochs between evaluations')
parser.add_argument('--max_length', type=int, default=512, 
                    help='Maximum sequence length in tokens')
parser.add_argument('--random', '-r', type=int, default=42, 
                    help='Random seed')

# LoRA (PEFT) Configuration
parser.add_argument('--lora', action='store_false', 
                    help='Disable LoRA and use full finetuning. (LoRA is ON by default)')
parser.add_argument('--lorar', type=int, default=32, 
                    help='Lora rank (r)')
parser.add_argument('--lalpha', type=int, default=32, 
                    help='Lora alpha')
parser.add_argument('--ldropout', type=float, default=0.1, 
                    help='Lora dropout')
args = parser.parse_args()

# --- PEFT (LoRA) Import ---
if args.lora:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        print("PEFT (LoRA) is enabled.")
    except ImportError:
        print("PEFT not installed. Please install with 'pip install peft'")
        args.lora = False
else:
    print("PEFAT (LoRA) is disabled (full finetuning).")


# --- Task Configuration (Fixed to 'mRNA Stability') ---
task_name = "mRNA Stability"
data_path = "downstream_data/mRNA_Stability.csv"
num_of_labels = 1  # This is a regression task
print(f"Task fixed to: {task_name} (Regression)")

# --- Setup Training Parameters ---
lr = args.lr
bs_train = args.batch
bs_test = args.batch
log_steps = args.eval_step
save_steps = args.eval_step
num_epoches = args.epochs
max_length = args.max_length

# Set model directory
model_dir = args.model_dir
if model_dir is None:
    model_dir = '%s_models' % task_name.replace(" ", "-")
os.makedirs(model_dir, exist_ok=True)
print(f"Model checkpoints will be saved to: {model_dir}")

# --- Default Model Hyperparameters (Do not change) ---
# These are inherent to the pretrained model architecture
hidden_size = 768
inter_size = 3072
num_heads = 12
num_layers = 12


# --- Data Loading and Processing ---

def mytok(seq, kmer_len, s):
    """Tokenizes a sequence into k-mers."""
    seq = seq.upper().replace("T", "U")
    kmer_list = []
    for j in range(0, (len(seq) - kmer_len) + 1, s):
        kmer_list.append(seq[j:j+kmer_len])
    return kmer_list

def load_data_from_csv(data_path, split):
    """Loads data from the CSV file for a specific split."""
    seqs = []
    ys = []
    skipped = 0

    df = pd.read_csv(data_path)
    # Filter by task name (e.g., "mRNA Stability")
    df = df.loc[df['Dataset'] == task_name]
    # Filter by split (train, val, test)
    df = df.loc[df['Split'] == split]

    raw_seqs = df["Sequence"]
    raw_ys = df["Value"]
    total = len(raw_seqs)

    for seq, y in zip(raw_seqs, raw_ys):
        # Tokenize into 3-mers (codons)
        lst_tok = mytok(seq, 3, 3)
        # Truncate to max_length (accounting for [CLS] and [SEP])
        lst_tok = lst_tok[:max_length - 2]
        
        if lst_tok:
            # This check is redundant due to the slice, but kept from original
            if len(lst_tok) > max_length - 2:
                skipped += 1
                print(f"Skip one sequence with length {len(lst_tok)} codons. "
                      f"Skipped {skipped} seqs from total {total} seqs.")
                continue
            
            seqs.append(" ".join(lst_tok))
            
            # This is a regression task, so store labels as float
            if num_of_labels > 1:
                ys.append(int(float(y)))
            else:
                ys.append(float(y))
                
    return seqs, ys

def build_dataset(data_path):
    """Builds train, eval, and test datasets."""
    X_train, y_train = load_data_from_csv(data_path, "train")
    X_eval, y_eval = load_data_from_csv(data_path, "val")
    X_test, y_test = load_data_from_csv(data_path, "test")

    print(f"Data size: Train={len(X_train)}, Eval={len(X_eval)}, Test={len(X_test)}")

    ds_train = Dataset.from_list([{"seq": seq, "label": y} for seq, y in zip(X_train, y_train)])
    ds_eval = Dataset.from_list([{"seq": seq, "label": y} for seq, y in zip(X_eval, y_eval)])
    ds_test = Dataset.from_list([{"seq": seq, "label": y} for seq, y in zip(X_test, y_test)])

    return ds_train, ds_eval, ds_test


# --- Vocabulary & Tokenizer ---
lst_ele = list('AUGC')
lst_voc = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
for a1 in lst_ele:
    for a2 in lst_ele:
        for a3 in lst_ele:
            lst_voc.extend([f'{a1}{a2}{a3}'])

dic_voc = dict(zip(lst_voc, range(len(lst_voc))))
print(f"Tokenizer vocabulary size: {len(dic_voc)}")

tokenizer_obj = Tokenizer(WordLevel(vocab=dic_voc, unk_token="[UNK]"))
tokenizer_obj.add_special_tokens(['[PAD]', '[CLS]', '[UNK]', '[SEP]', '[MASK]'])
tokenizer_obj.pre_tokenizer = Whitespace()
tokenizer_obj.post_processor = BertProcessing(
    ("[SEP]", dic_voc['[SEP]']),
    ("[CLS]", dic_voc['[CLS]']),
)

bert_tokenizer_fast = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer_obj,
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

def encode_string(data, tokenizer, max_length):
    """Encodes a batch of sequences."""
    return tokenizer(
        data['seq'],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_special_tokens_mask=True
    )

# --- Loading Dataset ---
ds_train, ds_eval, ds_test = build_dataset(data_path)

# Use partial to pass fixed arguments (tokenizer, max_length) to the map function
encode_fn = partial(encode_string, 
                    tokenizer=bert_tokenizer_fast, 
                    max_length=max_length)

train_loader = ds_train.map(encode_fn, batched=True)
eval_loader = ds_eval.map(encode_fn, batched=True)
test_loader = ds_test.map(encode_fn, batched=True)


# --- Loading Pretrained Model ---
model = BertForSequenceClassification.from_pretrained(
    args.pretrain, 
    num_labels=num_of_labels
)
print(f"Loading model from {args.pretrain} succesfully...")


# --- LoRA (PEFT) Setup ---
if args.lora:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lorar,
        lora_alpha=args.lalpha,
        lora_dropout=args.ldropout,
        use_rslora=True,
        # Save the classifier and pooler layers to be trained
        modules_to_save=["classifier", "pooler"] 
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()


metric_to_monitor = "spearmanr"
metric_for_best = f"eval_val_{metric_to_monitor}"
print(f"Monitoring metric for best model: {metric_for_best}")

# --- Training Settings & Metrics ---
training_args = TrainingArguments(
    optim='adamw_torch',
    learning_rate=lr,
    output_dir=model_dir,
    eval_strategy="epoch",
    overwrite_output_dir=True,
    num_train_epochs=num_epoches,
    per_device_train_batch_size=bs_train,
    per_device_eval_batch_size=bs_test,
    save_strategy="epoch",
    save_steps=save_steps,
    eval_steps=log_steps, # eval_steps is used by Trainer, log_steps is just an alias here
    load_best_model_at_end=True,
    metric_for_best_model=metric_for_best,
    greater_is_better=True,
    save_total_limit=5,
    report_to=[],
    fp16=True,
)

def compute_metrics(eval_pred):
    """Computes Pearson and Spearman correlation for regression."""
    logits, labels = eval_pred
    
    # This is a regression task (num_of_labels == 1)
    logits = logits.flatten()
    labels = labels.flatten()

    try:
        pearson_corr = pearsonr(logits, labels)[0]
        spearman_corr = spearmanr(logits, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
        }
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {"pearson": 0.0, "spearmanr": 0.0}

eval_datasets = {
    "val": eval_loader,  # Use the validation dataset with key "val"
    "test": test_loader, # Use the test dataset with key "test"
}

print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_loader,
    # NOTE: Original logic uses the TEST set as the evaluation set.
    # This will select the best model based on test performance.
    # For rigorous results, change this to `eval_loader`.
    eval_dataset=eval_datasets,
    compute_metrics=compute_metrics
)

# --- Training & Evaluation ---
print("Starting training...")
trainer.train()

# Evaluate the model (will use the best model loaded)
print("Training finished. Evaluating best model...")
metrics = trainer.evaluate()
print("Evaluation Metrics (on test set):")
print(metrics)

# Prediction on test set
print("Generating predictions on test set...")
pred, _, metrics = trainer.predict(test_loader)
print("Prediction Metrics (on test set):")
print(metrics)

print("Script finished.")



