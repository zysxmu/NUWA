#!/usr/bin/env python3
"""
Finetunes a pre-trained CodonBERT model for a REGRESSION task 
(e.g., predicting mRNA stability or translation efficiency)
using a Hugging Face Trainer.
"""

import os
import torch
import pandas as pd
import numpy as np
import argparse
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    BertModel,
    BertConfig,
    TrainingArguments,
    PreTrainedTokenizerFast,
    Trainer,
    PreTrainedModel
)
from scipy.stats import spearmanr
from typing import Dict, List, Any


# --- Tokenizer ---
def build_codon_tokenizer(tokenizer_path=None) -> PreTrainedTokenizerFast:
    """Builds or loads a 64-codon WordLevel tokenizer."""
    if tokenizer_path:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            bos_token="[CLS]",
            eos_token="[SEP]",
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
        )
    else:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.processors import BertProcessing

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
            mask_token='[MASK]'
        )
    return tokenizer


# --- Encoding Function ---
def encode_string(example: Dict, tokenizer: PreTrainedTokenizerFast, max_length: int) -> Dict:
    """Tokenizes a single data example."""
    encodings = tokenizer(
        example['seq'],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_special_tokens_mask=True
    )
    encodings['token_type_ids'] = [example['species_class']] * max_length
    encodings['species_id'] = example['species_id']
    return encodings


# --- Dataset ---
class RNACSVDataset(Dataset):
    """
    Dataset class to load RNA sequences, convert to 3-mers,
    and store tokenized examples.
    """
    def __init__(self, sequences: List[str], labels: List[float], 
                 tokenizer: PreTrainedTokenizerFast, max_length: int, species_id: int = 0):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.species_id = species_id

        for seq, label in zip(sequences, labels):
            example = {
                'seq': self.split_into_3mers(seq),
                'species_id': self.species_id,
                'species_class': self.species_id
            }
            encoded = encode_string(example, self.tokenizer, self.max_length)
            encoded['labels'] = label
            self.examples.append(encoded)

    def split_into_3mers(self, seq: str) -> str:
        """Splits 'AUGC...' into 'AUG C...'"""
        grouped = [seq[i:i+3] for i in range(0, len(seq), 3)]
        grouped = [kmer for kmer in grouped if len(kmer) == 3]  # Keep only full codons
        return " ".join(grouped)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# --- Regression Model ---
class BertForRegressionHF(PreTrainedModel):
    """
    Custom BERT model for regression.
    Loads a pre-trained encoder and adds a single linear regression head.
    """
    config_class = BertConfig

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)  # ✅ Use the BERT encoder only
        self.regressor = nn.Linear(config.hidden_size, 1)
        self.post_init()  # Must be called!

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        species_id=None,  # Unused, but kept for signature consistency
        labels=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Use pooled output (from [CLS] token)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        logits = self.regressor(pooled_output).squeeze(-1) # [batch_size]

        loss = None
        if labels is not None:
            # Use Mean Squared Error for regression
            loss = nn.functional.mse_loss(logits, labels)
            
        return {"loss": loss, "logits": logits}

    @classmethod
    def from_pretrained_encoder(cls, pretrained_path, *model_args, **kwargs):
        """Loads a pre-trained BERT *encoder* and initializes the new regression head."""
        # Load config
        config = BertConfig.from_pretrained(pretrained_path)

        # Initialize this model class
        model = cls(config, *model_args, **kwargs)

        # Load weights for the BERT encoder part
        model.bert = BertModel.from_pretrained(pretrained_path)

        return model


# --- Metrics ---
def compute_metrics(eval_pred):
    """Calculates MSE and Spearman correlation for regression."""
    predictions, labels = eval_pred
    mse = np.mean((predictions - labels) ** 2)
    # Handle potential NaNs in predictions if model collapses
    if np.isnan(predictions).any():
        spearman = 0.0
    else:
        spearman = spearmanr(predictions, labels).correlation
        
    return {
        "mse": mse,
        "spearman": spearman
    }


# --- Custom Trainer ---
class RegressionLoggingTrainer(Trainer):
    """
    Custom trainer to ensure 'total_loss' (MSE) is logged.
    (Renamed from SupConLoggingTrainer for clarity).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_logs = {}

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        outputs = model(**inputs)
        loss = outputs["loss"]

        self._loss_logs = {
            'total_loss': loss.detach().item()
        }

        self.accelerator.backward(loss)
        return loss.detach()

    def log(self, logs: Dict[str, float]) -> None:
        """Logs metrics, including the custom loss."""
        logs.update(self._loss_logs)
        super().log(logs)


# --- Data Collator ---
def data_collator(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Custom data collator.
    Converts labels to float32 for MSE loss.
    """
    return {
        key: torch.tensor(
            [f[key] for f in features], 
            dtype=torch.float32 if key == "labels" else torch.long
        )
        for key in features[0]
    }


# --- Main Function ---
def main(args):
    """Main training and evaluation function."""
    
    # 1. Build Tokenizer
    bert_tokenizer_fast = build_codon_tokenizer(args.tokenizer_path)

    # 2. Load Model
    model = BertForRegressionHF.from_pretrained_encoder(args.pretrain_model_path)
    

    # 3. Load and Prepare Datasets
    df = pd.read_csv(args.train_data_path)
    train_df = df[df['Split'] == 'train']
    val_df = df[df['Split'] == 'val']
    test_df = df[df['Split'] == 'test']

    train_dataset = RNACSVDataset(
        train_df['Sequence'].tolist(), train_df['Value'].tolist(),
        tokenizer=bert_tokenizer_fast, max_length=args.max_length, species_id=args.species_id
    )
    val_dataset = RNACSVDataset(
        val_df['Sequence'].tolist(), val_df['Value'].tolist(),
        tokenizer=bert_tokenizer_fast, max_length=args.max_length, species_id=args.species_id
    )
    test_dataset = RNACSVDataset(
        test_df['Sequence'].tolist(), test_df['Value'].tolist(),
        tokenizer=bert_tokenizer_fast, max_length=args.max_length, species_id=args.species_id
    )

    # 4. Define Training Arguments
    training_args = TrainingArguments(
        optim='adamw_torch',
        output_dir=args.checkpoint_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 100, # Larger eval batch size
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        num_train_epochs=args.max_epochs,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=args.log_steps,
        fp16=True,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=f"val_{args.metric_for_best_model}", # Add "val" prefix
        greater_is_better=args.greater_is_better,
        ddp_backend='nccl' if int(os.environ.get("LOCAL_RANK", -1)) != -1 else None,
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_to=[],
    )

    # 5. Create Datasets Dictionary for Evaluation
    eval_datasets = {
        "val": val_dataset,
        "test": test_dataset
    }

    # 6. Initialize Trainer
    trainer = RegressionLoggingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets, # Pass the dictionary here
        tokenizer=bert_tokenizer_fast,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 7. Train and Evaluate
    print("🚀 Starting training...")
    trainer.train()
    
    print("✅ Training complete.")
    print("🔍 Evaluating on test set (using best model)...")
    
    # Evaluate on the 'test' split specifically
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="best_test")
    print(test_metrics)


# --- Argparse ---
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Finetune CodonBERT for Regression")
    
    # Paths
    parser.add_argument("--tokenizer_path", type=str, default=None, 
                        help="Path to tokenizer JSON file (if not building from scratch)")
    parser.add_argument("--pretrain_model_path", type=str, required=True, 
                        help="Path to pretrained BERT encoder checkpoint")
    parser.add_argument("--train_data_path", type=str, required=True, 
                        help="CSV file with Sequence, Value, Split columns")
    parser.add_argument("--checkpoint_dir", type=str, default="./finetune_checkpoints", 
                        help="Directory to save checkpoints")
    
    # Data & Model Config
    parser.add_argument("--max_length", type=int, default=1024, 
                        help="Maximum sequence length")
    parser.add_argument("--species_id", type=int, default=0, 
                        help="Species ID to use for token_type_ids")

    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Per-device training batch size")
    parser.add_argument("--max_epochs", type=int, default=10, 
                        help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=1000, 
                        help="Total number of training steps (overrides epochs)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                        help="Number of steps to accumulate gradients")

    # Evaluation & Saving
    parser.add_argument("--log_steps", type=int, default=10, 
                        help="Log every N steps")
    parser.add_argument("--eval_steps", type=int, default=10, 
                        help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=10, 
                        help="Save checkpoint every N steps")
    parser.add_argument("--load_best_model_at_end", action=argparse.BooleanOptionalAction, default=True,
                        help="Load the best model at the end of training")
    parser.add_argument("--metric_for_best_model", type=str, default="val_spearman",
                        help="Metric to monitor (e.g., 'spearman' or 'mse')")
    parser.add_argument("--greater_is_better", action=argparse.BooleanOptionalAction, default=True,
                        help="Set to True if a higher metric is better (e.g., Spearman)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)




    