"""
t5_pretrain.py

This script defines the dataset, preprocessing, training utilities, and evaluation
functions used to pretrain a T5 model on a disaster-related dataset.

Author: Your Name
Date: YYYY-MM-DD
"""

import argparse
import glob
import os
import json
import time
import logging
import random
import re
import math
from itertools import chain
from string import punctuation
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import (
    AdamW,
    Adafactor,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
from tqdm.auto import tqdm

# -----------------------------
# Utility Functions
# -----------------------------

def set_seed(seed):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_answer(s):
    """
    Normalize text by removing articles, punctuation, extra whitespace, and lowercasing.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    """
    Compute exact match score (normalized).
    """
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def approx_match_score(prediction, ground_truth):
    """
    Compute approximate score based on word overlap.
    """
    answer = normalize_answer(prediction)
    gt = normalize_answer(ground_truth)
    return int(any(word in answer for word in gt.split(" ")))


def calculate_scores(predictions, ground_truths):
    """
    Compute average exact match and approximate match scores.
    """
    em_score = sum(exact_match_score(pred, gt) for pred, gt in zip(predictions, ground_truths)) / len(predictions)
    subset_match_score = sum(approx_match_score(pred, gt) for pred, gt in zip(predictions, ground_truths)) / len(predictions)
    return em_score * 100, subset_match_score * 100

# -----------------------------
# Dataset Definition
# -----------------------------

class PretrainDataset(Dataset):
    """
    Custom Dataset for T5 pretraining using span corruption.
    """
    def __init__(self, tokenizer, file_path, input_length=512, output_length=150):
        self.dataset = self.load_and_segment_data(file_path, input_length)
        self.tokenizer = tokenizer
        self.input_length = input_length
        self.output_length = output_length

    def load_and_segment_data(self, file_path, input_length):
        """
        Load CSV file and segment long contexts into smaller chunks.
        """
        df = pd.read_csv(file_path)
        segments = []

        for _, row in df.iterrows():
            words = row['context'].split()
            while len(words) > input_length:
                seg, words = words[:input_length], words[input_length:]
                segments.append(" ".join(seg))
            if words:
                segments.append(" ".join(words))

        return pd.DataFrame(segments, columns=['context'])

    def __len__(self):
        return len(self.dataset)

    def span_corruption_mask(self, text, noise_density=0.15, span_length=3):
        """
        Generate binary mask for span corruption with sentinel replacement.
        """
        words = text.split()
        mask = [0] * len(words)
        num_spans = math.ceil((len(words) * noise_density) / span_length)
        masked_indices = set()

        for _ in range(num_spans):
            start_idx = random.choice([i for i in range(len(words)) if i not in masked_indices])
            for j in range(span_length):
                if start_idx + j < len(words):
                    masked_indices.add(start_idx + j)
                    mask[start_idx + j] = 1

        return mask

    def apply_sentinels(self, text, mask):
        """
        Replace masked spans with <extra_id_X> tokens for T5 format.
        """
        words = text.split()
        sentinels = [f'<extra_id_{i}>' for i in range(100)]
        sentinel_idx = 0
        input_text, target_text = [], []

        for i, word in enumerate(words):
            if mask[i] == 1:
                if not target_text or target_text[-1] != sentinels[sentinel_idx]:
                    target_text.append(sentinels[sentinel_idx])
                target_text.append(word)
            else:
                if not input_text or input_text[-1] != sentinels[sentinel_idx]:
                    input_text.append(sentinels[sentinel_idx])
                    sentinel_idx += 1
                input_text.append(word)

        return " ".join(input_text), " ".join(target_text)

    def __getitem__(self, index):
        """
        Return tokenized input and target for the given index.
        """
        text = self.dataset.iloc[index]['context']
        mask = self.span_corruption_mask(text)
        input_text, target_text = self.apply_sentinels(text, mask)

        source = self.tokenizer(
            input_text, max_length=self.input_length, padding='max_length', truncation=True, return_tensors='pt'
        )
        target = self.tokenizer(
            target_text, max_length=self.output_length, padding='max_length', truncation=True, return_tensors='pt'
        )

        return {
            "input_ids": source['input_ids'].squeeze(),
            "attention_mask": source['attention_mask'].squeeze(),
            "target_ids": target['input_ids'].squeeze(),
            "target_mask": target['attention_mask'].squeeze()
        }

# -----------------------------
# Logging Callback
# -----------------------------

logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):
    """
    PyTorch Lightning callback for logging validation and test results.
    """
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info(f"{key} = {metrics[key]}")

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info(f"{key} = {metrics[key]}")
                        writer.write(f"{key} = {metrics[key]}\n")
