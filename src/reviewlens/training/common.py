"""Shared training utilities. Imports torch — training-time modules only."""

from __future__ import annotations

from pathlib import Path

import torch


class ListDataset(torch.utils.data.Dataset):
    """Minimal Dataset over pre-encoded example dicts (padded to max_length)."""

    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def training_arguments(
    checkpoint_dir: str | Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    bf16: bool = False,
):
    """Standard fine-tuning arguments; no intermediate checkpoints.

    fp32 by default — at this scale (base-size encoder, seq 128) it is fast
    enough that the numerical-safety trade is free; pass ``bf16=True`` to opt
    in. Note the historical trap this default guards against: with Trainer's
    gradient clipping, a numerically broken backward (e.g. deberta-v3 on
    transformers 5.6.x, which NaNs regardless of precision or device) doesn't
    crash — it silently trains only the classifier head to the class priors
    and reports dev F1 = 0. If metrics flatline at zero, probe gradients with
    a manual loop before blaming hyperparameters.
    """
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        seed=seed,
        bf16=bf16 and torch.cuda.is_available(),
        report_to=[],
    )


def describe_device() -> str:
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return "cpu (training will be slow — a CUDA build of torch is recommended)"
