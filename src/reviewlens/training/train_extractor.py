"""Fine-tune the BIO aspect-term extractor on SemEval-2014 train splits.

    python scripts/train_extractor.py            # config defaults
    python scripts/train_extractor.py --epochs 3

Saves the model + tokenizer to ``aspects.transformer_model_dir`` (default
``models/aspect-extractor``, git-ignored). Dev metrics printed per epoch are
span-level seqeval P/R/F1 on a held-out 10% of train; official test numbers
come from ``scripts/evaluate_semeval.py``.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from reviewlens.config import load_config, resolve_path
from reviewlens.training.bio import ID2LABEL, IGNORE_INDEX, LABEL2ID


def _compute_metrics(eval_pred: Any) -> dict[str, float]:
    import numpy as np
    from seqeval.metrics import f1_score, precision_score, recall_score

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    true_seqs: list[list[str]] = []
    pred_seqs: list[list[str]] = []
    for pred_row, label_row in zip(predictions, labels, strict=True):
        true_seq, pred_seq = [], []
        for pred, label in zip(pred_row, label_row, strict=True):
            if label == IGNORE_INDEX:
                continue
            true_seq.append(ID2LABEL[int(label)])
            pred_seq.append(ID2LABEL[int(pred)])
        true_seqs.append(true_seq)
        pred_seqs.append(pred_seq)

    return {
        "precision": precision_score(true_seqs, pred_seqs),
        "recall": recall_score(true_seqs, pred_seqs),
        "f1": f1_score(true_seqs, pred_seqs),
    }


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    tcfg = cfg["training"]

    parser = argparse.ArgumentParser(description="Fine-tune the BIO aspect extractor.")
    parser.add_argument("--base-model", default=tcfg["base_model"])
    parser.add_argument("--epochs", type=int, default=tcfg["extractor"]["epochs"])
    parser.add_argument("--batch-size", type=int, default=tcfg["extractor"]["batch_size"])
    parser.add_argument("--lr", type=float, default=tcfg["extractor"]["learning_rate"])
    parser.add_argument("--max-length", type=int, default=tcfg["max_length"])
    parser.add_argument("--seed", type=int, default=tcfg["seed"])
    parser.add_argument(
        "--output-dir", default=None, help="Defaults to aspects.transformer_model_dir."
    )
    parser.add_argument(
        "--bf16", action="store_true", help="Train in bf16 (default is fp32)."
    )
    args = parser.parse_args(argv)

    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        Trainer,
        set_seed,
    )

    from reviewlens.training.common import ListDataset, describe_device, training_arguments

    set_seed(args.seed)
    output_dir = resolve_path(args.output_dir or cfg["aspects"]["transformer_model_dir"])
    semeval_dir = resolve_path(cfg["eval"]["semeval_dir"])

    print(f"Base model : {args.base_model}")
    print(f"Device     : {describe_device()}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    from reviewlens.training.examples import build_extraction_examples

    train_examples, dev_examples = build_extraction_examples(
        tokenizer, semeval_dir, max_length=args.max_length, seed=args.seed
    )
    print(f"Examples   : {len(train_examples)} train / {len(dev_examples)} dev sentences")

    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model, num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID
    )

    trainer = Trainer(
        model=model,
        args=training_arguments(
            output_dir / "checkpoints", args.epochs, args.batch_size, args.lr, args.seed,
            bf16=args.bf16,
        ),
        train_dataset=ListDataset(train_examples),
        eval_dataset=ListDataset(dev_examples),
        compute_metrics=_compute_metrics,
        processing_class=tokenizer,
    )

    started = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - started

    final = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"\nTraining took {elapsed / 60:.1f} min")
    print(
        f"Dev span-F1: {final['eval_f1']:.4f} "
        f"(P={final['eval_precision']:.4f} R={final['eval_recall']:.4f})"
    )
    print(f"Saved to   : {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
