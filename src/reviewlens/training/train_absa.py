"""Fine-tune the (sentence, aspect) -> polarity classifier on SemEval-2014 train.

    python scripts/train_absa.py             # config defaults
    python scripts/train_absa.py --epochs 3

Unlike the pretrained checkpoint (whose training mix includes SemEval-2014
train data *and* other corpora), this model sees only the official train
splits — its test numbers are a clean, defensible measurement.

Saves to ``sentiment.absa_finetuned_dir`` (default ``models/absa-classifier``,
git-ignored). The saved model is a drop-in for
:class:`reviewlens.sentiment.transformer_absa.TransformerAspectSentiment`.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from reviewlens.config import load_config, resolve_path
from reviewlens.training.examples import SENTIMENT_ID2LABEL, SENTIMENT_LABEL2ID


def _compute_metrics(eval_pred: Any) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    tcfg = cfg["training"]

    parser = argparse.ArgumentParser(description="Fine-tune the ABSA polarity classifier.")
    parser.add_argument("--base-model", default=tcfg["base_model"])
    parser.add_argument("--epochs", type=int, default=tcfg["absa"]["epochs"])
    parser.add_argument("--batch-size", type=int, default=tcfg["absa"]["batch_size"])
    parser.add_argument("--lr", type=float, default=tcfg["absa"]["learning_rate"])
    parser.add_argument("--max-length", type=int, default=tcfg["max_length"])
    parser.add_argument("--seed", type=int, default=tcfg["seed"])
    parser.add_argument(
        "--output-dir", default=None, help="Defaults to sentiment.absa_finetuned_dir."
    )
    parser.add_argument(
        "--bf16", action="store_true", help="Train in bf16 (default is fp32)."
    )
    args = parser.parse_args(argv)

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        set_seed,
    )

    from reviewlens.training.common import ListDataset, describe_device, training_arguments

    set_seed(args.seed)
    output_dir = resolve_path(args.output_dir or cfg["sentiment"]["absa_finetuned_dir"])
    semeval_dir = resolve_path(cfg["eval"]["semeval_dir"])

    print(f"Base model : {args.base_model}")
    print(f"Device     : {describe_device()}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    from reviewlens.training.examples import build_absa_examples

    train_examples, dev_examples = build_absa_examples(
        tokenizer, semeval_dir, max_length=args.max_length, seed=args.seed
    )
    print(f"Examples   : {len(train_examples)} train / {len(dev_examples)} dev pairs")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(SENTIMENT_LABEL2ID),
        id2label=SENTIMENT_ID2LABEL,
        label2id=SENTIMENT_LABEL2ID,
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
    print(f"Dev accuracy={final['eval_accuracy']:.4f} macro-F1={final['eval_macro_f1']:.4f}")
    print(f"Saved to   : {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
