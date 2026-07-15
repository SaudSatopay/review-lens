# 🔍 ReviewLens — Aspect-Based Review Intelligence

[![CI](https://github.com/SaudSatopay/review-lens/actions/workflows/ci.yml/badge.svg)](https://github.com/SaudSatopay/review-lens/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**ReviewLens goes beyond one sentiment score per review.** It extracts the
*aspects* people actually talk about (battery, screen, price, service…),
classifies sentiment **per aspect**, discovers aspect **themes** by clustering,
and renders a business dashboard with per-aspect sentiment, trends,
representative quotes, and an (optional) LLM executive summary of what customers
love and hate.

> A single review is rarely all-positive or all-negative:
> *"Great screen. Terrible battery life."*
> A document-level score flattens that into one number. ReviewLens keeps the
> screen **positive** and the battery **negative** — which is what a product
> team actually needs.

---

## Why aspect-based? (the core idea)

| | Document-level (baseline) | Aspect-based (ReviewLens) |
|---|---|---|
| Output | 1 label per review | 1 label **per aspect** per review |
| *"Great screen, terrible battery"* | 🤷 neutral / mixed | ✅ screen **+**, battery **−** |
| Business value | "scores are okay-ish" | "fix the battery, keep the screen" |

The document-level [VADER](https://github.com/cjhutto/vaderSentiment) baseline is
kept in the codebase **on purpose** — it's the benchmark that demonstrates ABSA's
added value.

---

## Pipeline

```
CSV reviews
   │  ingest → clean → sentence-split
   ▼
Sentences ──► Aspect extraction ──►  (baseline: noun-phrase chunking)
   │                                 (next: fine-tuned BERT BIO tagger)
   ▼
(sentence, aspect) ──► Aspect sentiment ──► (baseline: VADER)
   │                                        (next: fine-tuned ABSA transformer)
   ▼
Aspect terms ──► Theme clustering ──► (baseline: keyword/normalized grouping)
   │                                  (next: MiniLM embeddings + KMeans/HDBSCAN)
   ▼
Aggregation ──► per-aspect distribution · trends · quotes
   ▼
Streamlit + Plotly dashboard  (+ optional LLM executive summary)
```

---

## Project structure

```
review-lens/
├── src/reviewlens/
│   ├── config.py            # loads config.yaml
│   ├── nltk_setup.py        # one-shot NLTK data bootstrap
│   ├── data/                # ingest · clean · sentence-split
│   ├── aspects/             # baseline (noun-phrase) + absa (transformer, next)
│   ├── sentiment/           # VADER doc-level baseline + per-aspect sentiment
│   ├── clustering/          # theme grouping (keyword baseline + embeddings, next)
│   ├── aggregate/           # per-aspect summaries, trends, quotes
│   └── pipeline.py          # end-to-end orchestration + CLI
├── app/streamlit_app.py     # dashboard
├── scripts/run_pipeline.py  # CLI wrapper (no install needed)
├── notebooks/               # exploration
├── tests/                   # pytest suite
├── data/sample/             # tiny committed sample so it runs out of the box
├── config.yaml              # paths · model names · thresholds
└── requirements*.txt        # core (light) · full (ML) · dev
```

---

## Quickstart

### 1. Baseline (lightweight — no torch, runs in seconds)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate

pip install -r requirements-core.txt
pip install -e .                     # register the `reviewlens` package
python -c "from reviewlens.nltk_setup import ensure_nltk_data; ensure_nltk_data()"
```

Run the pipeline on the bundled sample:

```bash
python scripts/run_pipeline.py --group-by theme
```

Launch the dashboard:

```bash
streamlit run app/streamlit_app.py
```

### 2. Full stack (transformers, embeddings — larger install)

```bash
pip install -r requirements.txt      # adds torch, transformers, sentence-transformers, …
```

---

## Datasets & models

- **Train / evaluate:** [SemEval-2014 Task 4](https://alt.qcri.org/semeval2014/task4/)
  (Restaurants + Laptops) and/or [MAMS](https://github.com/siat-nlp/MAMS-for-ABSA)
  — gold aspect + polarity labels.
- **Demo data:** a real Amazon/Yelp review set (drop into `data/raw/`).
- **Models (next slices):** fine-tune `deberta-v3-base` / `roberta-base`;
  strong pretrained baseline `yangheng/deberta-v3-base-absa-v1.1`.

## Evaluation

- **Aspect extraction:** span-level **F1** (`seqeval`) — fine-tuned BIO tagger vs
  the noun-phrase baseline.
- **Aspect sentiment:** **macro-F1** on the SemEval test split.
- **ABSA value:** benchmarked against the document-level VADER baseline.

---

## Roadmap (6–8 week solo plan)

- [x] **Slice 0 — Scaffold + runnable baseline** (this commit): ingest → clean →
      split → noun-phrase aspects → VADER sentiment → theme grouping →
      aggregation → dashboard → tests + CI.
- [ ] **Slice 1 — Transformer ABSA:** wire `yangheng/deberta-v3-base-absa-v1.1`
      for `(sentence, aspect) → polarity`; compare against VADER.
- [ ] **Slice 2 — SemEval eval harness:** load SemEval-2014, report extraction
      F1 + sentiment macro-F1.
- [ ] **Slice 3 — Fine-tuning:** train the BIO aspect tagger + ABSA classifier.
- [ ] **Slice 4 — Embedding clustering:** MiniLM + KMeans/HDBSCAN theme discovery.
- [ ] **Slice 5 — LLM executive summary** (optional; Ollama or API).
- [ ] **Slice 6 — Real Amazon/Yelp demo + polish.**

---

## License

[MIT](LICENSE) © 2026 Saud Satopay

*Final-year AI & Data Science NLP mini-project.*
