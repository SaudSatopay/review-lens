<div align="center">

# 🔍 ReviewLens

### **Aspect-Based Review Intelligence**

##### *Stop scoring reviews. Start understanding them.*

<p>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/🤗%20Transformers-FFD21E?style=for-the-badge" alt="HuggingFace Transformers">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly">
</p>

<p>
  <img src="https://img.shields.io/badge/license-MIT-yellow?style=flat-square" alt="MIT License">
  <img src="https://img.shields.io/badge/tests-79-brightgreen?style=flat-square" alt="Tests">
  <img src="https://img.shields.io/badge/code%20style-ruff-000000?style=flat-square" alt="Ruff">
  <img src="https://img.shields.io/badge/SemEval--14_macro--F1-0.79_vs_0.59_VADER-success?style=flat-square" alt="SemEval-2014 sentiment">
  <img src="https://img.shields.io/badge/extraction_F1-0.91_vs_0.52_baseline-8b5cf6?style=flat-square" alt="SemEval-2014 extraction">
</p>

</div>

---

<div align="center">

### Here is a **2-star** review.

> ### *"The display is stunning **but the battery is a dealbreaker**. Also the price keeps going up while quality stays the same."*

### Document-level sentiment scores it **`positive (+0.20)`**.

</div>

> [!WARNING]
> That is not a bug in the sentiment model — it's the ceiling of the whole approach.
> One score per review averages "stunning display" against "battery dealbreaker"
> and lands on a number that is **worse than useless**: it's confidently wrong.

**ReviewLens reads it the way a human does — one opinion per aspect:**

<div align="center">

| | 🖥️ display | 🔋 battery |
|:--|:--:|:--:|
| **Document-level VADER** | `positive +0.20` | `positive +0.20` ❌ |
| **Per-aspect VADER** *(baseline)* | `positive +0.20` | `positive +0.20` ❌ |
| **🏆 Transformer ABSA** | `positive +0.997` ✅ | **`negative −0.970`** ✅ |

</div>

Only the cross-encoder — which reads the sentence **and the aspect together** — holds
two opposite opinions inside one sentence. That gap *is* the project — and on the
**SemEval-2014** benchmark it measures **+18.5 to +21.7 macro-F1 points**
([details ↓](#-evaluation--measured-on-semeval-2014-task-4)).

> [!NOTE]
> **12 of 18** reviews in the sample carry mixed per-aspect sentiment. A single score
> silently flattens every one of them.

<br>

## 🧬 The pipeline

```mermaid
flowchart LR
    subgraph INGEST ["📥 Ingest"]
        A[Reviews CSV] --> B[Clean]
        B --> C[Sentence split]
    end

    subgraph EXTRACT ["🏷️ Extract"]
        C --> D[Aspect extraction]
    end

    subgraph CLASSIFY ["🎭 Classify"]
        D --> E[Aspect sentiment]
    end

    subgraph GROUP ["🧩 Group"]
        E --> F[Theme clustering]
        F --> G[Aggregation]
    end

    subgraph SERVE ["📊 Serve"]
        G --> H[Streamlit dashboard]
        G --> I[LLM exec summary]
    end

    style INGEST fill:#0f172a,stroke:#3b82f6,color:#fff
    style EXTRACT fill:#0f172a,stroke:#8b5cf6,color:#fff
    style CLASSIFY fill:#0f172a,stroke:#ec4899,color:#fff
    style GROUP fill:#0f172a,stroke:#f59e0b,color:#fff
    style SERVE fill:#0f172a,stroke:#10b981,color:#fff
```

Every stage ships a **fast baseline** first, then upgrades to a **transformer**:

| Stage | 🥉 Baseline (offline, seconds) | 🥇 Upgrade | Status |
|:--|:--|:--|:--:|
| **Aspect extraction** | Noun-phrase chunking (NLTK POS + grammar) | Fine-tuned **BIO** token classifier — ours | ✅ |
| **Aspect sentiment** | VADER on the aspect's sentence | ABSA cross-encoder — pretrained **or ours** | ✅ |
| **Theme clustering** | Keyword + normalization grouping | **MiniLM** embeddings → KMeans / HDBSCAN | ✅ |
| **Summary** | Top loved / hated ranking | **LLM** executive summary (Ollama / API) | ✅ |

> Baselines aren't throwaway scaffolding — they're the **benchmark**. Every upgrade has
> to beat a number, not a vibe.

<br>

## 📊 Baseline vs. transformer — measured, not claimed

Same 18 reviews, same aspects, only the sentiment model swapped
(`--aspect-model baseline` vs `--aspect-model absa`). Net score is
`(positive − negative) / mentions`, in `[-1, +1]`:

| Theme | 🥉 VADER | 🥇 ABSA | What the reviews actually say |
|:--|:--:|:--:|:--|
| 📞 **call** | `+1.00` | **`−1.00`** | *"they keep disconnecting during calls"*, *"microphone quality on calls is poor"* — the baseline was **100% inverted** |
| 💰 **price** | `+0.20` | **`−0.60`** | *"Disappointed for the price"*, *"the price is steep"* |
| 🎧 **service** | `−0.50` | **`−1.00`** | *"support ignored my emails"*, *"warranty process is a nightmare"* |
| 🔋 **battery** | `+0.50` | **`+0.12`** | genuinely split — praised on the watch, hated on the phone |

VADER inverts `call` because it scores whole sentences: *"Comfortable fit but they keep
disconnecting during calls"* reads as net-positive, so **every** aspect in it — including
the complaint — inherits `positive`.

<br>

## ⚡ Quickstart

```bash
git clone https://github.com/SaudSatopay/review-lens.git
cd review-lens
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

<details open>
<summary><b>🥉 Baseline — no torch, installs and runs in seconds</b></summary>

```bash
pip install -r requirements-core.txt
pip install -e .
python -c "from reviewlens.nltk_setup import ensure_nltk_data; ensure_nltk_data()"

python scripts/run_pipeline.py --group-by theme   # CLI report
streamlit run app/streamlit_app.py                # 📊 dashboard
```

</details>

<details>
<summary><b>🥇 Transformer ABSA — real (sentence, aspect) → polarity</b></summary>

```bash
pip install -e ".[ml]"                            # + torch, transformers

python scripts/run_pipeline.py --aspect-model absa --group-by theme
```

First run downloads `yangheng/deberta-v3-base-absa-v1.1` (~370 MB). Runs on CPU;
uses CUDA automatically when available. Or set it permanently in `config.yaml`:

```yaml
sentiment:
  aspect_model: absa
```

</details>

<details>
<summary><b>🎓 Train both models yourself — ~6 min on one GPU</b></summary>

```bash
pip install -e ".[ml]"
python scripts/download_semeval.py       # SemEval-2014 train/test XML
python scripts/train_extractor.py        # BIO aspect tagger  → models/aspect-extractor
python scripts/train_absa.py             # ABSA classifier    → models/absa-classifier

# run the pipeline fully on your own fine-tuned models
python scripts/run_pipeline.py --extractor transformer --aspect-model absa
```

GPU note: on Windows, the default `torch` wheel is CPU-only. For NVIDIA cards
(incl. RTX 50-series / Blackwell) install the CUDA build:
`pip install torch --index-url https://download.pytorch.org/whl/cu128`

</details>

<details>
<summary><b>🌍 Real-data demo — 500 Amazon reviews through the full stack</b></summary>

```bash
python scripts/download_reviews.py                      # Amazon "Software" 5-core, 500 reviews

python scripts/run_pipeline.py -i data/raw/amazon/software_reviews.csv \
    --extractor transformer --aspect-model absa --absa-checkpoint finetuned \
    --clustering kmeans --n-clusters 16
```

`--clustering kmeans` swaps the hand-written keyword themes for **MiniLM
embeddings clustered with KMeans** (or `hdbscan`), so themes are *discovered*
from the data — the only strategy that survives a vocabulary nobody enumerated.
`--absa-checkpoint finetuned` runs our own trained classifier instead of the
pretrained hub checkpoint.

</details>

<details>
<summary><b>🧠 LLM executive summary — optional, Ollama or any OpenAI-compatible API</b></summary>

```bash
# Option A: local — install Ollama, `ollama pull llama3.1`, then:
python scripts/run_pipeline.py --llm-summary

# Option B: hosted — copy .env.example to .env, set LLM_API_BASE / LLM_API_KEY / LLM_MODEL
```

The LLM never sees raw reviews: it receives the **aggregated stats digest**
(net scores, mention counts, representative quotes) and writes the "so what" —
an executive summary with prioritized recommendations, saved to
`reports/executive_summary.md`. No provider running? The step reports itself
skipped and everything else still completes. Zero new dependencies (stdlib HTTP).

</details>

<br>

## 🖥️ What you get

```console
$ python scripts/run_pipeline.py --extractor transformer --aspect-model absa --group-by theme

=== ReviewLens pipeline (extractor=transformer, sentiment=absa, themes=normalized) ===
Reviews processed : 18
Sentences         : 42
Aspect mentions   : 59

-- Top loved (theme) --
  camera             net=+1.00  (n=2)
  comfort            net=+1.00  (n=3)
  screen             net=+0.67  (n=6)
  delivery           net=+0.60  (n=5)
  sound              net=+0.43  (n=7)

-- Top hated (theme) --
  service            net=-1.00  (n=4)
  price              net=-0.60  (n=5)
  battery            net=+0.00  (n=9)

10 review(s) contain mixed per-aspect sentiment that a single
document-level VADER score would flatten.
```

The dashboard turns that into per-aspect stacked sentiment bars, a net-score ranking,
sentiment-over-time, representative quotes, product/rating filters — plus live
switching of extractor, sentiment model, theme grouping and dataset, and an
optional LLM summary panel.

<br>

## 🌍 Real data, real themes

The toy sample proves the mechanism; 500 real Amazon **Software** reviews
(5-core corpus, seeded sample) prove it survives contact with the wild
(`scripts/download_reviews.py`, then the full fine-tuned stack):

```console
=== ReviewLens pipeline (extractor=transformer, sentiment=absa, themes=kmeans) ===
Reviews processed : 500
Sentences         : 5294
Aspect mentions   : 6331

-- Top loved (theme) --                    -- Top hated (theme) --
  photoshop       net=+0.29  (n=278)         window          net=-0.05  (n=560)
  video editing   net=+0.27  (n=153)         hard drive      net=+0.02  (n=251)
  norton          net=+0.24  (n=263)         outlook         net=+0.05  (n=279)
  corel           net=+0.22  (n=85)          windows 8       net=+0.09  (n=664)

310 review(s) contain mixed per-aspect sentiment that a single
document-level VADER score would flatten.
```

Two things the real data makes obvious:

- **62% of real reviews are mixed** (310/500) — the failure mode a single
  document score hides isn't an edge case, it's the majority case.
- **Themes must be discovered, not enumerated.** `photoshop`, `norton`,
  `corel`, `outlook` — none of these exist in the hand-written keyword map,
  which fragments this corpus into thousands of singleton themes. The MiniLM
  KMeans/HDBSCAN clustering finds them and names each cluster after its most
  frequent member term. (The transformer extractor also emits **6.3k** aspect
  mentions where the noun-phrase baseline emits **16.1k** — the same
  precision-vs-noise gap SemEval measured, live.)

<br>

## 🗂️ Project structure

```
review-lens/
│
├── 📦 src/reviewlens/
│   ├── config.py               # loads config.yaml — single source of truth
│   ├── nltk_setup.py           # one-shot NLTK corpora bootstrap
│   ├── data/                   # ingest · clean · sentence-split · Amazon demo converter
│   ├── aspects/                # noun-phrase baseline · fine-tuned BIO tagger
│   ├── sentiment/              # VADER baseline · transformer ABSA cross-encoder
│   ├── clustering/             # keyword themes · MiniLM → KMeans / HDBSCAN
│   ├── aggregate/              # distributions · rankings · quotes · trends · LLM summary
│   ├── evaluation/             # SemEval-2014 downloader · parser · metrics · CLI
│   ├── training/               # BIO alignment · example builders · both fine-tunes
│   └── pipeline.py             # end-to-end orchestration + CLI
│
├── 📊 app/streamlit_app.py     # the dashboard (live model/theme/data switching)
├── 🔧 scripts/                 # run_pipeline · download_* · train_* · evaluate
├── 📈 reports/                 # committed benchmark results (JSON)
├── 📓 notebooks/               # exploration
├── ✅ tests/                   # 79 tests
├── 🗃️ data/sample/             # tiny sample — pipeline runs out of the box
├── 🧠 models/                  # fine-tuned weights land here (git-ignored)
└── ⚙️ config.yaml              # paths · models · thresholds
```

<br>

## ✅ Tests

```bash
pytest                                        # 77 tests, <4s (no downloads, no network)
REVIEWLENS_RUN_MODEL_TESTS=1 pytest           # + 2 tests that exercise the ABSA checkpoint
```

Model-dependent tests are opt-in by design — a default `pytest` should never pull
370 MB of weights. The SemEval parser and metrics are tested against tiny inline
fixtures with hand-computed expectations, not the real data. The embedding
clustering is tested with hand-placed fake vectors (real KMeans/HDBSCAN, no
MiniLM download), and the LLM step through an injected transport (no network).

<br>

## 📏 Evaluation — measured on SemEval-2014 Task 4

Gold test sets of the standard ABSA benchmark (Restaurants + Laptops), standard
3-class setup (`conflict` gold labels dropped). Full numbers live in
[`reports/semeval2014_results.json`](reports/semeval2014_results.json). Reproduce
everything, including training both models from scratch (~6 min on one GPU):

```bash
python scripts/download_semeval.py     # ~2.4 MB from public research mirrors
python scripts/train_extractor.py      # BIO tagger        (~3.5 min, RTX 5060 Ti)
python scripts/train_absa.py           # ABSA classifier   (~2.5 min)
python scripts/evaluate_semeval.py     # full benchmark, all models
```

### 🏷️ Aspect extraction — F1, gold test sets

| Test set | 🥉 Noun-phrase chunker | 🥇 **Our fine-tuned BIO tagger** | Δ |
|:--|:--:|:--:|:--:|
| **Restaurants** | 0.516 *(P 0.40 / R 0.73)* | **0.905** *(P 0.90 / R 0.92)* | **+39 pts** |
| **Laptops** | 0.353 *(P 0.25 / R 0.62)* | **0.850** *(P 0.86 / R 0.84)* | **+50 pts** |

The chunker's profile is classic unsupervised extraction: decent recall, poor
precision (it proposes noun phrases nobody has an opinion about). The fine-tune
(`roberta-base`, 5 epochs on the combined train splits) fixes precisely that.
Matching is case-insensitive exact term-set per sentence — the baseline emits no
character offsets, so scores are comparable to, but not identical with, the
official offset-based scorer.

### 🎭 Aspect sentiment — gold aspect terms, macro-F1 (accuracy)

| Test set | 🥉 VADER | 🥈 Pretrained checkpoint¹ | 🥇 **Our fine-tune²** |
|:--|:--:|:--:|:--:|
| **Restaurants** *(n=1,120)* | 0.608 *(0.733)* | 0.793 *(0.837)* | **0.792** *(**0.868**)* |
| **Laptops** *(n=638)* | 0.573 *(0.625)* | 0.790 *(0.828)* | **0.769** *(0.807)* |

¹ `yangheng/deberta-v3-base-absa-v1.1` — its training mix **includes the
SemEval-2014 train splits** (plus other ABSA corpora), so treat its scores as an
optimistic upper bound. The caveat is recorded in the results JSON itself.
² `roberta-base` trained on the official train splits **only** — the clean,
defensible number. It matches the upper bound on Restaurants macro-F1 and beats
it on accuracy, from a smaller model seeing a fraction of the data.

Where the gap over VADER lives (per-class F1, Restaurants, pretrained ABSA):

| Class | 🥉 VADER | 🥇 ABSA | |
|:--|:--:|:--:|:--|
| positive | 0.860 | 0.901 | the easy majority class — everyone scores here |
| negative | 0.572 | **0.823** | contrastive sentences sink VADER |
| neutral | 0.393 | **0.655** | ≈ coin flip vs. nearly doubled |

VADER survives on positive-heavy data and collapses exactly where per-aspect
understanding matters: negative and neutral opinions inside mixed sentences.

> [!NOTE]
> **Why `roberta-base` and not `deberta-v3-base`?** DeBERTa-v3's backward pass
> NaNs deterministically on the current stack (transformers 5.6.x + torch 2.11 —
> CPU and GPU alike, any precision; inference is unaffected). Under `Trainer`,
> gradient clipping masks the NaNs and the model silently degenerates to
> predicting class priors with dev F1 = 0. Diagnosed with a manual training loop
> watching encoder gradients; documented in `training/common.py`.

<br>

## 🗺️ Roadmap

- [x] **Slice 0 — Scaffold + runnable baseline** · ingest → aspects → sentiment → themes → dashboard → tests
- [x] **Slice 1 — Transformer ABSA** · `deberta-v3-base-absa` cross-encoder, swappable via config/CLI
- [x] **Slice 2 — Evaluation harness** · SemEval-2014 measured: sentiment macro-F1 **0.79 vs 0.59** (VADER), extraction baseline F1 0.52/0.35
- [x] **Slice 3 — Fine-tuning** · our BIO tagger: extraction F1 **0.91 / 0.85** (vs 0.52/0.35 baseline); our ABSA classifier: macro-F1 **0.79 / 0.77** trained on SemEval train only — matches the pretrained upper bound on Restaurants
- [x] **Slice 4 — Embedding clustering** · MiniLM + KMeans/HDBSCAN theme discovery, swappable via `--clustering`; clusters auto-named after their most frequent term
- [x] **Slice 5 — LLM executive summary** · optional `--llm-summary`, Ollama or any OpenAI-compatible API, stats-digest grounded, degrades gracefully
- [x] **Slice 6 — Real Amazon demo** + polish · 500 real reviews through the full fine-tuned stack: **62% mixed**, themes discovered that no keyword list contained

<br>

## 🛠️ Built with

<div align="center">

`Python` · `PyTorch` · `HuggingFace Transformers` · `sentence-transformers`
`scikit-learn` · `NLTK` · `VADER` · `pandas` · `Streamlit` · `Plotly` · `pytest` · `ruff`

</div>

<br>

---

<div align="center">

**[MIT](LICENSE)** © 2026 **Saud Satopay**

*Final-year AI & Data Science NLP mini-project*

</div>
