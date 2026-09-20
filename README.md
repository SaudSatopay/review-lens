<div align="center">

<br>

# ReviewLens.

**A S P E C T &thinsp;·&thinsp; B A S E D &thinsp;·&thinsp; R E V I E W &thinsp;·&thinsp; I N T E L L I G E N C E**

*Stop scoring reviews. Start reading them.*

<br>

<img src="https://img.shields.io/badge/python-3.10+-191511?style=flat-square&labelColor=191511&color=6E2231" alt="Python 3.10+">
<img src="https://img.shields.io/badge/PyTorch_·_Transformers-fine--tuned-191511?style=flat-square&labelColor=191511&color=6E2231" alt="PyTorch · Transformers">
<img src="https://img.shields.io/badge/Streamlit_·_Plotly-dashboard-191511?style=flat-square&labelColor=191511&color=6E2231" alt="Streamlit · Plotly">
<img src="https://img.shields.io/badge/license-MIT-191511?style=flat-square&labelColor=191511&color=A2988A" alt="MIT License">
<br>
<img src="https://img.shields.io/badge/sentiment_macro--F1-0.79_vs_0.59_VADER-191511?style=flat-square&labelColor=191511&color=0F8265" alt="SemEval-2014 sentiment">
<img src="https://img.shields.io/badge/extraction_F1-0.91_vs_0.52_baseline-191511?style=flat-square&labelColor=191511&color=0F8265" alt="SemEval-2014 extraction">
<img src="https://img.shields.io/badge/tests-79_·_no_network-191511?style=flat-square&labelColor=191511&color=0F8265" alt="Tests">

<br><br>

<img src="docs/dashboard.png" alt="The ReviewLens dashboard — a data journal on warm paper" width="100%">

</div>

<br>

## Exhibit A — a two-star review

> *“The **display** is stunning but the **battery** is a dealbreaker. Also the
> price keeps going up while quality stays the same.”*

Document-level sentiment scores it **`positive (+0.20)`** — and that is not a
bug in the sentiment model, it is the ceiling of the whole approach. One score
per review averages “stunning display” against “battery dealbreaker” and lands
on a number that is worse than useless: confidently wrong.

ReviewLens reads it the way a human does — **one opinion per aspect**:

| | display | battery |
|:--|:--:|:--:|
| Document-level VADER | `positive +0.20` | `positive +0.20` ✗ |
| Per-aspect VADER *(baseline)* | `positive +0.20` | `positive +0.20` ✗ |
| **Transformer ABSA** | `positive +0.997` ✓ | **`negative −0.970`** ✓ |

Only the cross-encoder — which reads the sentence **and the aspect together** —
holds two opposite opinions inside one sentence. That gap *is* the project, and
on the **SemEval-2014** benchmark it measures **+18.5 to +21.7 macro-F1 points**
([the numbers ↓](#evaluation)). On real
Amazon data, **62% of reviews** carry mixed per-aspect sentiment — the failure
mode a single score hides is not an edge case, it is the majority case.

<br>

## № 01 — The pipeline

```mermaid
flowchart LR
    subgraph INGEST ["Ingest"]
        A[Reviews CSV] --> B[Clean]
        B --> C[Sentence split]
    end

    subgraph EXTRACT ["Extract"]
        C --> D[Aspect extraction]
    end

    subgraph CLASSIFY ["Classify"]
        D --> E[Aspect sentiment]
    end

    subgraph GROUP ["Group"]
        E --> F[Theme clustering]
        F --> G[Aggregation]
    end

    subgraph SERVE ["Serve"]
        G --> H[Dashboard]
        G --> I[LLM exec summary]
    end

    style INGEST fill:#FCF9F1,stroke:#191511,color:#191511
    style EXTRACT fill:#FCF9F1,stroke:#0F8265,color:#191511
    style CLASSIFY fill:#FCF9F1,stroke:#C34A24,color:#191511
    style GROUP fill:#FCF9F1,stroke:#A2988A,color:#191511
    style SERVE fill:#FCF9F1,stroke:#6E2231,color:#191511
```

Every stage ships a **fast baseline** first, then a **transformer** upgrade
that has to beat a number, not a vibe:

| Stage | Baseline (offline, seconds) | Upgrade | |
|:--|:--|:--|:--:|
| **Aspect extraction** | Noun-phrase chunking (NLTK POS + grammar) | Fine-tuned **BIO** token classifier — ours | ✓ |
| **Aspect sentiment** | VADER on the aspect's sentence | ABSA cross-encoder — pretrained **or ours** | ✓ |
| **Theme clustering** | Keyword + normalization grouping | **MiniLM** embeddings → KMeans / HDBSCAN | ✓ |
| **Summary** | Top loved / hated ranking | **LLM** executive summary (Ollama / API) | ✓ |

<br>

## № 02 — Baseline vs. transformer, measured

Same 18 sample reviews, same aspects, only the sentiment model swapped
(`--aspect-model baseline` vs `absa`). Net score is
`(positive − negative) ÷ mentions`, in `[−1, +1]`:

| Theme | VADER | ABSA | What the reviews actually say |
|:--|:--:|:--:|:--|
| **call** | `+1.00` | **`−1.00`** | *“they keep disconnecting during calls”* — the baseline was **100% inverted** |
| **price** | `+0.20` | **`−0.60`** | *“Disappointed for the price”*, *“the price is steep”* |
| **service** | `−0.50` | **`−1.00`** | *“support ignored my emails”*, *“warranty process is a nightmare”* |
| **battery** | `+0.50` | **`+0.12`** | genuinely split — praised on the watch, hated on the phone |

VADER inverts `call` because it scores whole sentences: *“Comfortable fit but
they keep disconnecting during calls”* reads net-positive, so **every** aspect
in it — including the complaint — inherits `positive`.

<br>

## № 03 — Quickstart

```bash
git clone https://github.com/SaudSatopay/review-lens.git
cd review-lens
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

<details open>
<summary><b>Baseline — no torch, installs and runs in seconds</b></summary>

```bash
pip install -r requirements-core.txt
pip install -e .
python -c "from reviewlens.nltk_setup import ensure_nltk_data; ensure_nltk_data()"

python scripts/run_pipeline.py --group-by theme   # CLI report
streamlit run app/streamlit_app.py                # the dashboard
```

</details>

<details>
<summary><b>Transformer ABSA — real (sentence, aspect) → polarity</b></summary>

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
<summary><b>Train both models yourself — ~6 min on one GPU</b></summary>

```bash
pip install -e ".[ml]"
python scripts/download_semeval.py       # SemEval-2014 train/test XML
python scripts/train_extractor.py        # BIO aspect tagger  → models/aspect-extractor
python scripts/train_absa.py             # ABSA classifier    → models/absa-classifier

# run the pipeline fully on your own fine-tuned models
python scripts/run_pipeline.py --extractor transformer --aspect-model absa \
    --absa-checkpoint finetuned
```

GPU note: on Windows, the default `torch` wheel is CPU-only. For NVIDIA cards
(incl. RTX 50-series / Blackwell) install the CUDA build:
`pip install torch --index-url https://download.pytorch.org/whl/cu128`

</details>

<details>
<summary><b>Real-data demo — 500 Amazon reviews through the full stack</b></summary>

```bash
python scripts/download_reviews.py                      # Amazon "Software" 5-core, 500 reviews

python scripts/run_pipeline.py -i data/raw/amazon/software_reviews.csv \
    --extractor transformer --aspect-model absa --absa-checkpoint finetuned \
    --clustering kmeans --n-clusters 16
```

`--clustering kmeans` swaps the hand-written keyword themes for **MiniLM
embeddings clustered with KMeans** (or `hdbscan`), so themes are *discovered*
from the data — the only strategy that survives a vocabulary nobody enumerated.

</details>

<details>
<summary><b>LLM executive summary — optional, Ollama or any OpenAI-compatible API</b></summary>

```bash
# Option A: local — install Ollama, `ollama pull llama3.1`, then:
python scripts/run_pipeline.py --llm-summary

# Option B: hosted — copy .env.example to .env, set LLM_API_BASE / LLM_API_KEY / LLM_MODEL
```

The LLM never sees raw reviews: it receives the **aggregated stats digest**
(net scores, mention counts, representative quotes) and writes the “so what” —
an executive summary with prioritized recommendations, saved to
`reports/executive_summary.md`. No provider running? The step reports itself
skipped and everything else still completes. Zero new dependencies (stdlib HTTP).

</details>

<br>

## № 04 — The dashboard: a data journal

Warm paper, near-black ink, and an argument on every page. The masthead opens
onto *Exhibit A* — the two-star review, annotated. Below it, the figures row,
then six numbered sections: sentiment by theme, the verdict, **In one breath**
(reviews that praise one aspect and damn another — the contradictions a single
score erases), over time, the praise / the complaints, and an optional LLM memo.
Every model is hot-swappable from the sidebar: extractor, sentiment checkpoint,
theme grouping, dataset.

<div align="center">
<img src="docs/verdict.png" alt="The verdict — net sentiment ranking with most-loved and most-hated annotations" width="92%">
<br><br>
<img src="docs/breath.png" alt="In one breath — a review praising the camera and damning the price" width="92%">
<br><br>
<img src="docs/voices.png" alt="In their own words — the praise and the complaints as pull-quote clippings" width="92%">
</div>

Set in **Gloock**, **Instrument Serif & Sans**, and **Spline Sans Mono**. The
semantic palette — pine-teal `#0F8265` / warm-gray midpoint / vermilion
`#C34A24` on paper `#F5F0E6` — was validated for lightness, chroma,
colorblind-safety and contrast (a muted-olive candidate was *rejected*: protan
ΔE 3.3 against vermilion — loved and hated would have been indistinguishable).
Burgundy `#6E2231` carries the interactive chrome and never appears inside a
chart, so color always means sentiment.

<br>

## № 05 — Real data, real themes

The toy sample proves the mechanism; 500 real Amazon **Software** reviews
(5-core corpus, seeded sample) prove it survives contact with the wild:

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

- **62% of real reviews are mixed** (310/500) — per-aspect reading is the
  majority case, not a refinement.
- **Themes must be discovered, not enumerated.** `photoshop`, `norton`,
  `corel`, `outlook` — none of these exist in the hand-written keyword map,
  which fragments this corpus into ~1,500 singleton themes. The MiniLM
  clustering finds them and names each cluster after its most frequent member
  term. (The transformer extractor also emits **6.3k** aspect mentions where
  the noun-phrase baseline emits **16.1k** — the same precision-vs-noise gap
  SemEval measured, live.)

<br>

<a id="evaluation"></a>

## № 06 — Evaluation, measured on SemEval-2014 Task 4

Gold test sets of the standard ABSA benchmark (Restaurants + Laptops), standard
3-class setup (`conflict` gold labels dropped). Full numbers live in
[`reports/semeval2014_results.json`](reports/semeval2014_results.json).
Reproduce everything, including training both models from scratch:

```bash
python scripts/download_semeval.py     # ~2.4 MB from public research mirrors
python scripts/train_extractor.py      # BIO tagger        (~3.5 min, RTX 5060 Ti)
python scripts/train_absa.py           # ABSA classifier   (~2.5 min)
python scripts/evaluate_semeval.py     # full benchmark, all models
```

### Aspect extraction — F1, gold test sets

| Test set | Noun-phrase chunker | **Our fine-tuned BIO tagger** | Δ |
|:--|:--:|:--:|:--:|
| **Restaurants** | 0.516 *(P 0.40 / R 0.73)* | **0.905** *(P 0.90 / R 0.92)* | **+39 pts** |
| **Laptops** | 0.353 *(P 0.25 / R 0.62)* | **0.850** *(P 0.86 / R 0.84)* | **+50 pts** |

The chunker's profile is classic unsupervised extraction: decent recall, poor
precision (it proposes noun phrases nobody has an opinion about). The fine-tune
(`roberta-base`, 5 epochs on the combined train splits) fixes precisely that.
Matching is case-insensitive exact term-set per sentence — the baseline emits no
character offsets, so scores are comparable to, but not identical with, the
official offset-based scorer.

### Aspect sentiment — gold aspect terms, macro-F1 (accuracy)

| Test set | VADER | Pretrained checkpoint¹ | **Our fine-tune²** |
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

| Class | VADER | ABSA | |
|:--|:--:|:--:|:--|
| positive | 0.860 | 0.901 | the easy majority class — everyone scores here |
| negative | 0.572 | **0.823** | contrastive sentences sink VADER |
| neutral | 0.393 | **0.655** | ≈ coin flip vs. nearly doubled |

VADER survives on positive-heavy data and collapses exactly where per-aspect
understanding matters: negative and neutral opinions inside mixed sentences.

> **Why `roberta-base` and not `deberta-v3-base`?** DeBERTa-v3's backward pass
> NaNs deterministically on the current stack (transformers 5.6.x + torch 2.11 —
> CPU and GPU alike, any precision; inference is unaffected). Under `Trainer`,
> gradient clipping masks the NaNs and the model silently degenerates to
> predicting class priors with dev F1 = 0. Diagnosed with a manual training loop
> watching encoder gradients; documented in `training/common.py`.

<br>

## № 07 — Project structure

```
review-lens/
│
├── src/reviewlens/
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
├── app/                        # the dashboard (streamlit_app.py + style.css)
├── scripts/                    # run_pipeline · download_* · train_* · evaluate
├── reports/                    # committed benchmark results (JSON)
├── docs/                       # the screenshots on this page
├── notebooks/                  # exploration
├── tests/                      # 79 tests
├── data/sample/                # tiny sample — pipeline runs out of the box
├── models/                     # fine-tuned weights land here (git-ignored)
└── config.yaml                 # paths · models · thresholds
```

<br>

## № 08 — Tests

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

## № 09 — Roadmap

- [x] **Slice 0 — Scaffold + runnable baseline** · ingest → aspects → sentiment → themes → dashboard → tests
- [x] **Slice 1 — Transformer ABSA** · `deberta-v3-base-absa` cross-encoder, swappable via config/CLI
- [x] **Slice 2 — Evaluation harness** · SemEval-2014 measured: sentiment macro-F1 **0.79 vs 0.59** (VADER), extraction baseline F1 0.52/0.35
- [x] **Slice 3 — Fine-tuning** · our BIO tagger: extraction F1 **0.91 / 0.85** (vs 0.52/0.35 baseline); our ABSA classifier: macro-F1 **0.79 / 0.77** trained on SemEval train only — matches the pretrained upper bound on Restaurants
- [x] **Slice 4 — Embedding clustering** · MiniLM + KMeans/HDBSCAN theme discovery, swappable via `--clustering`; clusters auto-named after their most frequent term
- [x] **Slice 5 — LLM executive summary** · optional `--llm-summary`, Ollama or any OpenAI-compatible API, stats-digest grounded, degrades gracefully
- [x] **Slice 6 — Real Amazon demo** + polish · 500 real reviews through the full fine-tuned stack: **62% mixed**, themes discovered that no keyword list contained

<br>

---

<div align="center">

**Python · PyTorch · HuggingFace Transformers · sentence-transformers
scikit-learn · NLTK · VADER · pandas · Streamlit · Plotly · pytest · ruff**

<br>

*Set in Gloock & Instrument · measured on SemEval-2014*

**[MIT](LICENSE)** © 2026 **Saud Satopay** — *final-year AI & Data Science NLP project*

</div>
