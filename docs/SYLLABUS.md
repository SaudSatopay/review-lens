# ReviewLens × the NLP syllabus (CSDO7011)

Where each course module shows up in this repository — every claim points at
code, and every model rung is measured on the same SemEval-2014 gold sets
([`reports/semeval2014_results.json`](../reports/semeval2014_results.json)).

## Module 1 — Introduction

| Syllabus item | In ReviewLens |
|:--|:--|
| Ambiguity in natural language | The project's thesis: one review carries opposite verdicts; a document-level score flattens them (README, "Exhibit A") |
| Levels of NLP · applications | The pipeline stages mirror the levels: lexical → syntactic → semantic → discourse → application |

## Module 2 — Word Level Analysis

| Syllabus item | In ReviewLens |
|:--|:--|
| Tokenization | NLTK `word_tokenize` ([`aspects/baseline.py`](../src/reviewlens/aspects/baseline.py)); offset-preserving regex tokenizer ([`aspects/crf.py`](../src/reviewlens/aspects/crf.py)); subword BPE/SentencePiece in the transformers |
| Segmentation | Punkt sentence splitting ([`data/split.py`](../src/reviewlens/data/split.py)) |
| Lemmatization (naive) | `normalize_term` singularization ([`clustering/themes.py`](../src/reviewlens/clustering/themes.py)) |
| Regular expressions | POS-tag chunk grammar `NP: {<NN.*>+}`; cleaning regexes |
| **Collocations (PMI)** | Manning & Schütze bigram PMI with a min-count guard ([`aspects/collocations.py`](../src/reviewlens/aspects/collocations.py)); reported by `scripts/fetch_amazon.py` |
| Corpora · training & testing | SemEval-2014 corpora, official train/test splits, macro-F1 ([`evaluation/`](../src/reviewlens/evaluation)) |

## Module 3 — Syntax Analysis

| Syllabus item | In ReviewLens |
|:--|:--|
| POS tagging · Penn Treebank · open/closed classes | NLTK's stochastic tagger drives the baseline extractor; nouns kept, adjectives excluded by design |
| **CRF sequence labeling** | [`aspects/crf.py`](../src/reviewlens/aspects/crf.py): classic feature templates (shape, affixes, POS, ±1 context), L-BFGS + elastic net, BIO scheme — the measured middle rung |
| HMM / MaxEnt (context) | Discussed lineage of the same BIO task; the transformer token classifier is its neural successor |

**Measured (extraction F1, gold test):** chunker 0.52 / 0.35 → **CRF 0.78 / 0.70** → RoBERTa BIO 0.91 / 0.85 (Restaurants / Laptops).

## Module 4 — Semantic Analysis

| Syllabus item | In ReviewLens |
|:--|:--|
| Lexical semantics · synonymy / hyponymy | Theme grouping: keyword map → **WordNet synonym/direct-hyponym grouping** (`--clustering wordnet`, [`clustering/themes.py`](../src/reviewlens/clustering/themes.py)) → MiniLM distributional embeddings |
| WordNet | `wordnet_theme_map`: union-find over shared synsets + direct hypernyms; unknown brand words keep their own theme |
| **Supervised classification (Naïve Bayes)** | [`sentiment/naive_bayes.py`](../src/reviewlens/sentiment/naive_bayes.py): MultinomialNB over uni/bigrams + an aspect marker token — the measured middle rung |
| Semantic ambiguity / disambiguation | The ABSA cross-encoder disambiguates polarity *in context of the aspect* |

**Measured (sentiment, gold test):** VADER acc 0.73 / macro-F1 0.61 → **NB acc 0.75 / macro-F1 0.59** → our RoBERTa ABSA acc 0.87 / macro-F1 0.79 (Restaurants). The NB result is the honest lesson: supervised bag-of-words buys accuracy on the majority class but *not* per-class balance — only a contextual model fixes negative/neutral inside mixed sentences.

## Module 5 — Pragmatics & Discourse

| Syllabus item | In ReviewLens |
|:--|:--|
| Reference / anaphora resolution | [`aspects/anaphora.py`](../src/reviewlens/aspects/anaphora.py): Hobbs-flavored recency heuristic — an aspect-less, pronoun-initial sentence inherits the review's most recent aspect (*"The battery is huge. **It** drains in an hour."*). `--anaphora` / dashboard toggle; off by default so benchmark numbers stay reproducible |
| Coherence (informal) | The dashboard's "In one breath" section surfaces intra-review contradictions |

## Module 6 — Applications

| Syllabus item | In ReviewLens |
|:--|:--|
| Sentiment analysis | The whole project (aspect-based) |
| NER | Aspect extraction is NER-shaped sequence labeling (BIO) |
| Categorization | Theme clustering (keyword / WordNet / KMeans / HDBSCAN) |
| Summarization | Top loved/hated ranking (extractive); LLM executive memo (abstractive) |
| Information retrieval (light) | Representative-quote selection; live Amazon product fetch (`scripts/fetch_amazon.py`) |
| Indian regional languages | *Future work:* the architecture is model-swappable — point `training.base_model` at MuRIL/IndicBERT and retrain on Hindi/Marathi review data |

## Not used (and why)

FST/finite automata, edit distance, full HMM/MaxEnt training, Yarowsky/Hyperlex,
full Hobbs/Centering, MT and QA — out of scope for an aspect-sentiment product;
named here so the omission is a decision, not an oversight.
