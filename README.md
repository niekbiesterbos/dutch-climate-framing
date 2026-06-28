# Dutch Climate Framing

Replication code for: **From Rhetoric to Action: Climate Framing in Dutch Politics (2008-2025)**

Master's thesis, Information Science, University of Groningen.
---

Niek Biesterbos
Supervisor: T. Caselli 

## Overview

This repository contains all scripts to reproduce the full analysis pipeline:

1. Collecting parliamentary motions via the Tweede Kamer OData API (2008-2025)
2. Classifying motions as climate-related using a fine-tuned RoBERTa model
3. Scoring macro-frames (Likert 1-5) across 7 dimensions using open LLMs
4. Micro-frame actor/patient extraction using LOME semantic role labelling
5. Statistical analyses: party profiles, bloc comparisons, polarisation trends, shock event tests

The annotation scheme follows **Biesterbos (2025) v4**: seven behaviorally anchored macro-frames scored on a 1-5 Likert scale per parliamentary text unit.

---

## Repository structure

```
dutch-climate-framing/
├── data/
│   ├── motions/
│   │   └── all_motions_2008_2025.csv       All motions fetched from OData API
│   ├── speeches/
│   │   ├── ParlaMint-NL/                   ParlaMint XML corpus (2014–2022)
│   │   └── ParlaMint-Okky/                 ParlaMint JSONL (speeches)
│   └── manifestos/
│       ├── parties/                         Party election manifestos (PDF/TXT)
│       └── relevant_phrases.csv             Climate-relevant phrases extracted
│
├── results/
│   ├── motions/
│   │   ├── climate_motions.csv             Classifier output (all climate motions)
│   │   ├── gold_sample.csv                 Stratified sample for annotation
│   │   ├── gold_macro.csv                  Gold macro-frame scores (motions)
│   │   ├── gold_micro.csv                  Gold micro-frame annotations (motions)
│   │   ├── macro_scores/                   LLM scoring outputs (one CSV per model)
│   │   │   ├── gemma-3-12b-it.csv
│   │   │   ├── gemma-3-27b-it.csv
│   │   │   ├── qwen2.5-14b.csv
│   │   │   ├── qwen2.5-32b.csv
│   │   │   └── qwen3.5-27b.csv
│   │   └── micro_scores/                   LOME + actor/patient outputs
│   ├── speeches/
│   │   ├── utterances.csv                  Extracted speech utterances
│   │   ├── gold_sample.csv                 Stratified sample for annotation
│   │   ├── gold_macro.csv                  Gold macro-frame scores (speeches)
│   │   └── macro_scores/                   LLM scoring outputs
│   ├── manifestos/
│   │   ├── gold_sample.csv                 Stratified sample for annotation
│   │   ├── gold_macro.csv                  Gold macro-frame scores (manifestos)
│   │   ├── macro_scores/                   LLM scoring outputs
│   │   └── micro_scores/                   LOME outputs
│   ├── validation/
│   │   ├── model_selection_motions.csv     Model selection results (motions)
│   │   ├── model_selection_speeches.csv    Model selection results (speeches)
│   │   ├── model_selection_manifestos.csv  Model selection results (manifestos)
│   │   ├── per_frame_motions.csv           Per-frame validation (motions)
│   │   ├── per_frame_speeches.csv          Per-frame validation (speeches)
│   │   └── per_frame_manifestos.csv        Per-frame validation (manifestos)
│   ├── analysis/
│   │   ├── *.csv                           Analysis outputs
│   │   └── figures/                         Plots and figures
│   └── classifier/
│       └── binary_annotations.csv          Binary annotation set for classifier
│
├── src/
│   ├── collect/                            Data collection scripts
│   │   ├── fetch_motions_api.py            Fetch motions from Tweede Kamer OData API
│   │   ├── fetch_historical_motions.py     Scrape historical motion PDFs
│   │   ├── fetch_motion_texts.py           Download motion texts from URLs
│   │   ├── normalize_texts.py              Text cleaning pipeline
│   │   ├── populate_motion_texts.py        Fill motion text from URLs
│   │   ├── extract_parlamint_motions.py    Extract motions from ParlaMint XML
│   │   └── extract_parlamint_speeches.py   Extract utterances from ParlaMint JSONL
│   │
│   ├── classify/                           Climate motion classifier
│   │   ├── train_classifier.py             Fine-tune RoBERTa (initial)
│   │   ├── self_train.py                   Self-training loop
│   │   ├── predict.py                      Classifier inference
│   │   ├── label_all_motions.py            Apply classifier to all motions
│   │   └── filter_by_party.py             Filter motions by party
│   │
│   ├── score/                              LLM macro-frame scoring
│   │   ├── macro_frame_motions_gemma12b.py
│   │   ├── macro_frame_motions_gemma27b.py
│   │   ├── macro_frame_motions_qwen14b.py
│   │   ├── macro_frame_motions_qwen32b.py
│   │   ├── macro_frame_motions_qwen35_27b.py
│   │   ├── macro_frame_manifestos.py
│   │   └── macro_frame_speeches.py
│   │
│   ├── annotate/                           Gold standard annotation tools
│   │   ├── frames.py                       Shared frame definitions (Biesterbos 2025 v4)
│   │   ├── sample_gold_motions.py          Draw stratified sample for motions
│   │   ├── gold_motions_macro.py           Interactive macro-frame annotator (motions)
│   │   ├── gold_speeches_macro.py          Interactive macro-frame annotator (speeches)
│   │   ├── gold_manifestos_macro.py        Interactive macro-frame annotator (manifestos)
│   │   ├── gold_motions_micro.py           Interactive micro-frame annotator (motions)
│   │   └── finalize_gold_motions.py        Clean up and finalize gold CSV
│   │
│   ├── validate/                           Model validation
│   │   ├── validate_models.py              Full validation table (all models, all text types)
│   │   └── compare_models.py              Pairwise model vs. gold comparison
│   │
│   ├── analyse/                            Statistical analysis
│   │   ├── macro_analysis.py              Party profiles, bloc comparisons, temporal trends
│   │   ├── linguistic_analysis.py         Passive voice, negation, modality
│   │   ├── tfidf.py                        TF-IDF rankings per party
│   │   ├── party_similarity.py            Cosine similarity between parties (Figure 5.9)
│   │   ├── cooccurrence_issue_lexical.py  PPMI contextual collocations (Table 5.8)
│   │   ├── count_lome_frames.py           LOME FF-ICF fingerprints + temporal trends
│   │   ├── negation_in_frames.py          Negation within LOME frames (§5.3)
│   │   ├── polarisation.py                Polarisation trends (2008–2025)
│   │   ├── shock_analysis.py              2019 nitrogen case study (§5.4)
│   │   ├── negation_passive.py            Negation and passive voice by party
│   │   ├── event_analysis.py              Climate attention around key events
│   │   ├── mlm_probing.py                 MLM context window probing
│   │   ├── context_window.py
│   │   └── mlm_table.py
│   │
│   └── utils/                             Shared utilities
│       ├── lome_to_csv.py                 Convert LOME JSON output to CSV
│       ├── apply_spacy.py                 Apply spaCy NLP pipeline
│       ├── reconstruct_checkpoint.py      Reconstruct output from checkpoint files
│       ├── ner_patient_taxonomy.py        NER-based patient category classifier
│       ├── intensity_prober.py            MLM intensity probing
│       ├── match_seed_gold.py             TF-IDF matching of seed vs. gold motions
│       ├── merge_majortopic.py            Merge CAP major topic codes
│       └── topic_summary.py              Motions per CAP topic category
│
├── scripts/                               SLURM job scripts for HPC execution
│   ├── exp9_qwen35_27b_gold.sh
│   ├── exp18_classify.sh
│   ├── run_exp19.sh
│   └── exp20_speeches.sh
│
├── docs/
│   └── lome_hpc.md                        HPC setup guide for running LOME
├── requirements.txt
└── README.md
```

---

## Gold standard data

Gold standard files in `results/` contain manually annotated scores and are the primary input for all validation and analysis scripts. Do not regenerate these unless replicating the full annotation procedure.

| File | N | Description |
|------|---|-------------|
| `results/motions/gold_macro.csv` | ~465 | Motions with macro-frame scores (1–5) |
| `results/motions/gold_micro.csv` | ~200 | LOME clauses with gold actor/patient annotations |
| `results/speeches/gold_macro.csv` | ~755 | Speech utterances with macro-frame scores |
| `results/manifestos/gold_macro.csv` | ~200 | Manifesto phrases with macro-frame scores |

**Columns (macro gold):** `id/sample_idx`, `text/normalized_text`, `year`, `economic`, `moral`, `scientific`, `security`, `health_environment`, `crisis_urgency`, `weaponization`

All scores are integers 1–5 on a behaviorally anchored Likert scale (1 = Absent, 5 = Dominant). Frame definitions are in `src/annotate/frames.py`.

---

## Reproducing the pipeline

### Prerequisites

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download nl_core_news_lg
```

For GPU-based LLM inference (`src/score/`) a CUDA-capable GPU with at least 40 GB VRAM is required (tested on NVIDIA RTX Pro 6000). Scripts are designed to be run on a SLURM cluster; the `scripts/` directory contains the corresponding job submissions.

All scripts resolve paths relative to the project root using `Path(__file__).resolve().parents[N]` and `os.chdir(PROJECT_ROOT)`, so they can be run from any working directory.

---

### Step-by-step reproduction

#### Phase 1 — Data collection

```bash
# 1a. Fetch all motions from Tweede Kamer OData API (2008–2025)
python3 src/collect/fetch_motions_api.py
# Output: data/motions/all_motions_2008_2025.csv

# 1b. Download motion texts from PDF URLs (parallel, resumable)
python3 src/collect/fetch_motion_texts.py

# 1c. Normalize texts for classifier input
python3 src/collect/normalize_texts.py

# 1d. Extract speech utterances from ParlaMint corpus
python3 src/collect/extract_parlamint_speeches.py
# Output: results/speeches/utterances.csv
```

#### Phase 2 — Climate motion classification

```bash
# 2a. Extract ParlaMint motions as seed set
python3 src/collect/extract_parlamint_motions.py

# 2b. Fine-tune RoBERTa climate classifier
python3 src/classify/train_classifier.py

# 2c. Apply classifier to all motions
python3 src/classify/label_all_motions.py
# Output: results/motions/climate_motions.csv
```

#### Phase 3 — Macro-frame scoring

Requires GPU. Run locally or submit via SLURM:

```bash
# Motions (run sequentially or in separate SLURM jobs)
python3 src/score/macro_frame_motions_gemma12b.py
python3 src/score/macro_frame_motions_gemma27b.py
python3 src/score/macro_frame_motions_qwen14b.py
python3 src/score/macro_frame_motions_qwen32b.py
sbatch scripts/exp9_qwen35_27b_gold.sh          # Qwen3.5-27B via SLURM

# Manifestos
python3 src/score/macro_frame_manifestos.py
# sbatch scripts/run_exp19.sh                   # SLURM version

# Speeches
python3 src/score/macro_frame_speeches.py
# sbatch scripts/exp20_speeches.sh              # SLURM version
```

All outputs land in `results/{motions,speeches,manifestos}/macro_scores/`.

#### Phase 4 — Gold standard annotation (interactive terminal)

All annotation scripts are interactive terminal tools — no API key required. Each script is resumable: re-running it skips already-annotated items.

```bash
# Motions gold
python3 src/annotate/sample_gold_motions.py          # draw stratified sample
python3 src/annotate/gold_motions_macro.py            # annotate interactively
python3 src/annotate/finalize_gold_motions.py         # clean up final CSV

# Speeches gold
python3 src/annotate/gold_speeches_macro.py           # draws sample + annotates

# Manifestos gold
python3 src/annotate/gold_manifestos_macro.py         # draws sample + annotates
```

To annotate micro-frames (actor/patient in LOME clauses):

```bash
python3 src/annotate/gold_motions_micro.py
```

All annotators show the text, frame definition with behavioral anchors, and prompt for a score 1–5. Use `--debug` to test without writing files, `--max N` to limit the session to N items.

#### Phase 5 — Micro-frame extraction

LOME must be run on an HPC cluster with a GPU. Full setup instructions (Apptainer, overlay fix for modern GPUs, step-by-step pipeline) are in **[docs/lome_hpc.md](docs/lome_hpc.md)**. Once LOME outputs are available:

```bash
python3 src/utils/lome_to_csv.py motions
python3 src/utils/apply_spacy.py
sbatch scripts/exp18_classify.sh                     # LLM actor/patient classification
python3 src/utils/reconstruct_checkpoint.py          # reconstruct from checkpoint
python3 src/annotate/gold_motions_micro.py           # gold annotation (interactive)
```

#### Phase 6 — Validation

```bash
python3 src/validate/compare_models.py               # model vs. gold comparison (motions)
python3 src/validate/validate_models.py              # full validation table (all models, all text types)
```

Outputs land in `results/validation/`.

#### Phase 7 — Analysis

```bash
python3 src/analyse/macro_analysis.py               # party profiles, bloc comparisons, temporal trends
python3 src/analyse/linguistic_analysis.py          # passive voice, negation, modality
python3 src/analyse/tfidf.py                        # TF-IDF rankings per party
python3 src/analyse/party_similarity.py             # party cosine similarity (Figure 5.9)
python3 src/analyse/cooccurrence_issue_lexical.py   # PPMI contextual collocations (Table 5.8)
python3 src/analyse/count_lome_frames.py            # LOME FF-ICF fingerprints + temporal trends
python3 src/analyse/negation_in_frames.py           # negation within LOME frames (§5.3)
python3 src/analyse/polarisation.py                 # polarisation over time
python3 src/analyse/shock_analysis.py               # 2019 nitrogen case study (§5.4)
python3 src/analyse/negation_passive.py             # negation and passive voice by party
python3 src/analyse/event_analysis.py               # framing around key events
```

All analysis outputs (CSVs + figures) land in `results/analysis/`.

---

## Frame definitions

Seven macro-frames, behaviorally anchored (Biesterbos 2025 v4). Full definitions with decision rules and anchors are in [`src/annotate/frames.py`](src/annotate/frames.py).

| Frame | Label | Core question |
|-------|-------|---------------|
| `economic` | Economic | Is the argument structured around costs, jobs, or competitiveness? |
| `moral` | Moral & Ethical | Is duty, fairness, or justice the primary justification? |
| `scientific` | Scientific | Does the text invoke empirical evidence as the central argument? |
| `security` | Security | Is geopolitical stability or national interest the frame? |
| `health_environment` | Health/Environment | Are health impacts or ecosystem damage foregrounded? |
| `crisis_urgency` | Crisis & Urgency | Is existential threat or irreversibility the driving logic? |
| `weaponization` | Weaponization | Is climate used as a political conflict instrument? |

Scores: 1 = Absent, 2 = Marginal, 3 = Present, 4 = Prominent, 5 = Dominant.

---

## Parties and blocs

| Party | Bloc |
|-------|------|
| GroenLinks, PvdA, GroenLinks-PvdA, PvdD | Left |
| D66, CDA | Center |
| VVD, PVV, FvD, BBB | Right |

---

## Key shock events

| Year | Event |
|------|-------|
| 2015 | Paris Agreement (12 December) |
| 2019 | Council of State nitrogen ruling (29 May); Supreme Court Urgenda ruling (20 December) |
| 2021 | IPCC AR6 report (9 August) |
| 2022 | Russian invasion of Ukraine (24 February) |

---

## HPC configuration

Scripts with GPU requirements were run on a SLURM cluster with:
- GPU: NVIDIA RTX Pro 6000 (48 GB VRAM)
- RAM: 64 GB
- CPUs: 4

To adapt to a different cluster, update the paths in `scripts/*.sh` (project root and venv location) and the `HF_HOME` environment variable for HuggingFace model caching.

---

## Dataset: DutchClimateParl

The corpus is released as **DutchClimateParl** — 26,164 annotated texts across three genres with seven LLM-derived issue-frame scores per text.

The dataset is available for download at: *[HuggingFace DutchClimateParl](https://huggingface.co/datasets/niekbiesterbos/dutch-climate-parl)*

---

## Citation

Biesterbos, N. (2025). *Framing the Climate Crisis: Ideological Divergence in Dutch Parliamentary Discourse (2008–2025)*. Master's thesis, Radboud University / Universiteit van Amsterdam.
