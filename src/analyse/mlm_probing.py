"""
Masked Target Prediction for Party-Specific Climate Framing
=====================================================================
Implements the masked target prediction method of Hoeken et al. (2023,
EMNLP Findings) for detecting subtle semantic shifts between political
communities in Dutch.

Method (Section 4.1 of Hoeken et al.):
    For each target word and each party:
    1. Extract all sentences containing the target word from the party corpus
    2. Mask the target word in each sentence
    3. Predict top-k substitutions using RobBERT (no fine-tuning)
    4. Aggregate substitution frequencies across all sentences
    5. Compare frequency distributions between parties via Jensen-Shannon
       Divergence (JSD) — higher JSD = more distinct semantic associations

This approach differs from standard MLM probing in that it uses real
corpus sentences rather than constructed probe sentences, and measures
the aggregate distribution of substitutions rather than single predictions.

Reference:
    Hoeken, S., Alaçam, Ö., Fokkens, A., & Sommerauer, P. (2023).
    Methodological Insights in Detecting Subtle Semantic Shifts with
    Contextualized and Static Language Models. EMNLP Findings.

Input:
    results/analysis/motions_linguistic.csv
    results/analysis/manifestos_linguistic.csv
    results/analysis/speeches_linguistic.csv

Output:
    results/analysis/
        substitutions/          -- per party per term: token frequencies
        jsd_matrix.csv          -- pairwise JSD between parties per term
        top_substitutions.csv   -- top-k substitutions per party per term
        figures/                -- heatmaps and comparison plots
"""

import os
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from transformers import AutoTokenizer, AutoModelForMaskedLM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

OUT_DIR  = Path('results/analysis')
SUB_DIR  = OUT_DIR / 'substitutions'
FIG_DIR  = OUT_DIR / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = 'pdelobelle/robbert-v2-dutch-base'

# ── Configuration ─────────────────────────────────────────────────────────────

# Toggle parties — add or remove freely
PARTIES_TO_RUN = [
    'PVV',
    'PvdD',
    'GroenLinks',
    'PvdA',
    'D66',
    'CDA',
    'VVD',
    'FvD',
    'BBB',
]

# Target words whose connotation we hypothesize to differ between parties
TARGET_WORDS = [
    'klimaat',
    'energie',
    'uitstoot',
    'natuur',
    'transitie',
    'biodiversiteit',
    'kernenergie',
    'subsidie',
    'industrie',
    'landbouw',
    'stikstof',
    'duurzaam',
]

TOP_K = 10   # substitutions per masked instance (following Hoeken et al.)

PARTY_MAP = {
    'd66': 'D66', 'vvd': 'VVD', 'cda': 'CDA', 'pvda': 'PvdA',
    'pvv': 'PVV', 'fvd': 'FvD', 'bbb': 'BBB', 'pvdd': 'PvdD',
    'groenlinks': 'GroenLinks', 'groenlinks-pvda': 'GroenLinks-PvdA',
}

PARTY_COLORS = {
    'GroenLinks': '#2e8b57', 'PvdA': '#e63946', 'GroenLinks-PvdA': '#4a7c59',
    'D66': '#00b4d8', 'CDA': '#52b788', 'VVD': '#f4a261',
    'PVV': '#023e8a', 'FvD': '#6d023e', 'BBB': '#95d5b2', 'PvdD': '#74c69d',
}


# ── Data loading ──────────────────────────────────────────────────────────────

def normalize_party(x: str) -> str | None:
    if not isinstance(x, str):
        return None
    return PARTY_MAP.get(x.lower().strip(), x.strip())


def split_sentences(text: str) -> list:
    """Split text into sentences on sentence-ending punctuation."""
    if not isinstance(text, str):
        return []
    sents = re.split(r'(?<=[.!?;])\s+', text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 15]


def load_sentences_per_party() -> dict:
    """
    Load all sentences per party across all three text types.
    Returns dict: party -> list of sentences.
    """
    sentences = defaultdict(list)

    sources = [
        ('motions',    'party'),
        ('manifestos', 'party'),
        ('speeches',   'party'),
    ]

    for source, party_col in sources:
        path = Path(f'results/analysis/{source}_linguistic.csv')
        df   = pd.read_csv(path)
        df['party_norm'] = df[party_col].apply(normalize_party)
        df = df[df['party_norm'].isin(PARTIES_TO_RUN)].copy()
        df = df[df['text'].notna()].copy()

        for _, row in df.iterrows():
            for sent in split_sentences(row['text']):
                sentences[row['party_norm']].append(sent)

        print(f'  {source}: loaded {len(df)} docs')

    for party in PARTIES_TO_RUN:
        print(f'  {party}: {len(sentences[party])} sentences total')

    return dict(sentences)


# ── Masked target prediction ──────────────────────────────────────────────────

def get_target_sentences(sentences: list, target: str,
                          max_sentences: int = 2000) -> list:
    """
    Extract sentences containing the target word.
    Cap at max_sentences to avoid excessive computation.
    """
    target_lower = target.lower()
    matching = [s for s in sentences
                if target_lower in s.lower().split()
                or f' {target_lower}' in s.lower()
                or f'{target_lower} ' in s.lower()]

    if len(matching) > max_sentences:
        # Random sample to avoid temporal bias
        rng = np.random.default_rng(42)
        matching = list(rng.choice(matching, max_sentences, replace=False))

    return matching


def predict_substitutions(sentences: list, target: str,
                           tokenizer, model, device: str,
                           top_k: int = 10,
                           batch_size: int = 32) -> Counter:
    """
    For each sentence, mask all occurrences of the target word and collect
    top-k predicted substitutions. Returns aggregate token frequency Counter.

    Follows Hoeken et al. (2023): extract top k predicted candidates for the
    target word by masking it, then compare occurrence frequency of unique
    tokens across subcorpora.
    """
    mask_token = tokenizer.mask_token  # '<mask>' for RobBERT
    counter    = Counter()
    n_masked   = 0

    # Process in batches
    for batch_start in range(0, len(sentences), batch_size):
        batch = sentences[batch_start:batch_start + batch_size]

        # Mask target word in each sentence (case-insensitive, whole word)
        masked_batch = []
        sent_idx     = []
        for i, sent in enumerate(batch):
            masked = re.sub(
                rf'(?i)\b{re.escape(target)}\b',
                mask_token,
                sent
            )
            # Only include if masking actually occurred
            if mask_token in masked:
                masked_batch.append(masked)
                sent_idx.append(i)

        if not masked_batch:
            continue

        # Tokenize
        try:
            encoded = tokenizer(
                masked_batch,
                return_tensors='pt',
                truncation=True,
                max_length=256,
                padding=True,
            ).to(device)
        except Exception:
            continue

        # Find mask positions
        mask_id      = tokenizer.mask_token_id
        mask_pos_all = (encoded['input_ids'] == mask_id).nonzero(as_tuple=False)

        if len(mask_pos_all) == 0:
            continue

        # Forward pass
        with torch.no_grad():
            logits = model(**encoded).logits

        # Extract top-k predictions for each masked position
        for row_idx, col_idx in mask_pos_all:
            row_logits = logits[row_idx, col_idx, :]
            top_ids    = row_logits.topk(top_k).indices.tolist()
            tokens     = [tokenizer.decode([t]).strip().lower()
                          for t in top_ids]
            # Filter: keep alphabetic tokens of length > 2
            tokens = [t for t in tokens
                      if t.isalpha() and len(t) > 2 and t != target.lower()]
            counter.update(tokens)
            n_masked += 1

    return counter, n_masked


# ── Jensen-Shannon Divergence ─────────────────────────────────────────────────

def counters_to_distribution(c1: Counter, c2: Counter) -> tuple:
    """
    Convert two counters to aligned probability distributions over their
    union vocabulary. Following Hoeken et al.: compare tokens in the
    intersection of the two substitution sets.
    """
    # Use intersection vocabulary (tokens appearing in both)
    vocab = sorted(set(c1.keys()) & set(c2.keys()))
    if not vocab:
        return None, None, vocab

    v1 = np.array([c1[t] for t in vocab], dtype=float)
    v2 = np.array([c2[t] for t in vocab], dtype=float)

    # Normalize to probabilities
    v1 = v1 / v1.sum() if v1.sum() > 0 else v1
    v2 = v2 / v2.sum() if v2.sum() > 0 else v2

    return v1, v2, vocab


def compute_jsd(c1: Counter, c2: Counter) -> float:
    """Compute Jensen-Shannon Divergence between two substitution counters."""
    v1, v2, vocab = counters_to_distribution(c1, c2)
    if v1 is None:
        return np.nan
    return float(jensenshannon(v1, v2, base=2))


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_jsd_heatmap(jsd_df: pd.DataFrame, target: str):
    """
    Heatmap of pairwise Jensen-Shannon Divergence between parties
    for a given target word. Higher JSD = more distinct semantic associations.
    """
    parties = [p for p in PARTIES_TO_RUN if p in jsd_df.index]
    sub     = jsd_df.loc[parties, parties]

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.eye(len(parties), dtype=bool)
    sns.heatmap(
        sub.astype(float), ax=ax, cmap='YlOrRd',
        annot=True, fmt='.3f', linewidths=0.4, mask=mask,
        vmin=0, vmax=0.5,
        cbar_kws={'label': 'Jensen-Shannon Divergence'}
    )
    for lbl in ax.get_xticklabels():
        lbl.set_color(PARTY_COLORS.get(lbl.get_text(), '#333'))
        lbl.set_fontweight('bold')
    for lbl in ax.get_yticklabels():
        lbl.set_color(PARTY_COLORS.get(lbl.get_text(), '#333'))
        lbl.set_fontweight('bold')
    ax.set_title(
        f'Semantic distinctiveness of "{target}" per party pair\n'
        f'(Jensen-Shannon Divergence on masked substitution distributions)',
        fontsize=11)
    ax.tick_params(axis='x', rotation=40)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'jsd_{target}.pdf', bbox_inches='tight')
    plt.close()


def plot_top_substitutions(counters: dict, target: str, top_n: int = 15):
    """
    For each party, plot the top-n most frequent substitutions for the
    target word as horizontal bar charts.
    """
    parties = [p for p in PARTIES_TO_RUN if p in counters and counters[p]]
    n_cols  = min(3, len(parties))
    n_rows  = -(-len(parties) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 5, n_rows * 3.5),
                              constrained_layout=True)
    if len(parties) == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, party in enumerate(parties):
        ax      = axes[i]
        counter = counters[party]
        total   = sum(counter.values())
        if total == 0:
            ax.set_visible(False)
            continue
        top     = counter.most_common(top_n)
        tokens  = [t for t, _ in top][::-1]
        freqs   = [c / total for _, c in top][::-1]
        color   = PARTY_COLORS.get(party, '#555')
        ax.barh(tokens, freqs, color=color, alpha=0.85)
        ax.set_title(party, fontweight='bold', color=color, fontsize=10)
        ax.set_xlabel('Relative frequency', fontsize=8)
        ax.grid(True, alpha=0.3, axis='x')
        ax.tick_params(labelsize=8)

    for j in range(len(parties), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f'Top substitutions for "{target}" per party\n'
        f'(RobBERT masked target prediction, top-{top_n} tokens)',
        fontsize=12, y=1.01)
    plt.savefig(FIG_DIR / f'substitutions_{target}.pdf', bbox_inches='tight')
    plt.close()
    print(f'  Saved: substitutions_{target}.pdf')


def plot_contrastive_substitutions(counters: dict, target: str,
                                    party_a: str, party_b: str,
                                    top_n: int = 20):
    """
    Diverging bar chart showing tokens with the highest relative frequency
    difference between two parties for a given target word.
    Positive = more distinctive for party_a, negative = party_b.
    """
    if party_a not in counters or party_b not in counters:
        return

    c1, c2 = counters[party_a], counters[party_b]
    vocab  = sorted(set(c1.keys()) | set(c2.keys()))

    t1 = sum(c1.values())
    t2 = sum(c2.values())
    if t1 == 0 or t2 == 0:
        return

    diffs = {t: (c1[t] / t1) - (c2[t] / t2) for t in vocab}
    sorted_diffs = sorted(diffs.items(), key=lambda x: x[1])

    bottom = sorted_diffs[:top_n // 2]
    top    = sorted_diffs[-(top_n // 2):]
    combined = bottom + top

    tokens = [t for t, _ in combined]
    values = [v for _, v in combined]
    colors = [PARTY_COLORS.get(party_b, '#e63946') if v < 0
              else PARTY_COLORS.get(party_a, '#2e8b57') for v in values]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(tokens, values, color=colors, alpha=0.85)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel(f'← {party_b} distinctive   |   {party_a} distinctive →')
    ax.set_title(
        f'Contrastive substitutions for "{target}"\n'
        f'({party_a} vs {party_b}, relative frequency difference)',
        fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    fname = f'contrastive_{target}_{party_a}_vs_{party_b}.pdf'
    plt.savefig(FIG_DIR / fname, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {fname}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f'Loading model: {BASE_MODEL}')
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForMaskedLM.from_pretrained(BASE_MODEL).to(device)
    model.eval()
    print(f'Model loaded on {device}')

    print('\nLoading sentences...')
    party_sentences = load_sentences_per_party()

    all_substitutions = {}  # target -> party -> Counter
    all_jsd_rows      = []  # for summary CSV

    for target in TARGET_WORDS:
        print(f'\n{"="*55}')
        print(f'Target: "{target}"')
        print(f'{"="*55}')

        counters = {}

        for party in PARTIES_TO_RUN:
            sentences = party_sentences.get(party, [])
            target_sents = get_target_sentences(sentences, target)

            if len(target_sents) < 5:
                print(f'  {party:<15} — too few sentences ({len(target_sents)}), skipping')
                continue

            counter, n_masked = predict_substitutions(
                target_sents, target, tokenizer, model, device, top_k=TOP_K
            )
            counters[party] = counter

            print(f'  {party:<15} n_sents={len(target_sents):>4}  '
                  f'n_masked={n_masked:>4}  '
                  f'unique_subs={len(counter):>4}  '
                  f'top3={", ".join(t for t, _ in counter.most_common(3))}')

            # Save substitution frequencies
            sub_df = pd.DataFrame(counter.most_common(100),
                                  columns=['token', 'count'])
            sub_df['party']  = party
            sub_df['target'] = target
            sub_df.to_csv(SUB_DIR / f'{target}_{party}.csv', index=False)

        all_substitutions[target] = counters

        if len(counters) < 2:
            print(f'  Not enough parties for JSD, skipping')
            continue

        # Pairwise JSD matrix
        parties_available = list(counters.keys())
        jsd_matrix = pd.DataFrame(
            np.zeros((len(parties_available), len(parties_available))),
            index=parties_available, columns=parties_available
        )
        for i, p1 in enumerate(parties_available):
            for j, p2 in enumerate(parties_available):
                if i >= j:
                    continue
                jsd = compute_jsd(counters[p1], counters[p2])
                jsd_matrix.loc[p1, p2] = jsd
                jsd_matrix.loc[p2, p1] = jsd
                all_jsd_rows.append({
                    'target': target, 'party_a': p1, 'party_b': p2, 'jsd': jsd
                })

        jsd_matrix.to_csv(OUT_DIR / f'jsd_{target}.csv')

        # Most distinctive pair
        flat = [(p1, p2, jsd_matrix.loc[p1, p2])
                for p1 in parties_available
                for p2 in parties_available if p1 < p2]
        if flat:
            most_diff = max(flat, key=lambda x: x[2])
            print(f'  Most distinct pair: {most_diff[0]} vs {most_diff[1]} '
                  f'(JSD={most_diff[2]:.3f})')

        # Figures
        plot_jsd_heatmap(jsd_matrix, target)
        plot_top_substitutions(counters, target, top_n=15)

        # Contrastive plots for key ideological pairs
        for p_a, p_b in [('PVV', 'PvdD'), ('FvD', 'GroenLinks'),
                          ('BBB', 'PvdD'), ('VVD', 'GroenLinks')]:
            if p_a in counters and p_b in counters:
                plot_contrastive_substitutions(counters, target, p_a, p_b)

    # Summary JSD table
    jsd_summary = pd.DataFrame(all_jsd_rows)
    jsd_summary.to_csv(OUT_DIR / 'jsd_all.csv', index=False)

    # Print summary: most distinctive party pair per target
    print('\n\n=== Summary: most semantically distinct party pairs per term ===')
    for target in TARGET_WORDS:
        sub = jsd_summary[jsd_summary['target'] == target]
        if sub.empty:
            continue
        top = sub.loc[sub['jsd'].idxmax()]
        print(f'  {target:<18} {top["party_a"]} vs {top["party_b"]} '
              f'(JSD={top["jsd"]:.3f})')

    print(f'\nAll outputs saved to {OUT_DIR}')


if __name__ == '__main__':
    main()
