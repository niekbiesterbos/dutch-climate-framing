"""
PPMI Contextual Collocations per Party (§4.1.3, §5.2.1 Table 5.8)
===================================================================
Computes Positive Pointwise Mutual Information (PPMI) scores for context
words that appear within a symmetric ±7-word window around seven target
climate terms, separately per party. This reveals the ideologically distinct
lexical environments in which parties embed shared climate vocabulary.

Target terms: klimaat, energie, uitstoot, natuur, subsidie, stikstof, duurzaam
(the seven best-attested terms across all text types; §4.1.3).

Input:
    results/motions/macro_scores/qwen2.5-32b.csv
    results/manifestos/macro_scores/qwen2.5-32b.csv
    results/speeches/macro_scores/qwen2.5-32b.csv

Output:
    results/analysis/
        cooccurrence_ppmi_party.csv   full PPMI table (party × term × word)
        figures/ppmi_top_collocations.pdf
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

OUT_DIR = Path('results/analysis')
FIG_DIR = OUT_DIR / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TERMS = ['klimaat', 'energie', 'uitstoot', 'natuur',
                'subsidie', 'stikstof', 'duurzaam']

TARGET_PARTIES = ['GroenLinks', 'PvdA', 'GroenLinks-PvdA', 'D66', 'CDA',
                  'VVD', 'PVV', 'FvD', 'BBB', 'PvdD']

PARTY_NORM = {
    'groenlinks': 'GroenLinks', 'pvda': 'PvdA',
    'groenlinks-pvda': 'GroenLinks-PvdA', 'd66': 'D66',
    'cda': 'CDA', 'vvd': 'VVD', 'pvv': 'PVV',
    'fvd': 'FvD', 'forum voor democratie': 'FvD',
    'bbb': 'BBB', 'boerburgerbeweging': 'BBB',
    'pvdd': 'PvdD', 'partij voor de dieren': 'PvdD',
}

WINDOW_SIZE = 7
TOP_K       = 5   # top PPMI words per (party, term) cell
MIN_WINDOWS = 10  # minimum context windows to report a cell

DUTCH_STOPS = set(stopwords.words('dutch'))
EXTRA_STOPS = {
    'constaterende', 'overwegende', 'verzoekt', 'motie', 'kamer',
    'regering', 'minister', 'staatssecretaris', 'tweede', 'eerste',
    'lid', 'leden', 'vergadering', 'voorzitter', 'beraadslaging',
    'gehoord', 'gelet', 'mening', 'besluit', 'gegeven', 'verklaring',
    'nederland', 'nederlands', 'kabinet', 'jaar', 'jaren',
    'heer', 'mevrouw', 'collega', 'fractie', 'partij',
    'groenlinks', 'pvda', 'pvv', 'cda', 'd66', 'vvd', 'fvd', 'bbb',
    'pvdd', 'forum', 'democratie', 'volkspartij',
}
ALL_STOPS = DUTCH_STOPS | EXTRA_STOPS

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.titlesize': 11, 'figure.dpi': 150,
})


# ── Text loading ──────────────────────────────────────────────────────────────

def normalize_party(raw: str) -> str | None:
    if not isinstance(raw, str):
        return None
    p = raw.split(';')[0].strip().lower()
    return PARTY_NORM.get(p)


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return re.sub(r'[^\w\s]', ' ', text.lower()).split()


def load_source(csv_path: str, text_col: str, party_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Always apply normalize_party: handles fractions (semicolon-separated),
    # lowercase party names (manifestos), and proper-case names (speeches).
    df['party'] = df[party_col].apply(normalize_party)
    df = df.rename(columns={text_col: 'text'})
    df = df[['party', 'text']].dropna(subset=['party', 'text'])
    df = df[df['party'].isin(TARGET_PARTIES)].copy()
    return df


def load_all() -> pd.DataFrame:
    mot = load_source('results/motions/macro_scores/qwen2.5-32b.csv',
                      text_col='normalized_text', party_col='fractions')
    man = load_source('results/manifestos/macro_scores/qwen2.5-32b.csv',
                      text_col='text', party_col='party')
    sp  = load_source('results/speeches/macro_scores/qwen2.5-32b.csv',
                      text_col='text', party_col='party_name')
    combined = pd.concat([mot, man, sp], ignore_index=True)
    print(f'Loaded {len(combined)} texts')
    print(combined.groupby('party').size().sort_values(ascending=False).to_string())
    return combined


# ── Window extraction ─────────────────────────────────────────────────────────

def extract_context_tokens(text: str, term: str,
                            window: int = 7) -> list[list[str]]:
    """
    Extract lists of context tokens from symmetric windows around each
    occurrence of `term` in `text`. Prefix matches are included (stikstof
    catches stikstofcrisis etc.).
    """
    tokens = tokenize(text)
    tl     = term.lower()
    windows = []
    for i, tok in enumerate(tokens):
        if tok == tl or tok.startswith(tl):
            left  = tokens[max(0, i - window):i]
            right = tokens[i + 1:i + 1 + window]
            ctx   = [t for t in left + right
                     if t not in ALL_STOPS
                     and not t.startswith(tl)
                     and len(t) > 3
                     and t.isalpha()]
            if ctx:
                windows.append(ctx)
    return windows


def build_window_counts(df: pd.DataFrame) -> dict:
    """
    Returns nested dict:
      counts[party][term] = Counter of context tokens
      n_windows[party][term] = int  (number of context windows)
    """
    counts:    dict = defaultdict(lambda: defaultdict(Counter))
    n_windows: dict = defaultdict(lambda: defaultdict(int))

    for _, row in df.iterrows():
        party = row['party']
        text  = row['text']
        for term in TARGET_TERMS:
            wins = extract_context_tokens(text, term, WINDOW_SIZE)
            for win in wins:
                counts[party][term].update(win)
            n_windows[party][term] += len(wins)

    return counts, n_windows


# ── PPMI computation ──────────────────────────────────────────────────────────

def compute_ppmi(counts: dict, n_windows: dict) -> list[dict]:
    """
    For each (party, term) pair, compute PPMI of each context word relative
    to all other parties for the same term (Bullinaria & Levy 2007).

    P(w | party, term) = count(w, party, term) / total_tokens(party, term)
    P(w | ALL,   term) = count(w, ALL,   term) / total_tokens(ALL, term)
    PPMI = max(0, log2(P(w | party, term) / P(w | ALL, term)))
    """
    rows = []

    for term in TARGET_TERMS:
        # Aggregate reference corpus: all parties combined for this term
        ref_counter: Counter = Counter()
        for party in TARGET_PARTIES:
            ref_counter.update(counts[party][term])
        N_ref = max(sum(ref_counter.values()), 1)

        for party in TARGET_PARTIES:
            party_counter = counts[party][term]
            N_party = max(sum(party_counter.values()), 1)
            n_win   = n_windows[party][term]

            if n_win < MIN_WINDOWS:
                continue

            for word, cnt in party_counter.items():
                if cnt < 2:
                    continue
                p_w_given_party = cnt / N_party
                p_w_ref         = ref_counter.get(word, 0) / N_ref
                if p_w_ref <= 0:
                    continue
                pmi  = np.log2(p_w_given_party / p_w_ref)
                ppmi = max(0.0, pmi)
                rows.append({
                    'party': party,
                    'term':  term,
                    'word':  word,
                    'ppmi':  round(ppmi, 4),
                    'count': cnt,
                    'n_windows': n_win,
                })

    return rows


# ── Output ────────────────────────────────────────────────────────────────────

def top_collocations(df: pd.DataFrame, top_k: int = TOP_K) -> pd.DataFrame:
    """Return top-k PPMI words per (party, term) pair."""
    return (df.sort_values('ppmi', ascending=False)
              .groupby(['party', 'term'])
              .head(top_k)
              .reset_index(drop=True))


def print_table(top: pd.DataFrame):
    """Print Table 5.8 in a readable terminal format."""
    print('\n=== Table 5.8: PPMI Collocations per Party × Term ===\n')
    for term in TARGET_TERMS:
        print(f'─ {term} ─')
        sub = top[top['term'] == term].copy()
        for party in TARGET_PARTIES:
            psub = sub[sub['party'] == party]
            if psub.empty:
                continue
            words   = ', '.join(psub['word'].tolist())
            n_win   = psub['n_windows'].iloc[0]
            print(f'  {party:<20} N={n_win:<5}  {words}')
        print()


def plot_ppmi_heatmap(top: pd.DataFrame):
    """
    For each target term: heatmap with parties on one axis and top collocations
    on the other, coloured by PPMI score. Saved as one multi-panel PDF.
    """
    n_terms = len(TARGET_TERMS)
    fig, axes = plt.subplots(n_terms, 1,
                             figsize=(14, n_terms * 2.8),
                             constrained_layout=True)

    for ax, term in zip(axes, TARGET_TERMS):
        sub = top[top['term'] == term].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        # Union of top words across all parties for this term
        top_words = (sub.groupby('word')['ppmi'].max()
                     .sort_values(ascending=False)
                     .head(15).index.tolist())
        parties   = [p for p in TARGET_PARTIES
                     if p in sub['party'].unique()]

        matrix = pd.DataFrame(0.0, index=parties, columns=top_words)
        for _, row in sub.iterrows():
            if row['word'] in matrix.columns and row['party'] in matrix.index:
                matrix.loc[row['party'], row['word']] = row['ppmi']

        im = ax.imshow(matrix.values, aspect='auto', cmap='YlOrRd',
                       vmin=0, vmax=matrix.values.max())
        ax.set_xticks(range(len(top_words)))
        ax.set_xticklabels(top_words, rotation=35, ha='right', fontsize=8)
        ax.set_yticks(range(len(parties)))
        ax.set_yticklabels(parties, fontsize=8)
        ax.set_title(f'PPMI collocations — {term}', fontsize=10, fontweight='bold')
        fig.colorbar(im, ax=ax, label='PPMI', fraction=0.015, pad=0.01)

    fig.suptitle('Contextual Collocations per Party (PPMI, ±7-word window)',
                 fontsize=12)
    plt.savefig(FIG_DIR / 'ppmi_top_collocations.pdf', bbox_inches='tight')
    plt.close()
    print('Saved: ppmi_top_collocations.pdf')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('Loading corpora...')
    df = load_all()

    print('\nExtracting context windows...')
    counts, n_windows = build_window_counts(df)

    for term in TARGET_TERMS:
        total = sum(n_windows[p][term] for p in TARGET_PARTIES)
        print(f'  {term:<12} {total} windows across all parties')

    print('\nComputing PPMI...')
    rows    = compute_ppmi(counts, n_windows)
    ppmi_df = pd.DataFrame(rows)
    ppmi_df.to_csv(OUT_DIR / 'cooccurrence_ppmi_party.csv', index=False)
    print(f'Saved {len(ppmi_df)} (party, term, word) rows to cooccurrence_ppmi_party.csv')

    top = top_collocations(ppmi_df)
    print_table(top)

    print('Generating figure...')
    plot_ppmi_heatmap(top)

    print(f'\nAll outputs saved to {OUT_DIR}')


if __name__ == '__main__':
    main()
