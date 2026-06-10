"""
Contrastive Context Window Analysis (party-vs-all)
============================================================
For each target climate term, extracts ±7-word context windows around
each occurrence per party and computes contrastive TF-IDF to identify
which words each party distinctively uses near the target term relative
to ALL other parties combined.

Input:
    results/analysis/{motions,manifestos,speeches}_linguistic.csv
    results/motions/macro_scores/19/20 scores (for all-ones exclusion)

Output:
    results/analysis/
        contrastive_context_party_vs_all.csv
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

OUT_DIR = Path('results/analysis')
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_WORDS = [
    'klimaat', 'energie', 'uitstoot', 'natuur', 'transitie',
    'biodiversiteit', 'kernenergie', 'subsidie', 'industrie',
    'landbouw', 'stikstof', 'duurzaam',
]

PARTIES = [
    'PVV', 'FvD', 'BBB', 'VVD', 'CDA', 'D66',
    'GroenLinks', 'PvdA', 'PvdD',
]

TABLE_PARTIES = ['PVV', 'FvD', 'BBB', 'VVD', 'CDA', 'D66', 'GroenLinks', 'PvdA', 'PvdD']

WINDOW_SIZE = 7

PARTY_MAP = {
    'd66': 'D66', 'vvd': 'VVD', 'cda': 'CDA', 'pvda': 'PvdA',
    'pvv': 'PVV', 'fvd': 'FvD', 'bbb': 'BBB', 'pvdd': 'PvdD',
    'groenlinks': 'GroenLinks', 'groenlinks-pvda': 'GroenLinks-PvdA',
}

DUTCH_STOPS = set(stopwords.words('dutch'))
EXTRA_STOPS = {
    'constaterende', 'overwegende', 'verzoekt', 'motie', 'kamer',
    'regering', 'minister', 'staatssecretaris', 'tweede', 'eerste',
    'lid', 'leden', 'vergadering', 'voorzitter', 'beraadslaging',
    'gehoord', 'gelet', 'mening', 'besluit', 'gegeven', 'verklaring',
    'groenlinks', 'pvda', 'pvv', 'cda', 'd66', 'vvd', 'fvd', 'bbb',
    'pvdd', 'partij', 'fractie',
    'nederland', 'nederlands', 'kabinet', 'jaar', 'jaren',
    'heel', 'zeer', 'echt', 'gewoon', 'eigenlijk',
    'heer', 'mevrouw', 'voorzitter', 'collega',
    'ouwehand', 'thieme', 'wilders', 'klaver', 'rutte', 'kaag',
    'tongeren', 'thijssen', 'kröger', 'bromet', 'veldhoven',
    'mulder', 'plas', 'wassenberg', 'vestering', 'esch',
    'dijkstra', 'veltman', 'grashoff', 'moorlag',
}
ALL_STOPS = DUTCH_STOPS | EXTRA_STOPS

FRAME_COLS = ['economic', 'moral', 'scientific', 'security',
              'health_environment', 'crisis_urgency', 'weaponization']

SCORE_PATHS = {
    'motions':   ('results/motions/macro_scores/qwen2.5-32b.csv',   'id'),
    'manifestos':('results/manifestos/macro_scores/qwen2.5-32b.csv', 'text'),
    'speeches':  ('results/speeches/macro_scores/qwen2.5-32b.csv', 'doc_id'),
}


def load_all_ones_ids() -> dict:
    """Return set of all-ones IDs per source."""
    all_ones = {}
    for source, (path, id_col) in SCORE_PATHS.items():
        df   = pd.read_csv(path)
        mask = (df[FRAME_COLS] == 1).all(axis=1)
        all_ones[source] = set(df.loc[mask, id_col].astype(str))
        print(f'  {source}: excluding {mask.sum()} all-ones items')
    return all_ones


def normalize_party(x):
    if not isinstance(x, str):
        return None
    return PARTY_MAP.get(x.lower().strip(), x.strip())


def tokenize(text):
    if not isinstance(text, str):
        return []
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return [t for t in text.split() if len(t) > 2]


def extract_windows(text, target, window=7):
    tokens = tokenize(text)
    tl = target.lower()
    out = []
    for i, tok in enumerate(tokens):
        if tok == tl or tok.startswith(tl):
            ctx = [t for t in
                   (tokens[max(0, i-window):i] + tokens[i+1:i+1+window])
                   if t not in ALL_STOPS
                   and not t.startswith(tl)
                   and len(t) > 3]
            if ctx:
                out.append(' '.join(ctx))
    return out


def load_all_windows(target, all_ones_ids):
    """Load context windows per party for target, across all sources."""
    party_windows = defaultdict(list)
    for source in ['motions', 'manifestos', 'speeches']:
        path = Path(f'results/analysis/{source}_linguistic.csv')
        df   = pd.read_csv(path)
        if 'party' not in df.columns:
            continue

        # filter all-ones
        id_col = {'motions': 'id', 'manifestos': 'text', 'speeches': 'doc_id'}[source]
        if id_col in df.columns:
            df[id_col] = df[id_col].astype(str)
            df = df[~df[id_col].isin(all_ones_ids[source])]

        df['party_norm'] = df['party'].apply(normalize_party)
        df = df[df['party_norm'].isin(PARTIES) & df['text'].notna()]
        for _, row in df.iterrows():
            wins = extract_windows(row['text'], target, WINDOW_SIZE)
            party_windows[row['party_norm']].extend(wins)
    return dict(party_windows)


def contrastive_vs_all(focal_docs, all_party_windows, focal_party, top_k=5):
    """Party-vs-all contrastive TF-IDF."""
    other_docs = [w for p, ws in all_party_windows.items()
                  if p != focal_party for w in ws]
    if len(focal_docs) < 3 or len(other_docs) < 3:
        return []
    all_docs = focal_docs + other_docs
    try:
        vec = TfidfVectorizer(
            max_features=10000, min_df=2,
            stop_words=list(ALL_STOPS), sublinear_tf=True,
            token_pattern=r'\b[a-zA-Zàáâãäå]{4,}\b')
        mat   = vec.fit_transform(all_docs)
        feats = vec.get_feature_names_out()
        diff  = (mat[:len(focal_docs)].mean(axis=0).A1 -
                 mat[len(focal_docs):].mean(axis=0).A1)
        return [feats[i] for i in diff.argsort()[::-1][:top_k]
                if diff[i] > 0]
    except Exception:
        return []


def main():
    print('Loading all-ones IDs...')
    all_ones_ids = load_all_ones_ids()

    rows = []
    for target in TARGET_WORDS:
        print(f'Processing "{target}"...')
        party_windows = load_all_windows(target, all_ones_ids)
        for p, ws in party_windows.items():
            print(f'  {p:<18} {len(ws)} windows')

        for party in PARTIES:
            docs  = party_windows.get(party, [])
            words = contrastive_vs_all(docs, party_windows, party, top_k=5)
            rows.append({
                'target':           target,
                'party':            party,
                'n_windows':        len(docs),
                'distinctive_words': ', '.join(words) if words else '',
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'contrastive_context_party_vs_all.csv', index=False)

    # Print thesis table
    print('\n\n=== THESIS TABLE (party-vs-all) ===')
    pivot = df[df['party'].isin(TABLE_PARTIES)].pivot(
        index='target', columns='party', values='distinctive_words')
    pivot = pivot.reindex(columns=TABLE_PARTIES)
    pivot = pivot.reindex(TARGET_WORDS)
    for term in TARGET_WORDS:
        print(f'\n{term}:')
        for p in TABLE_PARTIES:
            print(f'  {p:<14} {pivot.loc[term, p]}')

    print(f'\nSaved to {OUT_DIR}')


if __name__ == '__main__':
    main()