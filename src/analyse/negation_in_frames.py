"""
Negation within LOME Frames
======================================
Couples linguistic negation (EXP_23) with LOME frame occurrences (EXP_25)
to test whether specific frames are more negation-heavy per party.

Input:
    results/analysis/motions_linguistic.csv
    results/analysis/manifestos_linguistic.csv
    results/analysis/speeches_linguistic.csv     # optional, skip if missing
    results/analysis/frame_occurrences_motions.csv
    results/analysis/frame_occurrences_manifestos.csv

Output:
    results/analysis/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

OUT_DIR = Path('results/analysis')
FIG_DIR = OUT_DIR / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = ['GroenLinks', 'PvdA', 'GroenLinks-PvdA', 'D66', 'CDA',
          'VVD', 'PVV', 'FvD', 'BBB', 'PvdD']

LEFT   = ['GroenLinks', 'PvdA', 'GroenLinks-PvdA', 'PvdD']
CENTER = ['D66', 'CDA']
RIGHT  = ['VVD', 'PVV', 'FvD', 'BBB']

PARTY_COLORS = {
    'GroenLinks': '#2e8b57', 'PvdA': '#e63946', 'GroenLinks-PvdA': '#4a7c59',
    'D66': '#00b4d8', 'CDA': '#52b788', 'VVD': '#f4a261',
    'PVV': '#023e8a', 'FvD': '#6d023e', 'BBB': '#95d5b2', 'PvdD': '#74c69d',
}

GENRE_FRAMES = {
    'Statement', 'Cogitation', 'Leadership', 'Discussion',
    'Attempt_suasion', 'Hearsay', 'Request', 'Topic',
}


def load_source(linguistic_path: str, frame_path: str,
                id_col: str, source_label: str) -> pd.DataFrame | None:
    """
    Load and merge linguistic features with frame occurrences for one source.
    Returns None if linguistic_path does not exist (e.g. speeches not yet done).
    """
    lp = Path(linguistic_path)
    fp = Path(frame_path)

    if not lp.exists():
        print(f'  Skipping {source_label}: {lp} not found')
        return None
    if not fp.exists():
        print(f'  Skipping {source_label}: {fp} not found')
        return None

    ling  = pd.read_csv(lp)
    frames = pd.read_csv(fp)

    print(f'  {source_label}: {len(ling)} linguistic rows, {len(frames)} frame rows')

    # Pivot frames: one row per doc, one column per frame (count)
    frame_counts = (
        frames[~frames['frame'].isin(GENRE_FRAMES)]
        .groupby([id_col, 'frame'])
        .size()
        .reset_index(name='count')
        .pivot(index=id_col, columns='frame', values='count')
        .fillna(0)
    )
    frame_counts.columns.name = None
    frame_counts = frame_counts.reset_index()

    # Merge on id_col
    frame_counts[id_col] = frame_counts[id_col].astype(str)
    if 'motie_id' in ling.columns and id_col == 'doc_id':
        ling = ling.rename(columns={'motie_id': 'doc_id'})
    ling[id_col] = ling[id_col].astype(str)
    merged = ling.merge(frame_counts, on=id_col, how='inner')
    merged['source'] = source_label
    print(f'  After merge: {len(merged)} rows')
    return merged


def negation_by_frame_party(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    For ALL non-genre frames, compute mean negation proportion per party
    for docs that have >= 1 occurrence of that frame.
    Ranked by negation level to reveal which frames attract most negation.
    """
    all_frames = [c for c in df.columns
                  if c not in GENRE_FRAMES
                  and c not in ['motie_id', 'doc_id', 'party', 'year',
                                'text', 'bloc', 'source',
                                'neg_prop_negated', 'neg_n_negated',  
                                'voice_prop_passive', 'voice_prop_active',
                                'mod_prop_obligation', 'mod_prop_possibility',
                                'mod_prop_hedging', 'mod_prop_negation_mod',
                                'mod_prop_any_modal']]

    rows = []
    for frame in all_frames:
        has_frame = df[df[frame] >= 1]
        no_frame  = df[df[frame] == 0]
        if len(has_frame) < 10:
            continue
        rows.append({
            'source': source,
            'frame':  frame,
            'party':  'ALL',
            'neg_with_frame':    has_frame['neg_prop_negated'].mean(),
            'neg_without_frame': no_frame['neg_prop_negated'].mean(),
            'delta':  has_frame['neg_prop_negated'].mean() - no_frame['neg_prop_negated'].mean(),
            'n_with': len(has_frame),
        })
        for party in TARGET:
            pf  = has_frame[has_frame['party'] == party]['neg_prop_negated']
            pnf = no_frame[no_frame['party'] == party]['neg_prop_negated']
            if len(pf) < 5:
                continue
            rows.append({
                'source': source,
                'frame':  frame,
                'party':  party,
                'neg_with_frame':    pf.mean(),
                'neg_without_frame': pnf.mean() if len(pnf) > 0 else None,
                'delta':  pf.mean() - (pnf.mean() if len(pnf) > 0 else 0),
                'n_with': len(pf),
            })
    return pd.DataFrame(rows)

def plot_negation_by_frame(result: pd.DataFrame, source: str):
    """
    Heatmap: rows = frames, cols = parties, values = neg_prop when frame present.
    """
    sub = result[(result['source'] == source) & (result['party'] != 'ALL')]
    if sub.empty:
        return

    pivot = sub.pivot(index='frame', columns='party', values='neg_with_frame')
    # Keep only parties we have
    cols = [p for p in TARGET if p in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(len(cols) * 1.2 + 2, len(pivot) * 0.7 + 1.5))
    sns.heatmap(pivot.astype(float), ax=ax, cmap='YlOrRd',
                annot=True, fmt='.2f', linewidths=0.4,
                cbar_kws={'label': 'Mean negation proportion'})
    ax.set_title(f'Negation within LOME Frames — {source}')
    ax.tick_params(axis='x', rotation=40)
    plt.tight_layout()
    fname = f'negation_by_frame_{source.lower()}.pdf'
    plt.savefig(FIG_DIR / fname, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fname}')


def plot_frame_neg_delta(result: pd.DataFrame, source: str):
    """
    Bar chart: for each frame, difference in negation (with - without frame), ALL parties.
    Shows which frames attract more negation overall.
    """
    sub = result[(result['source'] == source) & (result['party'] == 'ALL')].copy()
    if sub.empty:
        return
    sub['delta'] = sub['neg_with_frame'] - sub['neg_without_frame']
    sub = sub.sort_values('delta', ascending=True)

    colors = ['#e63946' if d < 0 else '#2e8b57' for d in sub['delta']]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(sub['frame'], sub['delta'], color=colors, alpha=0.85)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Δ negation (with frame − without frame)')
    ax.set_title(f'Negation Delta by Frame — {source}')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    fname = f'negation_delta_{source.lower()}.pdf'
    plt.savefig(FIG_DIR / fname, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fname}')


def main():
    sources = [
        {
            'label':    'motions',
            'ling':     'results/analysis/motions_linguistic.csv',
            'frames':   'results/analysis/frame_occurrences_motions.csv',
            'id_col':   'doc_id', 
        },
        {
            'label':    'manifestos',
            'ling':     'results/analysis/manifestos_linguistic.csv',
            'frames':   'results/analysis/frame_occurrences_manifestos.csv',
            'id_col':   'doc_id',
        },
        {
            'label':    'speeches',
            'ling':     'results/analysis/speeches_linguistic.csv',
            'frames':   'results/analysis/frame_occurrences_speeches.csv',
            'id_col':   'doc_id',  
        },
    ]

    all_results = []

    for s in sources:
        print(f'\nLoading {s["label"]}...')
        df = load_source(s['ling'], s['frames'], s['id_col'], s['label'])
        if df is None:
            continue

        result = negation_by_frame_party(df, s['label'])
        all_results.append(result)

        print(f'\n--- {s["label"].upper()} — top 15 frames by negation (ALL parties) ---')
        sub = result[result['party'] == 'ALL'].sort_values(
            'neg_with_frame', ascending=False).head(15)
        print(sub[['frame', 'neg_with_frame', 'neg_without_frame',
                    'delta', 'n_with']].to_string(index=False))

        print(f'\n--- top 15 frames by negation DELTA (ALL parties) ---')
        sub2 = result[result['party'] == 'ALL'].sort_values(
            'delta', ascending=False).head(15)
        print(sub2[['frame', 'neg_with_frame', 'neg_without_frame',
                     'delta', 'n_with']].to_string(index=False))

        plot_negation_by_frame(result, s['label'])
        plot_frame_neg_delta(result, s['label'])

    if all_results:
        combined = pd.concat(all_results)
        combined.to_csv(OUT_DIR / 'negation_by_frame_party.csv', index=False)
        print(f'\nAll outputs saved to {OUT_DIR}')


if __name__ == '__main__':
    main()