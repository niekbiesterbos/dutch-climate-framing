"""
LOME Frame Occurrence Analysis
========================================
Two analytical outputs:

  1. Combined FF-ICF party fingerprints across all text types
  2. Temporal trends in key LOME frames per ideological bloc
     (parliamentary motions 2008-2025, n >= 30 per bloc per year,
      no smoothing applied)

All-ones exclusion applied consistently across all three text types,
matching the exclusion procedure used in EXP_22/EXP_24.

Input:
    results/analysis/frame_occurrences_{motions,manifestos,speeches}.csv
    results/motions/macro_scores/qwen2.5-32b.csv
    results/manifestos/macro_scores/qwen2.5-32b.csv
    results/speeches/macro_scores/qwen2.5-32b.csv

Output:
    results/analysis/
        fficf_combined.csv
        lome_temporal_bloc.csv
        figures/
            fficf_combined.pdf
            lome_temporal_bloc.pdf
"""

import os
import numpy as np
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

BLOC_MAP = ({p: 'Left'   for p in LEFT}  |
            {p: 'Center' for p in CENTER} |
            {p: 'Right'  for p in RIGHT})

BLOC_COLORS = {'Left': '#2e8b57', 'Center': '#f4a261', 'Right': '#e63946'}

FRAME_COLS = ['economic', 'moral', 'scientific', 'security',
              'health_environment', 'crisis_urgency', 'weaponization']

TEMPORAL_FRAMES = [
    'Activity_stop',
    'Required_event',
    'Origin',
    'Expensiveness',
    'Intentionally_act',
    'Protecting',
]

TEMPORAL_LABELS = {
    'Activity_stop':     'Activity_stop\n(stopping climate policy)',
    'Required_event':    'Required_event\n(obligation framing)',
    'Origin':            'Origin\n(causal attribution)',
    'Expensiveness':     'Expensiveness\n(cost framing)',
    'Intentionally_act': 'Intentionally_act\n(agency attribution)',
    'Protecting':        'Protecting\n(protection framing)',
}

GENRE_FRAMES = {
    'Statement', 'Cogitation', 'Leadership', 'Discussion',
    'Attempt_suasion', 'Hearsay', 'Request', 'Topic',
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.size':   10,
    'axes.titlesize': 11,
    'figure.dpi':  150,
})


# ── All-ones exclusion ────────────────────────────────────────────────────────

def load_valid_ids() -> dict:
    """
    Return sets of valid (non-all-ones) doc IDs per source,
    matching the exclusion procedure in EXP_22.
    """
    valid = {}

    # Motions
    df = pd.read_csv('results/motions/macro_scores/qwen2.5-32b.csv')
    mask = (df[FRAME_COLS] == 1).all(axis=1)
    valid['motions'] = set(df.loc[~mask, 'id'].astype(str))
    print(f'Motions:    {mask.sum()} all-ones excluded, '
          f'{(~mask).sum()} retained')

    # Manifestos
    df = pd.read_csv('results/manifestos/macro_scores/qwen2.5-32b.csv')
    mask = (df[FRAME_COLS] == 1).all(axis=1)
    df = df.reset_index()
    df['file_idx'] = df['index'].astype(str).str.zfill(4)
    valid['manifestos'] = set(df.loc[~mask, 'file_idx'])
    print(f'Manifestos: {mask.sum()} all-ones excluded, '
          f'{(~mask).sum()} retained')

    # Speeches
    df = pd.read_csv('results/speeches/macro_scores/qwen2.5-32b.csv')
    mask = (df[FRAME_COLS] == 1).all(axis=1)
    valid['speeches'] = set(df.loc[~mask, 'doc_id'].astype(str))
    print(f'Speeches:   {mask.sum()} all-ones excluded, '
          f'{(~mask).sum()} retained')

    return valid


def load_and_filter(valid_ids: dict) -> tuple:
    """
    Load pre-computed frame occurrence CSVs and apply all-ones exclusion.
    Returns (mot_df, man_df, sp_df).
    """
    mot_df = pd.read_csv(OUT_DIR / 'frame_occurrences_motions.csv')
    man_df = pd.read_csv(OUT_DIR / 'frame_occurrences_manifestos.csv')
    sp_df  = pd.read_csv(OUT_DIR / 'frame_occurrences_speeches.csv')

    # Manifesto doc_id is path.stem e.g. 'manifesto_0042' — extract index
    man_df['file_idx'] = (man_df['doc_id'].astype(str)
                          .str.replace('manifesto_', '')
                          .str.zfill(4))

    before = len(mot_df), len(man_df), len(sp_df)

    mot_df = mot_df[mot_df['doc_id'].astype(str).isin(valid_ids['motions'])].copy()
    man_df = man_df[man_df['file_idx'].isin(valid_ids['manifestos'])].copy()
    sp_df  = sp_df[sp_df['doc_id'].astype(str).isin(valid_ids['speeches'])].copy()

    after = len(mot_df), len(man_df), len(sp_df)
    for src, b, a in zip(['motions', 'manifestos', 'speeches'], before, after):
        print(f'  {src}: {b} -> {a} frame occurrences after all-ones exclusion')

    return mot_df, man_df, sp_df


# ── Core computation ──────────────────────────────────────────────────────────

def freq_per_doc(df: pd.DataFrame) -> pd.DataFrame:
    """Mean frame occurrences per document per party, genre frames excluded."""
    df = df[~df['frame'].isin(GENRE_FRAMES)].copy()
    counts = (df.groupby(['party', 'frame'])
              .size().reset_index(name='count'))
    doc_n  = (df.groupby('party')['doc_id']
              .nunique().reset_index(name='n_docs'))
    merged = counts.merge(doc_n, on='party')
    merged['freq'] = merged['count'] / merged['n_docs']
    return merged.pivot(
        index='party', columns='frame', values='freq').fillna(0)


def compute_fficf(wide: pd.DataFrame) -> pd.DataFrame:
    """FF-ICF after Vossen et al. (2020)."""
    corpus_mean = wide.mean()
    n           = len(wide)
    result = pd.DataFrame(
        index=wide.index, columns=wide.columns, dtype=float)
    for frame in wide.columns:
        ff      = wide[frame]
        n_above = (ff > corpus_mean[frame]).sum()
        icf     = np.log((n + 1) / (n_above + 1))
        result[frame] = (ff * icf).round(4)
    return result


# ── Figure 1: Combined FF-ICF party fingerprints ──────────────────────────────

def plot_combined_fficf(mot_df, man_df, sp_df, top_k: int = 15):
    fficf_list = []
    for df in [mot_df, man_df, sp_df]:
        wide = freq_per_doc(df)
        ff   = compute_fficf(wide)
        fficf_list.append(ff)

    all_cols = sorted(set().union(*[f.columns for f in fficf_list]))
    aligned  = [f.reindex(columns=all_cols, fill_value=0)
                for f in fficf_list]
    combined = pd.concat(aligned).groupby(level=0).mean()
    combined = combined.reindex(
        [p for p in TARGET if p in combined.index])

    top_frames = (combined.max(axis=0)
                  .sort_values(ascending=False)
                  .head(top_k).index.tolist())
    sub = combined[top_frames]
    combined.to_csv(OUT_DIR / 'fficf_combined.csv')

    fig, ax = plt.subplots(
        figsize=(top_k * 0.9 + 2, len(sub) * 0.65 + 2))
    sns.heatmap(sub.astype(float), ax=ax, cmap='YlOrRd',
                annot=True, fmt='.2f', linewidths=0.4,
                cbar_kws={'label': 'FF-ICF score (averaged across text types)'})
    ax.set_title(
        'FF-ICF Frame Distinctiveness per Party\n'
        '(averaged across parliamentary motions, election manifestos, '
        'and parliamentary speeches)',
        fontsize=11)
    ax.set_xlabel('FrameNet frame')
    ax.set_ylabel('Party')
    ax.tick_params(axis='x', rotation=40)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fficf_combined.pdf', bbox_inches='tight')
    plt.close()
    print('Saved: fficf_combined.pdf')


# ── Figure 2: Temporal LOME frame trends by bloc ──────────────────────────────

def plot_lome_temporal(mot_df: pd.DataFrame, min_n: int = 30):
    mot = mot_df[~mot_df['frame'].isin(GENRE_FRAMES)].copy()
    mot['bloc'] = mot['party'].map(BLOC_MAP)
    mot = mot[mot['bloc'].notna()].copy()

    n_docs = (mot.groupby(['bloc', 'year'])['doc_id']
              .nunique().reset_index(name='n_docs'))

    sub    = mot[mot['frame'].isin(TEMPORAL_FRAMES)].copy()
    counts = (sub.groupby(['bloc', 'year', 'frame'])
              .size().reset_index(name='count'))
    merged = counts.merge(n_docs, on=['bloc', 'year'])
    merged['freq'] = merged['count'] / merged['n_docs']
    merged = merged[merged['n_docs'] >= min_n].copy()
    merged.to_csv(OUT_DIR / 'lome_temporal_bloc.csv', index=False)

    print('\n=== Temporal LOME frame means per bloc ===')
    for frame in TEMPORAL_FRAMES:
        fsub = merged[merged['frame'] == frame]
        print(f'\n{frame}:')
        for bloc in ['Left', 'Center', 'Right']:
            bsub = fsub[fsub['bloc'] == bloc]
            if bsub.empty:
                continue
            print(f'  {bloc:<8} mean={bsub["freq"].mean():.3f}  '
                  f'min={bsub["freq"].min():.3f}  '
                  f'max={bsub["freq"].max():.3f}  '
                  f'n_years={len(bsub)}')

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes = axes.flatten()

    for i, frame in enumerate(TEMPORAL_FRAMES):
        ax   = axes[i]
        fsub = merged[merged['frame'] == frame]

        for bloc, color in BLOC_COLORS.items():
            bsub = fsub[fsub['bloc'] == bloc].sort_values('year')
            if bsub.empty:
                continue
            ax.plot(bsub['year'], bsub['freq'],
                    label=bloc, color=color,
                    linewidth=2, marker='o', markersize=4)

        ax.set_title(TEMPORAL_LABELS[frame], fontsize=9.5, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Frame occurrences per document')
        ax.set_xlim(2008, 2025)
        ax.legend(fontsize=8.5)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        'Temporal trends in LOME frame usage by ideological bloc\n'
        f'Parliamentary motions 2008--2025 '
        f'(years with fewer than {min_n} motions per bloc omitted)',
        fontsize=12)
    plt.savefig(FIG_DIR / 'lome_temporal_bloc.pdf', bbox_inches='tight')
    plt.close()
    print('\nSaved: lome_temporal_bloc.pdf')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('Loading valid (non-all-ones) document IDs...')
    valid_ids = load_valid_ids()

    print('\nLoading and filtering frame occurrence CSVs...')
    mot_df, man_df, sp_df = load_and_filter(valid_ids)

    print('\nFigure 1: Combined FF-ICF party fingerprints...')
    plot_combined_fficf(mot_df, man_df, sp_df, top_k=15)

    print('\nFigure 2: Temporal LOME frame trends by bloc...')
    plot_lome_temporal(mot_df, min_n=30)

    print(f'\nAll outputs saved to {OUT_DIR}')


if __name__ == '__main__':
    main()
