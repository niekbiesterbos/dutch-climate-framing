"""
Ideological Polarisation in Climate Framing (2008-2025)
=================================================================
Two complementary analyses:

1. Polarisation index over time: yearly left-right bloc distance per frame,
   computed as absolute difference in mean frame scores between left and
   right blocs. Reveals whether ideological divergence in climate framing
   widens or narrows over the study period.

2. Shock event analysis per bloc: pre/post Mann-Whitney U tests per bloc
   rather than aggregate corpus, revealing whether landmark events
   differentially shift framing across ideological positions.

Input:
    results/motions/macro_scores/qwen2.5-32b.csv

Output:
    results/analysis/
        polarisation_index.csv
        shock_events_by_bloc.csv
        figures/
            polarisation_over_time.pdf
            shock_events_by_bloc.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

OUT_DIR = Path('results/analysis')
FIG_DIR = OUT_DIR / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

FRAMES = ['economic', 'moral', 'scientific', 'security',
          'health_environment', 'crisis_urgency', 'weaponization']

FRAME_LABELS = {
    'economic':           'Economic',
    'moral':              'Moral',
    'scientific':         'Scientific',
    'security':           'Security',
    'health_environment': 'Health/Env.',
    'crisis_urgency':     'Crisis/Urgency',
    'weaponization':      'Weaponization',
}

TARGET = ['GroenLinks', 'PvdA', 'GroenLinks-PvdA', 'D66', 'CDA',
          'VVD', 'PVV', 'FvD', 'BBB', 'PvdD']

LEFT   = ['GroenLinks', 'PvdA', 'GroenLinks-PvdA', 'PvdD']
CENTER = ['D66', 'CDA']
RIGHT  = ['VVD', 'PVV', 'FvD', 'BBB']

BLOC_MAP = ({p: 'Left'   for p in LEFT}  |
            {p: 'Center' for p in CENTER} |
            {p: 'Right'  for p in RIGHT})

BLOC_COLORS = {
    'Left':   '#2e8b57',
    'Center': '#f4a261',
    'Right':  '#e63946',
}

SHOCK_EVENTS = {
    2015: 'Paris Agreement',
    2019: 'Urgenda ruling',
    2021: 'IPCC AR6',
    2022: 'Russia-Ukraine',
}

# Frame colors for polarisation plot
FRAME_COLORS = [
    '#e63946', '#2e8b57', '#00b4d8', '#f4a261',
    '#6d023e', '#023e8a', '#95d5b2',
]

plt.rcParams.update({
    'font.family':    'serif',
    'font.size':      11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi':     150,
})


def sig_label(p: float) -> str:
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    return ''


def effect_size_r(U: float, n1: int, n2: int) -> float:
    return 1 - (2 * U) / (n1 * n2)


def interpret_r(r: float) -> str:
    r = abs(r)
    if r < 0.1:   return 'negligible'
    elif r < 0.3: return 'small'
    elif r < 0.5: return 'medium'
    else:         return 'large'


def load_motions() -> pd.DataFrame:
    df = pd.read_csv(
        'results/motions/macro_scores/qwen2.5-32b.csv')
    df['party'] = df['fractions'].str.split(';').str[0].str.strip()
    df['year']  = pd.to_datetime(
        df['date'], utc=True, errors='coerce').dt.year
    df = df[df['party'].isin(TARGET)].copy()
    frame_cols = [f'{f}' for f in FRAMES]
    df = df[df[frame_cols].ne(-1).all(axis=1)].copy()
    df = df.rename(columns={f'{f}': f for f in FRAMES})
    all_ones = (df[FRAMES] == 1).all(axis=1)
    print(f'Removed {all_ones.sum()} false positives')
    df = df[~all_ones].copy()
    df['bloc'] = df['party'].map(BLOC_MAP)
    return df


# ── Analysis 1: Polarisation index over time ──────────────────────────────────

def compute_polarisation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Yearly left-right absolute difference per frame.
    Smoothed with 3-year rolling average to reduce noise.
    """
    rows = []
    for year in sorted(df['year'].dropna().unique()):
        left  = df[(df['year'] == year) & (df['bloc'] == 'Left')]
        right = df[(df['year'] == year) & (df['bloc'] == 'Right')]
        if len(left) < 5 or len(right) < 5:
            continue
        for f in FRAMES:
            rows.append({
                'year':          year,
                'frame':         f,
                'left_mean':     left[f].mean(),
                'right_mean':    right[f].mean(),
                'polarisation':  abs(left[f].mean() - right[f].mean()),
                'direction':     left[f].mean() - right[f].mean(),
            })
    return pd.DataFrame(rows)


def plot_polarisation(pol: pd.DataFrame):
    """
    Two-panel plot of signed left-right divergence per frame over time.
    Panel 1: all frames (raw yearly values).
    Panel 2: top-3 most polarised frames with shock event regions shaded.
    """
    dir_pivot = pol.pivot(
        index='year', columns='frame', values='direction')

    pol_pivot = pol.pivot(
        index='year', columns='frame', values='polarisation')
    mean_pol  = pol_pivot.mean().sort_values(ascending=False)
    top3      = mean_pol.index[:3].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Panel 1: all frames ---
    ax = axes[0]
    ax.axhline(0, color='black', linewidth=0.8, zorder=2)

    for i, f in enumerate(FRAMES):
        if f not in dir_pivot.columns:
            continue
        ax.plot(dir_pivot.index, dir_pivot[f],
                label=FRAME_LABELS[f],
                color=FRAME_COLORS[i],
                linewidth=2 if f in top3 else 1,
                alpha=1.0 if f in top3 else 0.4,
                marker='o', markersize=3,
                zorder=3 if f in top3 else 2)

   
    ax.set_xlabel('Year')
    ax.set_ylabel('Left bloc mean minus right bloc mean')
    ax.set_title('Signed ideological divergence per frame\n(positive = left scores higher)')
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2008, 2025)

    # --- Panel 2: top-3 with shaded shock regions ---
    ax2 = axes[1]
    ax2.axhline(0, color='black', linewidth=0.8, zorder=2)

    top3_colors = [FRAME_COLORS[FRAMES.index(f)] for f in top3]
    for f, color in zip(top3, top3_colors):
        ax2.plot(dir_pivot.index, dir_pivot[f],
                 label=FRAME_LABELS[f], color=color,
                 linewidth=2.5, marker='o', markersize=4, zorder=3)
        ax2.fill_between(dir_pivot.index, 0, dir_pivot[f],
                         alpha=0.08, color=color)

    ax2.set_xlabel('Year')
    ax2.set_ylabel('Left bloc mean minus right bloc mean')
    ax2.set_title(
        f'Top-3 most polarised frames:\n'
        f'{", ".join(FRAME_LABELS[f] for f in top3)}')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(2008, 2025)

    plt.suptitle(
        'Ideological Polarisation in Climate Framing (2008-2025)\n'
        'Yearly left-right bloc distance per frame (raw values)',
        fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'polarisation_over_time.pdf', bbox_inches='tight')
    plt.close()
    print('Saved: polarisation_over_time.pdf')


def plot_polarisation_continuous(pol: pd.DataFrame):
    dir_pivot = pol.pivot(
        index='year', columns='frame', values='direction')

    pol_pivot = pol.pivot(
        index='year', columns='frame', values='polarisation')
    mean_pol  = pol_pivot.mean().sort_values(ascending=False)
    top3      = mean_pol.index[:3].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.axhline(0, color='black', linewidth=0.8, zorder=2)
    for i, f in enumerate(FRAMES):
        if f not in dir_pivot.columns:
            continue
        ax.plot(dir_pivot.index, dir_pivot[f],
                label=FRAME_LABELS[f],
                color=FRAME_COLORS[i],
                linewidth=2 if f in top3 else 1,
                alpha=1.0 if f in top3 else 0.4,
                marker='o', markersize=3,
                zorder=3 if f in top3 else 2)
    ax.set_xlabel('Year')
    ax.set_ylabel('Left bloc mean minus right bloc mean')
    ax.set_title('Signed ideological divergence per frame\n(continuous parties only)')
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2008, 2025)

    ax2 = axes[1]
    ax2.axhline(0, color='black', linewidth=0.8, zorder=2)
    top3_colors = [FRAME_COLORS[FRAMES.index(f)] for f in top3]
    for f, color in zip(top3, top3_colors):
        ax2.plot(dir_pivot.index, dir_pivot[f],
                 label=FRAME_LABELS[f], color=color,
                 linewidth=2.5, marker='o', markersize=4, zorder=3)
        ax2.fill_between(dir_pivot.index, 0, dir_pivot[f],
                         alpha=0.08, color=color)
    
    region_colors = ['#d0e8ff', '#ffe8d0', '#e8ffd0', '#ffd0e8']
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Left bloc mean minus right bloc mean')
    ax2.set_title(
        f'Top-3 most polarised frames:\n'
        f'{", ".join(FRAME_LABELS[f] for f in top3)}')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(2008, 2025)

    plt.suptitle(
        'Ideological Polarisation — Continuous Parties Only (2008-2025)\n'
        'Right: PVV, VVD | Centre: D66, CDA | Left: PvdD, PvdA, GroenLinks',
        fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'polarisation_over_time_continuous.pdf',
                bbox_inches='tight')
    plt.close()
    print('Saved: polarisation_over_time_continuous.pdf')


# ── Analysis 2: Shock events per bloc ─────────────────────────────────────────

def compute_shock_by_bloc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre/post Mann-Whitney U per bloc per frame per shock event.
    """
    rows = []
    for year, label in SHOCK_EVENTS.items():
        for bloc in ['Left', 'Center', 'Right']:
            bloc_df = df[df['bloc'] == bloc]
            pre  = bloc_df[bloc_df['year'] < year]
            post = bloc_df[bloc_df['year'] >= year]
            if len(pre) < 10 or len(post) < 10:
                continue
            for f in FRAMES:
                stat, p = mannwhitneyu(
                    pre[f].values, post[f].values,
                    alternative='two-sided')
                r = effect_size_r(stat, len(pre), len(post))
                rows.append({
                    'event':      label,
                    'year':       year,
                    'bloc':       bloc,
                    'frame':      f,
                    'pre_mean':   round(pre[f].mean(), 3),
                    'post_mean':  round(post[f].mean(), 3),
                    'diff':       round(post[f].mean() - pre[f].mean(), 3),
                    'r':          round(r, 3),
                    'effect':     interpret_r(r),
                    'p_value':    round(p, 4),
                    'sig':        sig_label(p),
                })
    return pd.DataFrame(rows)


def plot_shock_by_bloc(shock: pd.DataFrame):
    """
    For each frame: line plot of mean score per bloc over pre/post periods,
    one panel per shock event. Shows convergence/divergence directly.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, (year, label) in enumerate(SHOCK_EVENTS.items()):
        ax    = axes[i]
        edata = shock[shock['event'] == label]
        x     = np.arange(len(FRAMES))
        w     = 0.25

        for j, bloc in enumerate(['Left', 'Center', 'Right']):
            bdata = edata[edata['bloc'] == bloc]
            if bdata.empty:
                continue
            bdata = bdata.set_index('frame').reindex(FRAMES)

            # Pre bars (hatched) and post bars (solid) per bloc
            ax.bar(x + (j - 1) * w, bdata['pre_mean'], w,
                   color=BLOC_COLORS[bloc], alpha=0.35,
                   hatch='//', edgecolor=BLOC_COLORS[bloc])
            ax.bar(x + (j - 1) * w, bdata['diff'], w,
                   bottom=bdata['pre_mean'],
                   color=BLOC_COLORS[bloc], alpha=0.85,
                   label=bloc if i == 0 else '')

            for k, f in enumerate(FRAMES):
                if f not in bdata.index:
                    continue
                row = bdata.loc[f]
                if pd.notna(row['sig']) and row['sig']:
                    ypos = row['post_mean'] + 0.02
                    ax.text(k + (j - 1) * w, ypos,
                            row['sig'], ha='center', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [FRAME_LABELS[f] for f in FRAMES],
            rotation=30, ha='right')
        ax.set_title(f'{label} ({year})')
        ax.set_ylabel('Mean Likert score')
        ax.set_ylim(1.0, 2.6)
        ax.grid(True, alpha=0.3, axis='y')

        # Legend: hatched = pre, solid = post
        pre_patch  = mpatches.Patch(facecolor='grey', alpha=0.35,
                                     hatch='//', label='Pre-event')
        post_patch = mpatches.Patch(facecolor='grey', alpha=0.85,
                                     label='Post-event (increment)')
        left_patch   = mpatches.Patch(color=BLOC_COLORS['Left'],   label='Left')
        center_patch = mpatches.Patch(color=BLOC_COLORS['Center'], label='Center')
        right_patch  = mpatches.Patch(color=BLOC_COLORS['Right'],  label='Right')
        ax.legend(handles=[pre_patch, post_patch, left_patch,
                            center_patch, right_patch],
                  fontsize=8, loc='upper right', ncol=2)

    plt.suptitle(
        'Frame Scores Before and After Shock Events by Ideological Bloc\n'
        'Hatched bars show pre-event mean; solid increment shows post-event change\n'
        '(* p<0.05, ** p<0.01, *** p<0.001)',
        fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'shock_events_by_bloc.pdf', bbox_inches='tight')
    plt.close()
    print('Saved: shock_events_by_bloc.pdf')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('Loading motions...')
    df = load_motions()
    print(f'  {len(df)} rows after filtering')

    print('\nComputing polarisation index...')
    pol = compute_polarisation(df)
    pol.to_csv(OUT_DIR / 'polarisation_index.csv', index=False)
    print(pol.pivot(
        index='year', columns='frame',
        values='polarisation').round(3).to_string())
    plot_polarisation(pol)


    print('\nComputing continuous-parties polarisation...')
    CONTINUOUS_LEFT  = ['GroenLinks', 'PvdA', 'PvdD']
    CONTINUOUS_RIGHT = ['VVD', 'PVV']
    CONTINUOUS_CENTER = ['D66', 'CDA']
    CONTINUOUS = CONTINUOUS_LEFT + CONTINUOUS_RIGHT + CONTINUOUS_CENTER

    BLOC_MAP_CONTINUOUS = (
        {p: 'Left'   for p in CONTINUOUS_LEFT}  |
        {p: 'Center' for p in CONTINUOUS_CENTER} |
        {p: 'Right'  for p in CONTINUOUS_RIGHT}
    )

    df_cont = df[df['party'].isin(CONTINUOUS)].copy()
    df_cont['bloc'] = df_cont['party'].map(BLOC_MAP_CONTINUOUS)
    pol_cont = compute_polarisation(df_cont)
    pol_cont.to_csv(OUT_DIR / 'polarisation_index_continuous.csv', index=False)
    plot_polarisation_continuous(pol_cont)

    print('\nComputing shock events by bloc...')
    shock = compute_shock_by_bloc(df)
    shock.to_csv(OUT_DIR / 'shock_events_by_bloc.csv', index=False)

    print('\n=== Significant bloc-specific shock effects ===')
    sig = shock[shock['sig'] != ''].sort_values(
        ['event', 'frame', 'bloc'])
    print(sig[['event', 'bloc', 'frame', 'pre_mean',
               'post_mean', 'diff', 'r', 'effect', 'sig']].to_string(
        index=False))

    plot_shock_by_bloc(shock)

    print(f'\nAll outputs saved to {OUT_DIR}')


if __name__ == '__main__':
    main()