"""
Macro-Frame Analysis: Visualizations and Statistical Tests
====================================================================
Input:
    results/motions/macro_scores/qwen2.5-32b.csv   (motions)
    results/manifestos/macro_scores/qwen2.5-32b.csv (manifestos)
    results/speeches/macro_scores/qwen2.5-32b.csv (speeches)
Output:
    results/analysis/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy.stats import kruskal, mannwhitneyu
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

OUT_DIR = Path('results/analysis')
FIG_DIR = Path('results/analysis/figures')
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

SHOCK_EVENTS = {
    2015: 'Paris Agreement',
    2019: 'Urgenda ruling',
    2021: 'IPCC AR6',
    2022: 'Russia-Ukraine',
}

PARTY_COLORS = {
    'GroenLinks':      '#2e8b57',
    'PvdA':            '#e63946',
    'GroenLinks-PvdA': '#4a7c59',
    'D66':             '#00b4d8',
    'CDA':             '#52b788',
    'VVD':             '#f4a261',
    'PVV':             '#023e8a',
    'FvD':             '#6d023e',
    'BBB':             '#95d5b2',
    'PvdD':            '#74c69d',
}

plt.rcParams.update({
    'font.family':    'serif',
    'font.size':      11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi':     150,
})


def filter_false_positives(df: pd.DataFrame, frames: list) -> pd.DataFrame:
    """
    Remove rows where all frame scores are 1 (false positives from classifier).
    These texts contain no identifiable climate framing signal.
    """
    all_ones = (df[frames] == 1).all(axis=1)
    n_removed = all_ones.sum()
    print(f'  Removed {n_removed} false positives (all-1 rows)')
    return df[~all_ones].copy()

PARTY_NORM = {
    'fvd':             'FvD',
    'pvv':             'PVV',
    'vvd':             'VVD',
    'cda':             'CDA',
    'd66':             'D66',
    'pvda':            'PvdA',
    'pvdd':            'PvdD',
    'bbb':             'BBB',
    'groenlinks':      'GroenLinks',
    'groenlinks-pvda': 'GroenLinks-PvdA',
}

def normalize_party(p: str) -> str:
    if not isinstance(p, str):
        return ''
    return PARTY_NORM.get(p.lower().strip(), p.strip())


def load_motions() -> pd.DataFrame:
    """Load and preprocess motion macro-frame scores."""
    df = pd.read_csv('results/motions/macro_scores/qwen2.5-32b.csv')
    df['party'] = (df['fractions'].str.split(';').str[0]
               .str.strip().apply(normalize_party))
    df['year']  = pd.to_datetime(df['date'], utc=True, errors='coerce').dt.year
    df = df[df['party'].isin(TARGET)].copy()
    frame_cols = [f'{f}' for f in FRAMES]
    df = df[df[frame_cols].ne(-1).all(axis=1)].copy()
    df = df.rename(columns={f'{f}': f for f in FRAMES})
    df = filter_false_positives(df, FRAMES)
    df['bloc'] = df['party'].map(BLOC_MAP)
    return df


def load_manifestos() -> pd.DataFrame:
    """Load and preprocess manifesto macro-frame scores."""
    df = pd.read_csv('results/manifestos/macro_scores/qwen2.5-32b.csv')
    df['party'] = df['party'].str.strip().str.upper()
    name_map = {
        'BBB': 'BBB', 'CDA': 'CDA', 'D66': 'D66', 'FVD': 'FvD',
        'GROENLINKS': 'GroenLinks', 'GROENLINKS-PVDA': 'GroenLinks-PvdA',
        'PVDA': 'PvdA', 'PVDD': 'PvdD', 'PVV': 'PVV', 'VVD': 'VVD'
    }
    df['party'] = df['party'].map(name_map)
    df = df[df['party'].isin(TARGET)].copy()
    df = df[df[FRAMES].ne(-1).all(axis=1)].copy()
    df = filter_false_positives(df, FRAMES)
    df['bloc'] = df['party'].map(BLOC_MAP)
    return df


def load_speeches() -> pd.DataFrame:
    """Load and preprocess speech macro-frame scores."""
    df = pd.read_csv('results/speeches/macro_scores/qwen2.5-32b.csv')
    df = df.rename(columns={'party_name': 'party'})
    df = df[df['party'].isin(TARGET)].copy()
    df = df[df[FRAMES].ne(-1).all(axis=1)].copy()
    df = filter_false_positives(df, FRAMES)
    df['bloc'] = df['party'].map(BLOC_MAP)
    return df


def effect_size_r(U: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation r as effect size for Mann-Whitney U."""
    return 1 - (2 * U) / (n1 * n2)


def interpret_r(r: float) -> str:
    """Interpret effect size magnitude."""
    r = abs(r)
    if r < 0.1:   return 'negligible'
    elif r < 0.3: return 'small'
    elif r < 0.5: return 'medium'
    else:         return 'large'


def sig_label(p: float) -> str:
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    return ''


def party_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean frame score per party."""
    return df.groupby('party')[FRAMES].mean().round(3)


def dominant_frame_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Compute dominant frame counts per party."""
    df = df.copy()
    df['dominant'] = df[FRAMES].idxmax(axis=1)
    return df.groupby(['party', 'dominant']).size().unstack(fill_value=0)


def temporal_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Compute yearly mean frame scores."""
    return df.groupby('year')[FRAMES].mean().round(3)


def bloc_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean frame scores per ideological bloc."""
    return df.groupby('bloc')[FRAMES].mean().round(3)


def compute_fficf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute FF-ICF (Frame Frequency - Inverse Corpus Frequency) per party per frame.
    FF  = mean frame score for party p
    ICF = log((N+1) / (n_p+1)) where n_p = number of parties with FF above corpus mean
    """
    party_means  = df.groupby('party')[FRAMES].mean()
    corpus_means = df[FRAMES].mean()
    n_parties    = len(party_means)
    result = pd.DataFrame(index=party_means.index, columns=FRAMES, dtype=float)
    for f in FRAMES:
        n_above = (party_means[f] > corpus_means[f]).sum()
        icf     = np.log((n_parties + 1) / (n_above + 1))
        result[f] = (party_means[f] * icf).round(3)
    return result


def test_shock_events(motions: pd.DataFrame) -> pd.DataFrame:
    """
    Mann-Whitney U test comparing pre vs post frame distributions for each shock event.
    """
    rows = []
    for year, label in SHOCK_EVENTS.items():
        pre  = motions[motions['year'] < year]
        post = motions[motions['year'] >= year]
        for f in FRAMES:
            stat, p = mannwhitneyu(pre[f].values, post[f].values, alternative='two-sided')
            r = effect_size_r(stat, len(pre), len(post))
            rows.append({
                'event':    label, 'year': year, 'frame': f,
                'pre_mean': round(pre[f].mean(), 3),
                'post_mean':round(post[f].mean(), 3),
                'diff':     round(post[f].mean() - pre[f].mean(), 3),
                'U_stat':   round(stat, 1), 'p_value': round(p, 4),
                'sig':      sig_label(p), 'r': round(r, 3),
                'effect':   interpret_r(r),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'shock_event_tests.csv', index=False)
    return df


def test_bloc_differences(motions: pd.DataFrame, manifestos: pd.DataFrame,
                          speeches: pd.DataFrame) -> pd.DataFrame:
    """
    Kruskal-Wallis + pairwise Mann-Whitney U for bloc differences per frame.
    """
    rows = []
    for df, source in [(motions, 'motions'), (manifestos, 'manifestos'), (speeches, 'speeches')]:
        left   = df[df['bloc'] == 'Left']
        center = df[df['bloc'] == 'Center']
        right  = df[df['bloc'] == 'Right']
        for f in FRAMES:
            h, p_kw = kruskal(left[f], center[f], right[f])
            u_lc, p_lc = mannwhitneyu(left[f], center[f], alternative='two-sided')
            u_lr, p_lr = mannwhitneyu(left[f], right[f],  alternative='two-sided')
            u_cr, p_cr = mannwhitneyu(center[f], right[f], alternative='two-sided')
            rows.append({
                'source': source, 'frame': f,
                'left_mean':   round(left[f].mean(), 3),
                'center_mean': round(center[f].mean(), 3),
                'right_mean':  round(right[f].mean(), 3),
                'H_stat': round(h, 2), 'p_kruskal': round(p_kw, 4),
                'sig_kruskal': sig_label(p_kw),
                'p_left_center':   round(p_lc, 4),
                'r_left_center':   round(effect_size_r(u_lc, len(left), len(center)), 3),
                'effect_lc':       interpret_r(effect_size_r(u_lc, len(left), len(center))),
                'p_left_right':    round(p_lr, 4),
                'r_left_right':    round(effect_size_r(u_lr, len(left), len(right)), 3),
                'effect_lr':       interpret_r(effect_size_r(u_lr, len(left), len(right))),
                'p_center_right':  round(p_cr, 4),
                'r_center_right':  round(effect_size_r(u_cr, len(center), len(right)), 3),
                'effect_cr':       interpret_r(effect_size_r(u_cr, len(center), len(right))),
            })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / 'bloc_tests.csv', index=False)
    return df_out


def test_rhetoric_action_gap(motions: pd.DataFrame, manifestos: pd.DataFrame,
                              speeches: pd.DataFrame) -> pd.DataFrame:
    """
    Mann-Whitney U tests for manifesto vs motion and speech vs motion gaps per party.
    """
    rows = []
    common_man = set(manifestos['party'].unique()) & set(motions['party'].unique())
    common_sp  = set(speeches['party'].unique())   & set(motions['party'].unique())

    for party in sorted(common_man):
        man_vals = manifestos[manifestos['party'] == party]
        mot_vals = motions[motions['party'] == party]
        for f in FRAMES:
            stat, p = mannwhitneyu(man_vals[f], mot_vals[f], alternative='two-sided')
            r = effect_size_r(stat, len(man_vals), len(mot_vals))
            rows.append({
                'comparison': 'manifesto_vs_motion', 'party': party, 'frame': f,
                'source_a_mean': round(man_vals[f].mean(), 3),
                'source_b_mean': round(mot_vals[f].mean(), 3),
                'gap':    round(man_vals[f].mean() - mot_vals[f].mean(), 3),
                'U_stat': round(stat, 1), 'p_value': round(p, 4),
                'sig':    sig_label(p), 'r': round(r, 3),
                'effect': interpret_r(r),
            })

    for party in sorted(common_sp):
        sp_vals  = speeches[speeches['party'] == party]
        mot_vals = motions[motions['party'] == party]
        for f in FRAMES:
            stat, p = mannwhitneyu(sp_vals[f], mot_vals[f], alternative='two-sided')
            r = effect_size_r(stat, len(sp_vals), len(mot_vals))
            rows.append({
                'comparison': 'speech_vs_motion', 'party': party, 'frame': f,
                'source_a_mean': round(sp_vals[f].mean(), 3),
                'source_b_mean': round(mot_vals[f].mean(), 3),
                'gap':    round(sp_vals[f].mean() - mot_vals[f].mean(), 3),
                'U_stat': round(stat, 1), 'p_value': round(p, 4),
                'sig':    sig_label(p), 'r': round(r, 3),
                'effect': interpret_r(r),
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / 'rhetoric_action_gap_tests.csv', index=False)
    return df_out


def plot_party_profile_heatmap(motions: pd.DataFrame, manifestos: pd.DataFrame,
                                speeches: pd.DataFrame):
    """Heatmap of z-score normalized mean frame scores per party for all three sources."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    for ax, df, title in zip(
        axes,
        [motions, manifestos, speeches],
        ['Parliamentary Motions', 'Election Manifestos', 'Parliamentary Speeches']
    ):
        means = df.groupby('party')[FRAMES].mean()
        z = (means - means.mean()) / means.std()
        z = z.rename(columns=FRAME_LABELS)
        sns.heatmap(
            z, ax=ax, cmap='RdBu_r', center=0,
            annot=means.rename(columns=FRAME_LABELS).round(2),
            fmt='.2f', linewidths=0.5,
            cbar_kws={'label': 'Z-score'},
        )
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=30)

    plt.suptitle('Party Frame Profiles: Mean Likert Scores (Z-normalized per frame)', y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'party_profiles_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print('Saved: party_profiles_heatmap.pdf')


def plot_temporal_trends(motions: pd.DataFrame):
    """Line plot of yearly mean frame scores with shock event markers."""
    yearly = motions.groupby('year')[FRAMES].mean()
    fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
    axes = axes.flatten()

    for i, f in enumerate(FRAMES):
        ax = axes[i]
        ax.plot(yearly.index, yearly[f], color='#2b4590', linewidth=2)
        ax.fill_between(yearly.index, yearly[f], alpha=0.1, color='#2b4590')
        for year, label in SHOCK_EVENTS.items():
            ax.axvline(x=year, color='#e63946', linestyle='--', alpha=0.7, linewidth=1)
            ax.text(year + 0.1, ax.get_ylim()[1] * 0.98, label,
                    rotation=90, va='top', fontsize=8, color='#e63946')
        ax.set_title(FRAME_LABELS[f])
        ax.set_ylabel('Mean Likert score')
        ax.grid(True, alpha=0.3)

    axes[-1].set_visible(False)
    fig.suptitle('Temporal Trends in Climate Framing (Motions, 2008-2025)', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'temporal_trends.pdf', bbox_inches='tight')
    plt.close()
    print('Saved: temporal_trends.pdf')


def plot_shock_events(shock_df: pd.DataFrame):
    """Bar chart of pre/post mean frame scores with significance annotations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (year, label) in enumerate(SHOCK_EVENTS.items()):
        ax    = axes[i]
        edata = shock_df[shock_df['event'] == label]
        x     = np.arange(len(FRAMES))
        w     = 0.35

        ax.bar(x - w/2, edata['pre_mean'],  w, label='Pre',  color='#4a7c59', alpha=0.8)
        ax.bar(x + w/2, edata['post_mean'], w, label='Post', color='#e63946', alpha=0.8)

        for j, row in edata.reset_index().iterrows():
            if row['sig']:
                y = max(row['pre_mean'], row['post_mean']) + 0.02
                ax.text(j, y, row['sig'], ha='center', fontsize=12)

        ax.set_xticks(x)
        ax.set_xticklabels([FRAME_LABELS[f] for f in FRAMES], rotation=30, ha='right')
        ax.set_title(f'{label} ({year})')
        ax.set_ylabel('Mean Likert score')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(1.0, 2.3)

    plt.suptitle('Frame Score Changes Around Shock Events\n(* p<0.05, ** p<0.01, *** p<0.001)',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'shock_events.pdf', bbox_inches='tight')
    plt.close()
    print('Saved: shock_events.pdf')


def plot_bloc_comparison(motions: pd.DataFrame, manifestos: pd.DataFrame,
                          speeches: pd.DataFrame, bloc_df: pd.DataFrame):
    """Grouped bar chart of mean frame scores per bloc for all three sources."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = {'Left': '#2e8b57', 'Center': '#f4a261', 'Right': '#e63946'}
    x = np.arange(len(FRAMES))
    w = 0.25

    for ax, df, source, title in zip(
        axes,
        [motions, manifestos, speeches],
        ['motions', 'manifestos', 'speeches'],
        ['Parliamentary Motions', 'Election Manifestos', 'Parliamentary Speeches']
    ):
        bloc_means = df.groupby('bloc')[FRAMES].mean()
        edata = bloc_df[bloc_df['source'] == source]

        for j, bloc in enumerate(['Left', 'Center', 'Right']):
            if bloc not in bloc_means.index:
                continue
            vals = [bloc_means.loc[bloc, f] for f in FRAMES]
            ax.bar(x + (j - 1) * w, vals, w, label=bloc, color=colors[bloc], alpha=0.85)

        for k, f in enumerate(FRAMES):
            row = edata[edata['frame'] == f]
            if not row.empty and row['sig_kruskal'].values[0]:
                ymax = max(
                    bloc_means.loc[b, f] for b in ['Left','Center','Right']
                    if b in bloc_means.index
                ) + 0.05
                ax.text(k, ymax, row['sig_kruskal'].values[0], ha='center', fontsize=12)

        ax.set_xticks(x)
        ax.set_xticklabels([FRAME_LABELS[f] for f in FRAMES], rotation=30, ha='right')
        ax.set_title(title)
        ax.set_ylabel('Mean Likert score')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(1.0, 2.5)

    plt.suptitle('Frame Scores by Ideological Bloc\n(* p<0.05, ** p<0.01, *** p<0.001)',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'bloc_comparison.pdf', bbox_inches='tight')
    plt.close()
    print('Saved: bloc_comparison.pdf')


def plot_fficf_heatmap(motions: pd.DataFrame, manifestos: pd.DataFrame,
                        speeches: pd.DataFrame):
    """Heatmap of FF-ICF scores per party for all three sources."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    for ax, df, title in zip(
        axes,
        [motions, manifestos, speeches],
        ['Parliamentary Motions', 'Election Manifestos', 'Parliamentary Speeches']
    ):
        ff = compute_fficf(df).rename(columns=FRAME_LABELS)
        sns.heatmap(
            ff.astype(float), ax=ax, cmap='YlOrRd',
            annot=True, fmt='.2f', linewidths=0.5,
            cbar_kws={'label': 'FF-ICF score'},
        )
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=30)

    plt.suptitle('FF-ICF: Frame Distinctiveness per Party', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fficf_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print('Saved: fficf_heatmap.pdf')


def plot_rhetoric_action_gap(gap_df: pd.DataFrame):
    for comparison, title, filename, source_label in [
        ('manifesto_vs_motion', 'Rhetoric-Action Gap: Manifesto vs Motion',
         'rhetoric_action_gap_manifesto.pdf', 'Manifesto'),
        ('speech_vs_motion', 'Rhetoric-Action Gap: Speech vs Motion',
         'rhetoric_action_gap_speech.pdf', 'Speech'),
    ]:
        sub = gap_df[gap_df['comparison'] == comparison]
        if sub.empty:
            continue

        parties = sorted(sub['party'].unique())
        fig, axes = plt.subplots(4, 2, figsize=(14, 18))
        axes = axes.flatten()

        for i, f in enumerate(FRAMES):
            ax    = axes[i]
            fdata = sub[sub['frame'] == f].set_index('party').reindex(parties)
            colors = ['#2e8b57' if v > 0 else '#e63946' for v in fdata['gap']]
            ax.barh(parties, fdata['gap'], color=colors, alpha=0.85)

            for j, (party, row) in enumerate(fdata.iterrows()):
                if pd.notna(row['sig']) and row['sig']:
                    ax.text(row['gap'] + 0.01 * np.sign(row['gap']),
                            j, row['sig'], va='center', fontsize=11)

            ax.axvline(0, color='black', linewidth=0.8)
            ax.set_title(FRAME_LABELS[f])
            ax.set_xlabel(f'{source_label} mean − Motion mean')
            ax.grid(True, alpha=0.3, axis='x')

        axes[-1].set_visible(False)
        green_patch = mpatches.Patch(color='#2e8b57', alpha=0.85,
                                     label=f'Higher in {source_label.lower()}')
        red_patch   = mpatches.Patch(color='#e63946', alpha=0.85,
                                     label='Higher in motions')
        fig.legend(handles=[green_patch, red_patch], loc='lower right', fontsize=11)
        plt.suptitle(f'{title}\n(* p<0.05, ** p<0.01, *** p<0.001)', fontsize=13)
        plt.tight_layout()
        plt.savefig(FIG_DIR / filename, bbox_inches='tight')
        plt.close()
        print(f'Saved: {filename}')


def plot_party_temporal(motions: pd.DataFrame, parties: list):
    """Line plot of dominant frame proportion over time for selected parties."""
    df = motions.copy()
    df['dominant'] = df[FRAMES].idxmax(axis=1)

    fig, axes = plt.subplots(len(parties), 1, figsize=(12, 4 * len(parties)), sharex=True)

    for ax, party in zip(axes, parties):
        pdata = df[df['party'] == party]
        yearly_dom = pdata.groupby('year')['dominant'].value_counts(
            normalize=True).unstack(fill_value=0)

        for f in FRAMES:
            if f in yearly_dom.columns:
                ax.plot(yearly_dom.index, yearly_dom[f],
                        label=FRAME_LABELS[f], linewidth=1.8)

        for year, label in SHOCK_EVENTS.items():
            ax.axvline(x=year, color='grey', linestyle='--', alpha=0.5, linewidth=1)

        ax.set_title(party)
        ax.set_ylabel('Proportion dominant')
        ax.legend(loc='upper left', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.xlabel('Year')
    plt.suptitle('Dominant Frame Over Time: Selected Parties', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'party_temporal_dominant.pdf', bbox_inches='tight')
    plt.close()
    print('Saved: party_temporal_dominant.pdf')

def test_rhetoric_action_correlation(motions: pd.DataFrame,
                                      manifestos: pd.DataFrame,
                                      speeches: pd.DataFrame) -> pd.DataFrame:
    """
    Spearman correlation between mean manifesto/speech score and mean motion
    score per party, per frame. Tests whether parties that promise more in
    manifestos also deliver more in motions.
    """
    mot_means  = motions.groupby('party')[FRAMES].mean()
    man_means  = manifestos.groupby('party')[FRAMES].mean()
    sp_means   = speeches.groupby('party')[FRAMES].mean()

    rows = []
    for comparison, source_means in [('manifesto_vs_motion', man_means),
                                      ('speech_vs_motion',   sp_means)]:
        common = source_means.index.intersection(mot_means.index)
        for f in FRAMES:
            x = source_means.loc[common, f].values
            y = mot_means.loc[common, f].values
            rho, p = spearmanr(x, y)
            rows.append({
                'comparison': comparison,
                'frame':      f,
                'n_parties':  len(common),
                'rho':        round(rho, 3),
                'p_value':    round(p, 4),
                'sig':        sig_label(p),
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / 'rhetoric_action_correlations.csv', index=False)
    print('\n=== Rhetoric-Action Correlations (Spearman rho) ===')
    print(df_out.to_string(index=False))
    return df_out


def plot_rhetoric_action_scatter(motions: pd.DataFrame,
                                  manifestos: pd.DataFrame,
                                  speeches: pd.DataFrame):
    """
    Scatter plots: manifesto mean (x) vs motion mean (y) per party per frame.
    Points above diagonal = party does MORE in motions than manifestos promise.
    Points below diagonal = party promises MORE than it delivers in motions.
    """
    mot_means = motions.groupby('party')[FRAMES].mean()
    man_means = manifestos.groupby('party')[FRAMES].mean()
    common    = man_means.index.intersection(mot_means.index)

    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    axes = axes.flatten()

    for i, f in enumerate(FRAMES):
        ax = axes[i]
        x  = man_means.loc[common, f]
        y  = mot_means.loc[common, f]

        for party in common:
            color = PARTY_COLORS.get(party, '#555555')
            ax.scatter(x[party], y[party], color=color, s=80, zorder=3)
            ax.annotate(party, (x[party], y[party]),
                        fontsize=7.5, xytext=(4, 2),
                        textcoords='offset points')

        # Diagonal = perfect correspondence
        lim_min = min(x.min(), y.min()) - 0.05
        lim_max = max(x.max(), y.max()) + 0.05
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                color='grey', linestyle='--', linewidth=1, alpha=0.6)

        rho, p = spearmanr(x, y)
        ax.set_title(f'{FRAME_LABELS[f]}  '
                     f'($\\rho={rho:.2f}$, $p={p:.3f}${sig_label(p)})')
        ax.set_xlabel('Manifesto mean')
        ax.set_ylabel('Motion mean')
        ax.grid(True, alpha=0.3)

    axes[-1].set_visible(False)
    plt.suptitle(
        'Manifesto vs Motion Frame Scores per Party\n'
        'Points below diagonal: party promises more than it delivers\n'
        'Points above diagonal: party does more in motions than manifestos suggest',
        fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'rhetoric_action_scatter.pdf', bbox_inches='tight')
    plt.close()
    print('Saved: rhetoric_action_scatter.pdf')



def main():
    print('Loading data...')
    motions    = load_motions()
    manifestos = load_manifestos()
    speeches   = load_speeches()
    print(f'Motions:    {len(motions)} rows')
    print(f'Manifestos: {len(manifestos)} rows')
    print(f'Speeches:   {len(speeches)} rows')

    print('\n=== Party profiles (motions) ===')
    print(party_profiles(motions).to_string())
    print('\n=== Party profiles (manifestos) ===')
    print(party_profiles(manifestos).to_string())
    print('\n=== Party profiles (speeches) ===')
    print(party_profiles(speeches).to_string())

    print('\n=== Bloc comparison (motions) ===')
    print(bloc_comparison(motions).to_string())
    print('\n=== Bloc comparison (manifestos) ===')
    print(bloc_comparison(manifestos).to_string())
    print('\n=== Bloc comparison (speeches) ===')
    print(bloc_comparison(speeches).to_string())

    print('\nGenerating party profile heatmap...')
    plot_party_profile_heatmap(motions, manifestos, speeches)

    print('\nGenerating temporal trends plot...')
    plot_temporal_trends(motions)

    print('\nRunning shock event tests...')
    shock_df = test_shock_events(motions)
    plot_shock_events(shock_df)
    print(shock_df.to_string(index=False))

    print('\nRunning bloc difference tests...')
    bloc_df = test_bloc_differences(motions, manifestos, speeches)
    plot_bloc_comparison(motions, manifestos, speeches, bloc_df)
    print(bloc_df.to_string(index=False))

    print('\nGenerating FF-ICF heatmap...')
    plot_fficf_heatmap(motions, manifestos, speeches)

    print('\nRunning rhetoric-action gap tests...')
    gap_df = test_rhetoric_action_gap(motions, manifestos, speeches)
    plot_rhetoric_action_gap(gap_df)
    print(gap_df.to_string(index=False))

    print('\nGenerating party temporal plots...')
    plot_party_temporal(motions, ['PVV', 'PvdD', 'GroenLinks', 'VVD'])


    print('\nRunning rhetoric-action correlations...')
    corr_df = test_rhetoric_action_correlation(motions, manifestos, speeches)
    plot_rhetoric_action_scatter(motions, manifestos, speeches)

    # Save summary CSVs
    party_profiles(motions).to_csv(OUT_DIR / 'party_profiles_motions.csv')
    party_profiles(manifestos).to_csv(OUT_DIR / 'party_profiles_manifestos.csv')
    party_profiles(speeches).to_csv(OUT_DIR / 'party_profiles_speeches.csv')
    temporal_trends(motions).to_csv(OUT_DIR / 'temporal_trends_motions.csv')
    compute_fficf(motions).to_csv(OUT_DIR / 'fficf_motions.csv')
    compute_fficf(manifestos).to_csv(OUT_DIR / 'fficf_manifestos.csv')
    compute_fficf(speeches).to_csv(OUT_DIR / 'fficf_speeches.csv')

    print(f'\nAll outputs saved to {OUT_DIR}')


if __name__ == '__main__':
    main()