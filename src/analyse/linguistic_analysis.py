"""
Linguistic Framing Analysis: Voice, Negation, and Modality
====================================================================
Analyses applied directly to raw text across all three text types.

Input:
    results/motions/macro_scores/qwen2.5-32b.csv   (motions)
    results/manifestos/macro_scores/qwen2.5-32b.csv (manifestos)
    results/speeches/macro_scores/qwen2.5-32b.csv (speeches)

Output:
    results/analysis/
"""

import os
import re
import numpy as np
import pandas as pd
import spacy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import kruskal, mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

OUT_DIR = Path('results/analysis')
FIG_DIR = OUT_DIR / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

nlp = spacy.load('nl_core_news_lg')

FRAMES = ['economic', 'moral', 'scientific', 'security',
          'health_environment', 'crisis_urgency', 'weaponization']

TARGET = ['GroenLinks', 'PvdA', 'GroenLinks-PvdA', 'D66', 'CDA',
          'VVD', 'PVV', 'FvD', 'BBB', 'PvdD']

LEFT   = ['GroenLinks', 'PvdA', 'GroenLinks-PvdA', 'PvdD']
CENTER = ['D66', 'CDA']
RIGHT  = ['VVD', 'PVV', 'FvD', 'BBB']

BLOC_MAP = ({p: 'Left'   for p in LEFT}  |
            {p: 'Center' for p in CENTER} |
            {p: 'Right'  for p in RIGHT})

MODALS_OBLIGATION   = {'moet', 'moeten', 'dient', 'dienen', 'heeft te', 'hebben te'}
MODALS_POSSIBILITY  = {'kan', 'kunnen', 'mag', 'mogen'}
MODALS_HEDGING      = {'zou', 'zouden', 'zou kunnen', 'zouden kunnen',
                       'zou mogen', 'lijkt', 'lijken', 'schijnt', 'schijnen'}
MODALS_NEGATION_MOD = {'hoeft niet', 'hoeven niet', 'hoeft geen', 'hoeven geen'}
ALL_MODALS = MODALS_OBLIGATION | MODALS_POSSIBILITY | MODALS_HEDGING | MODALS_NEGATION_MOD

NEGATION_CUES = {'niet', 'geen', 'nooit', 'nergens', 'niemand', 'niets',
                 'noch', 'nauwelijks', 'onvoldoende', 'onmogelijk',
                 'weigert', 'weigeren', 'afwijst', 'afwijzen'}

MOTION_HEADER_RE = re.compile(
    r'(?i)motie\s+van\s+het\s+lid\s+[\w\s\-]+?(c\.s\.)?\s*(over\s+)?',
    re.IGNORECASE
)

plt.rcParams.update({
    'font.family':    'serif',
    'font.size':      11,
    'axes.titlesize': 13,
    'figure.dpi':     150,
})


def strip_motion_header(text: str) -> str:
    """Remove 'Motie van het lid ...' header from motion text."""
    if not isinstance(text, str):
        return ''
    return MOTION_HEADER_RE.sub('', text).strip()


def load_motions() -> pd.DataFrame:
    """Load motions with party, year, motie_id, and normalized text."""
    df = pd.read_csv('results/motions/macro_scores/qwen2.5-32b.csv')
    df['party'] = df['fractions'].str.split(';').str[0].str.strip()
    df['year']  = pd.to_datetime(df['date'], utc=True, errors='coerce').dt.year
    target_lower = {p.lower(): p for p in TARGET}
    df['party'] = df['party'].apply(
        lambda x: target_lower.get(x.lower().strip(), x) if isinstance(x, str) else x
    )
    df = df[df['party'].isin(TARGET)].copy()
    frame_cols = [f'frame_{f}' for f in FRAMES]
    df = df[df[frame_cols].ne(-1).all(axis=1)].copy()
    all_ones = (df[frame_cols] == 1).all(axis=1)
    df = df[~all_ones].copy()
    df['bloc']   = df['party'].map(BLOC_MAP)
    df['source'] = 'motions'
    df['text']   = df['normalized_text'].fillna(df['text']).apply(strip_motion_header)
    df = df.rename(columns={f'frame_{f}': f for f in FRAMES})
    df = df.rename(columns={'id': 'motie_id'})
    return df[['motie_id', 'party', 'year', 'text', 'bloc', 'source'] + FRAMES]


def load_manifestos() -> pd.DataFrame:
    """Load manifestos with party, year, and text."""
    df = pd.read_csv('results/manifestos/macro_scores/qwen2.5-32b.csv')
    target_lower = {p.lower(): p for p in TARGET}
    df['party'] = df['party'].apply(
        lambda x: target_lower.get(x.lower().strip(), x) if isinstance(x, str) else x
    )
    df = df[df['party'].isin(TARGET)].copy()
    df = df[df[FRAMES].ne(-1).all(axis=1)].copy()
    all_ones = (df[FRAMES] == 1).all(axis=1)
    df = df[~all_ones].copy()
    df['bloc'] = df['party'].map(BLOC_MAP)
    df['source'] = 'manifestos'
    df['motie_id'] = None
    return df[['motie_id', 'party', 'year', 'text', 'bloc', 'source'] + FRAMES]


def load_speeches() -> pd.DataFrame:
    """Load speeches with party, year, and text."""
    df = pd.read_csv('results/speeches/macro_scores/qwen2.5-32b.csv')
    df = df.rename(columns={'party_name': 'party', 'doc_id': 'motie_id'})
    target_lower = {p.lower(): p for p in TARGET}
    df['party'] = df['party'].apply(
        lambda x: target_lower.get(x.lower().strip(), x) if isinstance(x, str) else x
    )
    df = df[df['party'].isin(TARGET)].copy()
    df = df[df[FRAMES].ne(-1).all(axis=1)].copy()
    all_ones = (df[FRAMES] == 1).all(axis=1)
    df = df[~all_ones].copy()
    df['bloc'] = df['party'].map(BLOC_MAP)
    df['source'] = 'speeches'
    return df[['motie_id', 'party', 'year', 'text', 'bloc', 'source'] + FRAMES]


def is_passive(token) -> bool:
    """Detect passive construction via spaCy dependency labels."""
    if token.dep_ in ('aux:pass', 'nsubj:pass'):
        return True
    if token.lemma_ in ('worden', 'zijn') and token.dep_ == 'aux':
        head = token.head
        if head.tag_ in ('WW|vd|vrij|zonder', 'WW|vd|prenom|zonder',
                         'WW|vd|prenom|met-e', 'WW|vd|vrij|met'):
            return True
    return False


def analyze_voice(text: str) -> dict:
    """Analyze passive vs active framing per sentence."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {'n_sentences': 0, 'n_passive': 0, 'n_active': 0,
                'prop_passive': None, 'prop_active': None}
    doc = nlp(text[:5000])
    n_passive = n_active = 0
    for sent in doc.sents:
        has_verb  = any(t.pos_ == 'VERB' for t in sent)
        sent_pass = any(is_passive(t) for t in sent)
        if has_verb:
            if sent_pass: n_passive += 1
            else:         n_active  += 1
    total = n_passive + n_active
    return {
        'n_sentences': total,
        'n_passive':   n_passive,
        'n_active':    n_active,
        'prop_passive': n_passive / total if total > 0 else None,
        'prop_active':  n_active  / total if total > 0 else None,
    }


def analyze_negation(text: str) -> dict:
    """Detect negation cues per sentence."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {'n_sentences': 0, 'n_negated': 0, 'prop_negated': None}
    doc = nlp(text[:5000])
    n_negated = n_sentences = 0
    for sent in doc.sents:
        n_sentences += 1
        lemmas = {t.lemma_.lower() for t in sent}
        if lemmas & NEGATION_CUES:
            n_negated += 1
    return {
        'n_sentences': n_sentences,
        'n_negated':   n_negated,
        'prop_negated': n_negated / n_sentences if n_sentences > 0 else None,
    }


def analyze_modality(text: str) -> dict:
    """Detect modal verbs per sentence."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {
            'n_sentences': 0,
            'prop_obligation': None, 'prop_possibility': None,
            'prop_hedging': None,    'prop_negation_mod': None,
            'prop_any_modal': None,
        }
    doc = nlp(text[:5000])
    counts = {'obligation': 0, 'possibility': 0,
              'hedging': 0, 'negation_mod': 0, 'any': 0}
    n_sentences = sum(1 for _ in doc.sents)
    for sent in doc.sents:
        sent_text  = sent.text.lower()
        lemmas     = {t.lemma_.lower() for t in sent}
        sent_flags = {k: False for k in counts}
        for m in MODALS_NEGATION_MOD:
            if m in sent_text: sent_flags['negation_mod'] = True
        for m in MODALS_HEDGING:
            if m in sent_text: sent_flags['hedging'] = True
        for token in sent:
            lemma = token.lemma_.lower()
            if lemma in MODALS_OBLIGATION:    sent_flags['obligation']  = True
            elif lemma in MODALS_POSSIBILITY: sent_flags['possibility'] = True
        for k in ['obligation', 'possibility', 'hedging', 'negation_mod']:
            if sent_flags[k]:
                counts[k] += 1
                counts['any'] += 1
                break
    return {
        'n_sentences':       n_sentences,
        'prop_obligation':   counts['obligation']   / n_sentences if n_sentences > 0 else None,
        'prop_possibility':  counts['possibility']  / n_sentences if n_sentences > 0 else None,
        'prop_hedging':      counts['hedging']      / n_sentences if n_sentences > 0 else None,
        'prop_negation_mod': counts['negation_mod'] / n_sentences if n_sentences > 0 else None,
        'prop_any_modal':    counts['any']          / n_sentences if n_sentences > 0 else None,
    }


def analyze_dataframe(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Apply voice, negation, and modality analysis to every row."""
    print(f'Analyzing {source} ({len(df)} rows)...')
    voice_rows = []
    negation_rows = []
    modality_rows = []
    for i, text in enumerate(df['text']):
        if i % 500 == 0:
            print(f'  {i}/{len(df)}')
        voice_rows.append(analyze_voice(text))
        negation_rows.append(analyze_negation(text))
        modality_rows.append(analyze_modality(text))
    voice_df    = pd.DataFrame(voice_rows,    index=df.index).add_prefix('voice_')
    negation_df = pd.DataFrame(negation_rows, index=df.index).add_prefix('neg_')
    modality_df = pd.DataFrame(modality_rows, index=df.index).add_prefix('mod_')
    return pd.concat([df, voice_df, negation_df, modality_df], axis=1)


def effect_size_r(U: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation r as effect size for Mann-Whitney U."""
    return 1 - (2 * U) / (n1 * n2)


def interpret_r(r: float) -> str:
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


def bloc_tests(df: pd.DataFrame, features: list, source: str) -> pd.DataFrame:
    """Kruskal-Wallis + pairwise Mann-Whitney U for bloc differences."""
    rows = []
    left   = df[df['bloc'] == 'Left']
    center = df[df['bloc'] == 'Center']
    right  = df[df['bloc'] == 'Right']
    for f in features:
        l = left[f].dropna()
        c = center[f].dropna()
        r = right[f].dropna()
        if len(l) < 5 or len(c) < 5 or len(r) < 5:
            continue
        h, p_kw = kruskal(l, c, r)
        u_lr, p_lr = mannwhitneyu(l, r, alternative='two-sided')
        r_lr = effect_size_r(u_lr, len(l), len(r))
        rows.append({
            'source':       source,
            'feature':      f,
            'left_mean':    round(l.mean(), 3),
            'center_mean':  round(c.mean(), 3),
            'right_mean':   round(r.mean(), 3),
            'H_stat':       round(h, 2),
            'p_kruskal':    round(p_kw, 4),
            'sig':          sig_label(p_kw),
            'r_left_right': round(r_lr, 3),
            'effect_lr':    interpret_r(r_lr),
        })
    return pd.DataFrame(rows)


def party_means(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Compute mean feature values per party."""
    return df.groupby('party')[features].mean().round(3)


def plot_bloc_bars(results: dict, feature: str, title: str, filename: str):
    """Bar chart of feature mean per bloc across all three text types."""
    sources = list(results.keys())
    blocs   = ['Left', 'Center', 'Right']
    colors  = {'Left': '#2e8b57', 'Center': '#f4a261', 'Right': '#e63946'}
    x = np.arange(len(sources))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for j, bloc in enumerate(blocs):
        vals = []
        for src, df in results.items():
            sub = df[df['bloc'] == bloc][feature].dropna()
            vals.append(sub.mean() if len(sub) > 0 else 0)
        ax.bar(x + (j - 1) * w, vals, w,
               label=bloc, color=colors[bloc], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(sources)
    ax.set_title(title)
    ax.set_ylabel('Mean proportion')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, bbox_inches='tight')
    plt.close()
    print(f'Saved: {filename}')


def plot_party_heatmap(party_df: pd.DataFrame, features: list,
                       title: str, filename: str):
    """Heatmap of feature means per party."""
    import seaborn as sns
    fig, ax = plt.subplots(
        figsize=(len(features) * 1.6 + 2, len(party_df) * 0.6 + 1.5))
    sns.heatmap(
        party_df[features].astype(float),
        ax=ax, cmap='RdYlGn', annot=True, fmt='.2f',
        linewidths=0.5, cbar_kws={'label': 'Mean proportion'}
    )
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, bbox_inches='tight')
    plt.close()
    print(f'Saved: {filename}')


def main():
    print('Loading data...')
    motions    = load_motions()
    manifestos = load_manifestos()
    speeches   = load_speeches()
    print(f'Motions: {len(motions)} | Manifestos: {len(manifestos)} | Speeches: {len(speeches)}')

    mot_rich = analyze_dataframe(motions,    'motions')
    man_rich = analyze_dataframe(manifestos, 'manifestos')
    sp_rich  = analyze_dataframe(speeches,   'speeches')

    mot_rich.to_csv(OUT_DIR / 'motions_linguistic.csv',    index=False)
    man_rich.to_csv(OUT_DIR / 'manifestos_linguistic.csv', index=False)
    sp_rich.to_csv( OUT_DIR / 'speeches_linguistic.csv',   index=False)
    print('Enriched CSVs saved.')

    voice_feats    = ['voice_prop_passive', 'voice_prop_active']
    negation_feats = ['neg_prop_negated']
    modality_feats = ['mod_prop_obligation', 'mod_prop_possibility',
                      'mod_prop_hedging',    'mod_prop_negation_mod',
                      'mod_prop_any_modal']
    all_feats = voice_feats + negation_feats + modality_feats

    results = {
        'Motions':    mot_rich,
        'Manifestos': man_rich,
        'Speeches':   sp_rich,
    }

    print('\n=== Party means (motions) ===')
    pm_mot = party_means(mot_rich, all_feats)
    print(pm_mot.to_string())
    pm_mot.to_csv(OUT_DIR / 'party_means_motions.csv')

    print('\n=== Party means (manifestos) ===')
    pm_man = party_means(man_rich, all_feats)
    print(pm_man.to_string())
    pm_man.to_csv(OUT_DIR / 'party_means_manifestos.csv')

    print('\n=== Party means (speeches) ===')
    pm_sp = party_means(sp_rich, all_feats)
    print(pm_sp.to_string())
    pm_sp.to_csv(OUT_DIR / 'party_means_speeches.csv')

    print('\n=== Bloc tests ===')
    bloc_rows = []
    for src, df in results.items():
        bt = bloc_tests(df, all_feats, src)
        bloc_rows.append(bt)
        print(f'\n{src}:')
        print(bt.to_string(index=False))
    pd.concat(bloc_rows).to_csv(OUT_DIR / 'bloc_tests.csv', index=False)

    print('\nGenerating figures...')
    for feat, title, fname in [
        ('voice_prop_passive',  'Proportion Passive Voice per Bloc',     'passive_bloc.pdf'),
        ('neg_prop_negated',    'Proportion Negated Sentences per Bloc', 'negation_bloc.pdf'),
        ('mod_prop_obligation', 'Obligation Modality per Bloc',          'modal_obligation_bloc.pdf'),
        ('mod_prop_hedging',    'Hedging Modality per Bloc',             'modal_hedging_bloc.pdf'),
        ('mod_prop_any_modal',  'Any Modal per Bloc',                    'modal_any_bloc.pdf'),
    ]:
        plot_bloc_bars(results, feat, title, fname)

    for src, df, pm, fname in [
        ('Motions',    mot_rich, pm_mot, 'heatmap_motions.pdf'),
        ('Manifestos', man_rich, pm_man, 'heatmap_manifestos.pdf'),
        ('Speeches',   sp_rich,  pm_sp,  'heatmap_speeches.pdf'),
    ]:
        plot_party_heatmap(pm, all_feats,
                           f'Linguistic Features per Party — {src}', fname)

    print(f'\nAll outputs saved to {OUT_DIR}')


if __name__ == '__main__':
    main()