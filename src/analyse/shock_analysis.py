"""
Micro-Level Shock Analysis: The 2019 Legal Turning Point
==================================================================
Tests whether the discursive embedding of climate terms shifts around
the 2019 legal turning point, and whether the lexical distance between
ideological blocs changes.

Two rulings bracket 2019:
  Council of State nitrogen ruling  — 29 May 2019 (first; triggers
                                      the nitrogen crisis)
  Supreme Court Urgenda ruling      — 20 December 2019

The cutoff is anchored on the nitrogen ruling (29 May 2019) and a
symmetric two-year DATE window is used (pre: 2017-05-29 to 2019-05-29,
post: 2019-05-29 to 2021-05-29) rather than calendar years, so that
documents are assigned to pre/post by their actual date. The Urgenda
ruling falls inside the post window. Both motions and speeches carry a
full date column, so assignment is exact for both sources.

Outputs:
  - Contrastive TF-IDF terms per bloc (emergent vs faded)
  - Between-bloc left-right JSD per period, with bootstrap CI and a
    permutation test on the pre-vs-post change
  - Within-bloc JSD (pre vs post) per bloc, showing which bloc shifted
    its own climate vocabulary most

Input:
    results/motions/macro_scores/qwen2.5-32b.csv   (motions)
    results/speeches/macro_scores/qwen2.5-32b.csv (speeches)

Output:
    results/analysis/
        legal_turning_point_shift_table.csv
        jsd_results.csv
        figures/jsd_pre_post.pdf
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)
rng = np.random.default_rng(42)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

OUT_DIR = Path('results/analysis')
FIG_DIR = OUT_DIR / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_WORDS = ['klimaat', 'energie', 'uitstoot', 'natuur', 'transitie',
                'stikstof', 'duurzaam', 'subsidie']

LEFT   = ['GroenLinks', 'PvdA', 'GroenLinks-PvdA', 'PvdD']
RIGHT  = ['VVD', 'PVV', 'FVD', 'BBB']
CENTER = ['D66', 'CDA']
TARGET_PARTIES = LEFT + RIGHT + CENTER

# 2019 legal turning point: cutoff on the Council of State nitrogen
# ruling, symmetric two-year date window.
EVENT_DATE  = pd.Timestamp('2019-05-29', tz='UTC')
WINDOW      = pd.DateOffset(years=2)
PRE_START   = EVENT_DATE - WINDOW   # 2017-05-29
POST_END    = EVENT_DATE + WINDOW   # 2021-05-29
WINDOW_SIZE = 7
N_BOOTSTRAP = 1000

BLOC_COLORS = {'Left': '#2e8b57', 'Right': '#e63946'}

DUTCH_STOPS = set(stopwords.words('dutch'))
EXTRA_STOPS = {
    'constaterende', 'overwegende', 'verzoekt', 'motie', 'kamer',
    'regering', 'minister', 'staatssecretaris', 'tweede', 'eerste',
    'lid', 'leden', 'vergadering', 'voorzitter', 'beraadslaging',
    'gehoord', 'gelet', 'mening', 'besluit', 'gegeven', 'verklaring',
    'nederland', 'nederlands', 'kabinet', 'jaar', 'jaren',
    'heer', 'mevrouw', 'collega', 'fractie', 'partij',
}
ALL_STOPS = DUTCH_STOPS | EXTRA_STOPS

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 11, 'figure.dpi': 150,
})


def bloc_of(p: str) -> str:
    if p in LEFT:  return 'Left'
    if p in RIGHT: return 'Right'
    return 'Center'


def to_period(ts):
    """Assign a timestamp to 'pre', 'post', or None (outside the window)."""
    if pd.isna(ts):
        return None
    if PRE_START <= ts < EVENT_DATE:
        return 'pre'
    if EVENT_DATE <= ts < POST_END:
        return 'post'
    return None


def tokenize(text: str) -> list:
    if not isinstance(text, str):
        return []
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return [t for t in text.split() if len(t) > 2]


def extract_windows(text: str, target: str, window: int = 7) -> list:
    tokens = tokenize(text)
    tl = target.lower()
    out = []
    for i, tok in enumerate(tokens):
        if tok == tl or tok.startswith(tl):
            ctx = [t for t in (tokens[max(0, i - window):i] +
                               tokens[i + 1:i + 1 + window])
                   if t not in ALL_STOPS
                   and not t.startswith(tl) and len(t) > 3]
            if ctx:
                out.append(' '.join(ctx))
    return out


def load_combined() -> pd.DataFrame:
    # Motions: full date in 'date'
    mot = pd.read_csv(
        'results/motions/macro_scores/qwen2.5-32b.csv')
    mot['party'] = mot['fractions'].str.split(';').str[0].str.strip()
    mot['ts'] = pd.to_datetime(mot['date'], utc=True, errors='coerce')
    mot = mot[mot['party'].isin(TARGET_PARTIES)].copy()
    mot = mot[['party', 'ts', 'normalized_text']].rename(
        columns={'normalized_text': 'text'})
    mot['source'] = 'motion'

    # Speeches: full date in 'date'
    sp = pd.read_csv(
        'results/speeches/macro_scores/qwen2.5-32b.csv'
    ).rename(columns={'party_name': 'party'})
    sp['ts'] = pd.to_datetime(sp['date'], utc=True, errors='coerce')
    sp = sp[sp['party'].isin(TARGET_PARTIES)].copy()
    sp = sp[['party', 'ts', 'text']].copy()
    sp['source'] = 'speech'

    both = pd.concat([mot, sp], ignore_index=True)
    both = both[both['text'].notna()].copy()
    both['bloc']   = both['party'].apply(bloc_of)
    both['period'] = both['ts'].apply(to_period)
    both = both[both['period'].notna()].copy()
    return both


def get_windows_by_period(df: pd.DataFrame, bloc: str) -> dict:
    out = {'pre': [], 'post': []}
    for _, row in df[df['bloc'] == bloc].iterrows():
        for target in TARGET_WORDS:
            out[row['period']].extend(
                extract_windows(row['text'], target, WINDOW_SIZE))
    return out


def contrastive_terms(focal: list, other: list, top_k: int = 12) -> list:
    if len(focal) < 5 or len(other) < 5:
        return []
    all_docs = focal + other
    vec = TfidfVectorizer(
        max_features=8000, min_df=3, stop_words=list(ALL_STOPS),
        sublinear_tf=True, token_pattern=r'\b[a-zA-Zà-ÿ]{4,}\b')
    mat   = vec.fit_transform(all_docs)
    feats = vec.get_feature_names_out()
    diff  = mat[:len(focal)].mean(axis=0).A1 - mat[len(focal):].mean(axis=0).A1
    return [feats[i] for i in diff.argsort()[::-1][:top_k] if diff[i] > 0]


def word_distribution(windows: list) -> Counter:
    counter = Counter()
    for w in windows:
        for tok in w.split():
            if tok not in ALL_STOPS and len(tok) > 3:
                counter[tok] += 1
    return counter


def jsd_between(dist_a: Counter, dist_b: Counter) -> float:
    vocab = set(dist_a) | set(dist_b)
    ta, tb = sum(dist_a.values()), sum(dist_b.values())
    pa = np.array([dist_a.get(w, 0) / ta for w in vocab])
    pb = np.array([dist_b.get(w, 0) / tb for w in vocab])
    return jensenshannon(pa, pb, base=2)


def all_tokens(windows: list) -> list:
    """Flatten windows to a token list."""
    toks = []
    for w in windows:
        toks.extend([t for t in w.split()
                     if t not in ALL_STOPS and len(t) > 3])
    return toks


def bootstrap_jsd(left_toks: list, right_toks: list,
                  n: int = 1000) -> tuple:
    """Bootstrap JSD by resampling tokens. Returns (observed, array)."""
    left_arr  = np.array(left_toks)
    right_arr = np.array(right_toks)
    observed  = jsd_between(Counter(left_toks), Counter(right_toks))
    out = np.empty(n)
    for b in range(n):
        l = Counter(rng.choice(left_arr,  size=len(left_arr),  replace=True))
        r = Counter(rng.choice(right_arr, size=len(right_arr), replace=True))
        out[b] = jsd_between(l, r)
    return observed, out


def permutation_test_delta(pre_left: list, pre_right: list,
                            post_left: list, post_right: list,
                            n: int = 1000) -> tuple:
    """
    Permutation test on the change in left-right JSD (post - pre).
    Shuffles period labels within each bloc and recomputes the delta.
    """
    obs_pre   = jsd_between(Counter(pre_left),  Counter(pre_right))
    obs_post  = jsd_between(Counter(post_left), Counter(post_right))
    obs_delta = obs_post - obs_pre

    left_all  = np.array(pre_left + post_left)
    right_all = np.array(pre_right + post_right)
    n_pre_l, n_pre_r = len(pre_left), len(pre_right)

    null_deltas = np.empty(n)
    for b in range(n):
        lperm = rng.permutation(left_all)
        rperm = rng.permutation(right_all)
        pre_l, post_l = lperm[:n_pre_l], lperm[n_pre_l:]
        pre_r, post_r = rperm[:n_pre_r], rperm[n_pre_r:]
        d_pre  = jsd_between(Counter(pre_l),  Counter(pre_r))
        d_post = jsd_between(Counter(post_l), Counter(post_r))
        null_deltas[b] = d_post - d_pre

    p_val = (np.sum(np.abs(null_deltas) >= np.abs(obs_delta)) + 1) / (n + 1)
    return obs_pre, obs_post, obs_delta, p_val


def plot_jsd(obs_pre, obs_post, p_val):
    fig, ax = plt.subplots(figsize=(7, 5))
    periods = ['Pre\n(2017-05 to 2019-05)', 'Post\n(2019-05 to 2021-05)']
    means   = [obs_pre, obs_post]

    ax.bar(periods, means, color=['#8da0cb', '#fc8d62'], alpha=0.85,
           edgecolor='black', linewidth=0.8, width=0.6)
    for i, m in enumerate(means):
        ax.text(i, m + 0.008, f'{m:.3f}', ha='center', fontsize=12)

    y = max(means) + 0.04
    ax.plot([0, 0, 1, 1], [y, y + 0.01, y + 0.01, y],
            color='black', linewidth=1)
    ax.text(0.5, y + 0.015, f'p = {p_val:.3f}', ha='center', fontsize=11)

    ax.set_ylabel('Jensen-Shannon divergence\n(left vs right climate discourse)')
    ax.set_title('Left-right lexical divergence around climate terms')
    ax.set_ylim(0, max(means) * 1.3)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'jsd_pre_post.pdf', bbox_inches='tight')
    plt.close()
    print('Saved: jsd_pre_post.pdf')


def main():
    print('Loading combined motions + speeches...')
    df = load_combined()
    print(f'  {len(df)} texts in window')
    print(df.groupby(['bloc', 'period', 'source']).size().to_string())

    rows = []
    for bloc in ['Left', 'Center', 'Right']:
        print(f'\nProcessing {bloc}...')
        wins = get_windows_by_period(df, bloc)
        print(f'  pre: {len(wins["pre"])} windows, '
              f'post: {len(wins["post"])} windows')
        emerged = contrastive_terms(wins['post'], wins['pre'], top_k=12)
        faded   = contrastive_terms(wins['pre'], wins['post'], top_k=12)
        print(f'  Emerged: {", ".join(emerged)}')
        print(f'  Faded:   {", ".join(faded)}')
        rows.append({'bloc': bloc, 'n_pre': len(wins['pre']),
                     'n_post': len(wins['post']),
                     'emerged': ', '.join(emerged),
                     'faded': ', '.join(faded)})
    pd.DataFrame(rows).to_csv(
        OUT_DIR / 'legal_turning_point_shift_table.csv', index=False)

    # ── Token sets per bloc per period ──
    left_wins  = get_windows_by_period(df, 'Left')
    right_wins = get_windows_by_period(df, 'Right')
    pre_left   = all_tokens(left_wins['pre'])
    pre_right  = all_tokens(right_wins['pre'])
    post_left  = all_tokens(left_wins['post'])
    post_right = all_tokens(right_wins['post'])

    # ── Between-bloc left-right JSD per period ──
    print('\n=== Between-bloc left-right JSD ===')
    print(f'Bootstrapping ({N_BOOTSTRAP} resamples)...')
    obs_pre,  pre_ci  = bootstrap_jsd(pre_left,  pre_right,  N_BOOTSTRAP)
    obs_post, post_ci = bootstrap_jsd(post_left, post_right, N_BOOTSTRAP)

    print('Running permutation test on the change...')
    _, _, _, p_val = permutation_test_delta(
        pre_left, pre_right, post_left, post_right, N_BOOTSTRAP)
    obs_delta = obs_post - obs_pre

    print(f'\nJSD pre:   {obs_pre:.4f}  '
          f'95% CI [{np.percentile(pre_ci, 2.5):.4f}, '
          f'{np.percentile(pre_ci, 97.5):.4f}]')
    print(f'JSD post:  {obs_post:.4f}  '
          f'95% CI [{np.percentile(post_ci, 2.5):.4f}, '
          f'{np.percentile(post_ci, 97.5):.4f}]')
    print(f'Delta:     {obs_delta:+.4f}')
    print(f'Permutation test p-value (two-sided): {p_val:.4f}')

    # ── Within-bloc shift: how much did each bloc's own vocabulary move? ──
    print('\n=== Within-bloc lexical shift (pre vs post) ===')
    left_shift  = jsd_between(Counter(post_left),  Counter(pre_left))
    right_shift = jsd_between(Counter(post_right), Counter(pre_right))
    print(f'  Left  (pre vs post): {left_shift:.4f}  '
          f'(n_pre={len(pre_left)}, n_post={len(post_left)})')
    print(f'  Right (pre vs post): {right_shift:.4f}  '
          f'(n_pre={len(pre_right)}, n_post={len(post_right)})')

    pd.DataFrame([{
        'jsd_pre': obs_pre, 'jsd_post': obs_post, 'delta': obs_delta,
        'pre_ci_low': np.percentile(pre_ci, 2.5),
        'pre_ci_high': np.percentile(pre_ci, 97.5),
        'post_ci_low': np.percentile(post_ci, 2.5),
        'post_ci_high': np.percentile(post_ci, 97.5),
        'perm_p_value': p_val,
        'within_left_shift':  left_shift,
        'within_right_shift': right_shift,
    }]).to_csv(OUT_DIR / 'jsd_results.csv', index=False)

    plot_jsd(obs_pre, obs_post, p_val)
    print(f'\nSaved to {OUT_DIR}')


if __name__ == '__main__':
    main()