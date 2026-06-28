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
ruling falls inside the post window.

Three analytical levels (§4.1.6):
  Level 1 — Keyness of context words per bloc (log-likelihood G2 +
             log-ratio; Rayson & Garside 2000; Hardie 2014).
  Level 2 — LOME lexical-frame shift per bloc (Mann-Whitney U on
             per-document frame counts; Table 5.11).
  Level 3 — Left-right JSD per period and within-bloc JSD shift,
             reported as plain descriptive distances (no inferential
             test, per §4.1.6).

Input:
    results/motions/macro_scores/qwen2.5-32b.csv
    results/speeches/macro_scores/qwen2.5-32b.csv
    results/analysis/frame_occurrences_motions.csv   (optional)
    results/analysis/frame_occurrences_speeches.csv  (optional)

Output:
    results/analysis/
        legal_turning_point_shift_table.csv
        lome_frame_shift.csv
        jsd_results.csv
        figures/jsd_pre_post.pdf
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

OUT_DIR = Path('results/analysis')
FIG_DIR = OUT_DIR / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_WORDS = ['klimaat', 'energie', 'uitstoot', 'natuur', 'transitie',
                'stikstof', 'duurzaam', 'subsidie']

LEFT   = ['GroenLinks', 'PvdA', 'GroenLinks-PvdA', 'PvdD']
RIGHT  = ['VVD', 'PVV', 'FvD', 'BBB']
CENTER = ['D66', 'CDA']
TARGET_PARTIES = LEFT + RIGHT + CENTER

# Table 5.11 frames
FOCUS_FRAMES = ['Expensiveness', 'Protecting', 'Catastrophe', 'Activity_stop', 'Origin']

EVENT_DATE = pd.Timestamp('2019-05-29', tz='UTC')
WINDOW     = pd.DateOffset(years=2)
PRE_START  = EVENT_DATE - WINDOW   # 2017-05-29
POST_END   = EVENT_DATE + WINDOW   # 2021-05-29
WINDOW_SIZE = 7

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
    mot = pd.read_csv('results/motions/macro_scores/qwen2.5-32b.csv')
    mot['party'] = mot['fractions'].str.split(';').str[0].str.strip()
    mot['ts'] = pd.to_datetime(mot['date'], utc=True, errors='coerce')
    mot = mot[mot['party'].isin(TARGET_PARTIES)].copy()
    mot = mot[['party', 'ts', 'normalized_text']].rename(
        columns={'normalized_text': 'text'})
    mot['source'] = 'motion'

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


def keyness_terms(target_wins: list, ref_wins: list, top_k: int = 12,
                  min_freq: int = 5, g2_thresh: float = 6.63) -> list:
    """
    Log-likelihood G2 + log-ratio keyness (Rayson & Garside 2000; Hardie 2014).
    Returns terms over-used in target vs. reference, sorted by G2 descending.
    Retains a term if: freq >= min_freq in target, log-ratio > 0, G2 > g2_thresh
    (p < 0.01 at 1 df). Log-ratio uses add-0.5 smoothing to handle zero counts.
    """
    def count_toks(wins: list) -> Counter:
        c: Counter = Counter()
        for w in wins:
            for tok in w.split():
                if tok not in ALL_STOPS and len(tok) > 3:
                    c[tok] += 1
        return c

    t_cnt = count_toks(target_wins)
    r_cnt = count_toks(ref_wins)
    N_t   = max(sum(t_cnt.values()), 1)
    N_r   = max(sum(r_cnt.values()), 1)

    results = []
    for term, O11 in t_cnt.items():
        if O11 < min_freq:
            continue
        O12 = r_cnt.get(term, 0)
        N   = N_t + N_r
        E11 = N_t * (O11 + O12) / N
        E12 = N_r * (O11 + O12) / N
        if E11 <= 0 or E12 <= 0:
            continue
        g2 = 2 * (
            (O11 * np.log(O11 / E11) if O11 > 0 else 0) +
            (O12 * np.log(O12 / E12) if O12 > 0 else 0)
        )
        lr = np.log2((O11 + 0.5) / N_t) - np.log2((O12 + 0.5) / N_r)
        if lr > 0 and g2 > g2_thresh:
            results.append((term, g2))

    results.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in results[:top_k]]


def word_distribution(windows: list) -> Counter:
    counter: Counter = Counter()
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


def frame_shift_by_bloc() -> pd.DataFrame | None:
    """
    Table 5.11: LOME frame frequency change per bloc (Left/Center/Right) around
    the 2019 nitrogen ruling. Computes mean occurrences per document per period,
    with Mann-Whitney U test (two-sided) and rank-biserial r effect size.
    Requires frame_occurrences_{motions,speeches}.csv in results/analysis/.
    """
    from scipy.stats import mannwhitneyu

    occ_mot_path = OUT_DIR / 'frame_occurrences_motions.csv'
    occ_sp_path  = OUT_DIR / 'frame_occurrences_speeches.csv'
    if not occ_mot_path.exists() or not occ_sp_path.exists():
        print('  Frame occurrence CSVs not found — skipping frame shift.')
        return None

    def load_meta(csv_path: str, id_col: str, party_col: str) -> pd.DataFrame:
        cols = {id_col, 'date', party_col}
        if party_col == 'fractions':
            cols.add('fractions')
        df = pd.read_csv(csv_path, usecols=lambda c: c in cols)
        if party_col == 'fractions':
            df['party'] = df['fractions'].str.split(';').str[0].str.strip()
        else:
            df['party'] = df[party_col]
        df['ts']     = pd.to_datetime(df['date'], utc=True, errors='coerce')
        df['period'] = df['ts'].apply(to_period)
        df['doc_id'] = df[id_col].astype(str)
        return df[['doc_id', 'party', 'period']].dropna(subset=['period'])

    mot_meta = load_meta('results/motions/macro_scores/qwen2.5-32b.csv',
                         'id', 'fractions')
    sp_meta  = load_meta('results/speeches/macro_scores/qwen2.5-32b.csv',
                         'doc_id', 'party_name')
    meta = pd.concat([mot_meta, sp_meta], ignore_index=True)
    meta = meta[meta['party'].isin(TARGET_PARTIES)].copy()
    meta['bloc'] = meta['party'].apply(bloc_of)

    frames = pd.concat([
        pd.read_csv(occ_mot_path),
        pd.read_csv(occ_sp_path),
    ], ignore_index=True)
    frames['doc_id'] = frames['doc_id'].astype(str)
    frames = frames[frames['frame'].isin(FOCUS_FRAMES)].copy()
    frames = frames.merge(meta[['doc_id', 'bloc', 'period']], on='doc_id', how='inner')

    result_rows = []
    for frame in FOCUS_FRAMES:
        f_sub = frames[frames['frame'] == frame]
        for bloc in ['Left', 'Center', 'Right']:
            docs_bloc = meta[meta['bloc'] == bloc]
            per_period: dict = {}
            for period in ('pre', 'post'):
                docs_in  = docs_bloc[docs_bloc['period'] == period]['doc_id'].unique()
                occ_cnt  = (f_sub[(f_sub['bloc'] == bloc) &
                                  (f_sub['period'] == period)]
                            .groupby('doc_id').size()
                            .reindex(docs_in, fill_value=0))
                per_period[period] = occ_cnt.values

            pre_c, post_c = per_period['pre'], per_period['post']
            pre_mean  = pre_c.mean()  if len(pre_c)  > 0 else float('nan')
            post_mean = post_c.mean() if len(post_c) > 0 else float('nan')
            delta = post_mean - pre_mean

            sig, r_bs = '', None
            if len(pre_c) > 1 and len(post_c) > 1:
                try:
                    stat, p = mannwhitneyu(post_c, pre_c, alternative='two-sided')
                    n1, n2  = len(post_c), len(pre_c)
                    r_bs    = 1 - 2 * stat / (n1 * n2)
                    sig = ('***' if p < .001 else
                           ('**'  if p < .01  else
                            ('*'   if p < .05  else '')))
                except Exception:
                    pass

            result_rows.append({
                'frame': frame, 'bloc': bloc,
                'pre':   round(pre_mean,  3),
                'post':  round(post_mean, 3),
                'delta': round(delta,     3),
                'r_bs':  round(r_bs, 3) if r_bs is not None else None,
                'sig':   sig,
            })

    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(OUT_DIR / 'lome_frame_shift.csv', index=False)

    print('\n=== LOME Frame Shift (Table 5.11) ===')
    header = f"{'Frame':<20} {'Bloc':<8} {'Pre':>6} {'Post':>6} {'Δ':>7} {'Sig':>4}"
    print(header)
    print('-' * len(header))
    for frame in FOCUS_FRAMES:
        sub = result_df[result_df['frame'] == frame]
        for _, r in sub.iterrows():
            print(f"{r['frame']:<20} {r['bloc']:<8} {r['pre']:>6.3f} "
                  f"{r['post']:>6.3f} {r['delta']:>+7.3f} {r['sig']:>4}")
    return result_df


def plot_jsd(obs_pre: float, obs_post: float):
    fig, ax = plt.subplots(figsize=(7, 5))
    periods = ['Pre\n(2017-05 to 2019-05)', 'Post\n(2019-05 to 2021-05)']
    means   = [obs_pre, obs_post]

    ax.bar(periods, means, color=['#8da0cb', '#fc8d62'], alpha=0.85,
           edgecolor='black', linewidth=0.8, width=0.6)
    for i, m in enumerate(means):
        ax.text(i, m + 0.008, f'{m:.3f}', ha='center', fontsize=12)

    ax.set_ylabel('Jensen-Shannon divergence\n(left vs right climate discourse)')
    ax.set_title('Left-right lexical divergence around climate terms\n'
                 '(JSD reported as descriptive distance, §4.1.6)')
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

    # ── Level 1: Keyness of context words per bloc ──────────────────────────────
    rows = []
    for bloc in ['Left', 'Center', 'Right']:
        print(f'\nProcessing {bloc}...')
        wins = get_windows_by_period(df, bloc)
        print(f'  pre: {len(wins["pre"])} windows, '
              f'post: {len(wins["post"])} windows')
        emerged = keyness_terms(wins['post'], wins['pre'], top_k=12)
        faded   = keyness_terms(wins['pre'],  wins['post'], top_k=12)
        print(f'  Emerged: {", ".join(emerged)}')
        print(f'  Faded:   {", ".join(faded)}')
        rows.append({'bloc': bloc, 'n_pre': len(wins['pre']),
                     'n_post': len(wins['post']),
                     'emerged': ', '.join(emerged),
                     'faded':   ', '.join(faded)})
    pd.DataFrame(rows).to_csv(
        OUT_DIR / 'legal_turning_point_shift_table.csv', index=False)

    # ── Level 2: LOME frame shift per bloc ──────────────────────────────────────
    print('\n=== Level 2: LOME frame shift by bloc ===')
    frame_shift_by_bloc()

    # ── Level 3: JSD — reported descriptively, no inferential test (§4.1.6) ─────
    left_wins  = get_windows_by_period(df, 'Left')
    right_wins = get_windows_by_period(df, 'Right')
    pre_left   = all_tokens(left_wins['pre'])
    pre_right  = all_tokens(right_wins['pre'])
    post_left  = all_tokens(left_wins['post'])
    post_right = all_tokens(right_wins['post'])

    print('\n=== Between-bloc left-right JSD (descriptive) ===')
    obs_pre   = jsd_between(Counter(pre_left),  Counter(pre_right))
    obs_post  = jsd_between(Counter(post_left), Counter(post_right))
    obs_delta = obs_post - obs_pre
    print(f'JSD pre:   {obs_pre:.4f}')
    print(f'JSD post:  {obs_post:.4f}')
    print(f'Delta:     {obs_delta:+.4f}')

    print('\n=== Within-bloc lexical shift (pre vs post) ===')
    left_shift  = jsd_between(Counter(post_left),  Counter(pre_left))
    right_shift = jsd_between(Counter(post_right), Counter(pre_right))
    print(f'  Left  (pre vs post): {left_shift:.4f}  '
          f'(n_pre={len(pre_left)}, n_post={len(post_left)})')
    print(f'  Right (pre vs post): {right_shift:.4f}  '
          f'(n_pre={len(pre_right)}, n_post={len(post_right)})')

    pd.DataFrame([{
        'jsd_pre':            obs_pre,
        'jsd_post':           obs_post,
        'delta':              obs_delta,
        'within_left_shift':  left_shift,
        'within_right_shift': right_shift,
    }]).to_csv(OUT_DIR / 'jsd_results.csv', index=False)

    plot_jsd(obs_pre, obs_post)
    print(f'\nSaved to {OUT_DIR}')


if __name__ == '__main__':
    main()
