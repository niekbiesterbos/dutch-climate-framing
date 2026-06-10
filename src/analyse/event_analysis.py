"""
Climate Attention and Framing Around Key Events (motions only)
========================================================================
For climate-relevant events, measures (a) the share of all parliamentary
motions concerning climate in a symmetric 12-month window before and
after the event, as a measure of parliamentary attention, and (b)
whether issue-frame scores shift across the same window.

Motions only: they span 2008-2025 continuously, so every event gets a
clean symmetric window, and the full motion corpus provides a
denominator for the attention measure. Speeches are excluded because the
speech corpus ends in 2022, giving no coverage for the Paris Agreement
and a truncated post-window for the Russian invasion, which would break
comparability across events.

Input:
    data/motions/all_motions_2008_2025.csv
    results/motions/macro_scores/qwen2.5-32b.csv
Output:
    results/analysis/
        event_volume.csv
        event_framing.csv
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
OUT = Path('results/analysis')
OUT.mkdir(parents=True, exist_ok=True)

FRAMES = ['economic', 'moral', 'scientific', 'security',
          'health_environment', 'crisis_urgency', 'weaponization']

EVENTS = {
    'Paris Agreement':      pd.Timestamp('2015-12-12', tz='UTC'),
    '2019 nitrogen ruling': pd.Timestamp('2019-05-29', tz='UTC'),
    'IPCC AR6':             pd.Timestamp('2021-08-09', tz='UTC'),
    'Russia-Ukraine':       pd.Timestamp('2022-02-24', tz='UTC'),
}
WINDOW = pd.DateOffset(months=12)

TARGET = ['GroenLinks', 'PvdA', 'GroenLinks-PvdA', 'PvdD', 'D66', 'CDA',
          'VVD', 'PVV', 'FVD', 'BBB']


def sig(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''


def load_climate_motions():
    mot = pd.read_csv(
        'results/motions/macro_scores/qwen2.5-32b.csv')
    mot['party'] = mot['fractions'].str.split(';').str[0].str.strip()
    mot['ts'] = pd.to_datetime(mot['date'], utc=True, errors='coerce')
    mot = mot.rename(columns={f'frame_{f}': f for f in FRAMES})
    mot = mot[mot['party'].isin(TARGET)]
    mot = mot[mot[FRAMES].ne(-1).all(axis=1)]
    mot = mot[~(mot[FRAMES] == 1).all(axis=1)]
    return mot[['ts'] + FRAMES].copy()


def load_all_motions():
    df = pd.read_csv(
        'data/TK_motions/'
        'all_motions_2008_2025_with_text_and_normalized.csv')
    df['party'] = df['fractions'].str.split(';').str[0].str.strip()
    df['ts'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
    df = df[df['party'].isin(TARGET)]
    print(f'  all motions (target parties): {len(df)}')
    return df[['ts']].copy()


def count_window(df, ev):
    pre  = df[(df['ts'] >= ev - WINDOW) & (df['ts'] < ev)]
    post = df[(df['ts'] >= ev) & (df['ts'] < ev + WINDOW)]
    return pre, post


def main():
    mot  = load_climate_motions()
    print('Loading full motion corpus for denominator...')
    allm = load_all_motions()

    vol_rows, frame_rows = [], []
    for name, ev in EVENTS.items():
        cpre, cpost = count_window(mot, ev)
        apre, apost = count_window(allm, ev)
        vol_rows.append({
            'event': name, 'date': ev.date(),
            'n_climate_pre': len(cpre), 'n_climate_post': len(cpost),
            'n_all_pre': len(apre), 'n_all_post': len(apost),
            'share_pre_pct':  round(100 * len(cpre) / max(len(apre), 1), 2),
            'share_post_pct': round(100 * len(cpost) / max(len(apost), 1), 2),
        })
        for f in FRAMES:
            if len(cpre) < 10 or len(cpost) < 10:
                continue
            s, p = mannwhitneyu(cpre[f], cpost[f], alternative='two-sided')
            r = 1 - (2 * s) / (len(cpre) * len(cpost))
            frame_rows.append({
                'event': name, 'frame': f,
                'pre': round(cpre[f].mean(), 3),
                'post': round(cpost[f].mean(), 3),
                'diff': round(cpost[f].mean() - cpre[f].mean(), 3),
                'r': round(r, 3), 'sig': sig(p)})

    vol = pd.DataFrame(vol_rows)
    vol.to_csv(OUT / 'event_volume.csv', index=False)
    fr = pd.DataFrame(frame_rows)
    fr.to_csv(OUT / 'event_framing.csv', index=False)

    print('\n=== Climate share of all motions around events (±12 months) ===')
    print(vol.to_string(index=False))
    print('\n=== Significant framing shifts (motions) ===')
    print(fr[fr['sig'] != ''].to_string(index=False))


if __name__ == '__main__':
    main()