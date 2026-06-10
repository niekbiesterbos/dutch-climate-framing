"""
Negation Profiling, Active/Passive Voice, and Example Sentences
=========================================================================
"""

import os
import re
import numpy as np
import pandas as pd
import spacy
from pathlib import Path
from collections import Counter, defaultdict
from scipy.stats import chi2_contingency

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

nlp = spacy.load('nl_core_news_lg')
if 'ner' in nlp.pipe_names:
    nlp.disable_pipes(['ner'])

OUT_DIR = Path('results/analysis')
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEFT   = ['GroenLinks', 'PvdA', 'GroenLinks-PvdA', 'PvdD']
RIGHT  = ['VVD', 'PVV', 'FvD', 'BBB']
CENTER = ['D66', 'CDA']
TARGET = LEFT + RIGHT + CENTER
PARTIES = ['PVV', 'FvD', 'BBB', 'VVD', 'CDA', 'D66',
           'GroenLinks', 'PvdA', 'PvdD']

PARTY_NORM = {
    'fvd': 'FvD', 'pvv': 'PVV', 'vvd': 'VVD', 'cda': 'CDA',
    'd66': 'D66', 'pvda': 'PvdA', 'pvdd': 'PvdD', 'bbb': 'BBB',
    'groenlinks': 'GroenLinks', 'groenlinks-pvda': 'GroenLinks-PvdA',
}

NEG_LEMMAS = {
    'niet', 'geen', 'noch', 'nooit', 'nauwelijks', 'zelden',
    'onvoldoende', 'nimmer', 'amper', 'nergens', 'geenszins',
    'allerminst',
}

STOP_NEG = {
    'zijn', 'worden', 'hebben', 'kunnen', 'zullen', 'moeten',
    'willen', 'gaan', 'doen', 'zeggen', 'komen', 'weten', 'maken',
    'staan', 'liggen', 'blijven', 'lijken', 'heten',
    'dat', 'dit', 'het', 'die', 'wat', 'meer', 'lang', 'veel',
    'alleen', 'altijd', 'ook', 'nog', 'wel', 'al', 'verder',
    'direct', 'vaak', 'soms', 'zeker', 'reeds', 'echter', 'toch',
    'andere', 'ander', 'groot', 'goed', 'nieuw', 'hoog', 'laag',
    'lang', 'kort', 'snel', 'duur', 'vrij', 'eigen',
    'iemand', 'iedereen', 'niemand', 'alles', 'niets', 'men',
    'motie', 'kamer', 'vergadering', 'lid', 'leden', 'voorzitter',
    'beraadslaging', 'stemming', 'behandeling', 'agenda',
    'wilders', 'ouwehand', 'thieme', 'klaver', 'kaag', 'rutte',
    'tongeren', 'grashoff', 'moorlag', 'jacobi', 'veldhoven',
    'plas', 'eppink', 'eerdmans', 'jansen', 'haga', 'raan',
    'kröger', 'thijssen', 'paternotte', 'bouchallikh',
    'dijkstra', 'veltman', 'vestering', 'esch', 'kostic',
    'bromet', 'mulder', 'bontenbal', 'vedder', 'geurts',
    'ïndexeren', 'deelen',
    'ding', 'week', 'manier', 'gebruiken', 'vallen', 'trekken',
    'overgaan', 'inzien', 'keer', 'benutten', 'vaststellen',
    'richten', 'verwachten',
}

FRAME_COLS = ['economic', 'moral', 'scientific', 'security',
              'health_environment', 'crisis_urgency', 'weaponization']

SCORE_PATHS = {
    'motions':   ('results/motions/macro_scores/qwen2.5-32b.csv',   'id'),
    'manifestos':('results/manifestos/macro_scores/qwen2.5-32b.csv', 'text'),
    'speeches':  ('results/speeches/macro_scores/qwen2.5-32b.csv', 'doc_id'),
}


def normalize_party(p: str) -> str:
    if not isinstance(p, str):
        return ''
    return PARTY_NORM.get(p.lower().strip(), p.strip())


def load_all_ones_ids() -> dict:
    """Return set of all-ones IDs per source."""
    all_ones = {}
    for source, (path, id_col) in SCORE_PATHS.items():
        df   = pd.read_csv(path)
        mask = (df[FRAME_COLS] == 1).all(axis=1)
        all_ones[source] = set(df.loc[mask, id_col].astype(str))
        print(f'  {source}: excluding {mask.sum()} all-ones items')
    return all_ones


def clean_motion_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = re.sub(
        r'(?i)^.*?gehoord\s+de\s+beraadslaging[,\s]*',
        '', text, flags=re.DOTALL)
    text = re.sub(
        r'(?i)en\s+gaat\s+over\s+tot\s+de\s+orde\s+van\s+de\s+dag\.?',
        '', text)
    return ' '.join(text.split()).strip()


def is_passive(sent) -> bool:
    for token in sent:
        if token.dep_ in ('nsubj:pass', 'aux:pass'):
            return True
    return False


def get_neg_targets(sent) -> list:
    targets = []
    for token in sent:
        if token.lemma_.lower() in NEG_LEMMAS:
            head = token.head
            if (head.pos_ in ('NOUN', 'VERB', 'ADJ', 'PROPN', 'ADV')
                    and len(head.lemma_) > 2
                    and not head.is_stop
                    and head.lemma_.lower() not in STOP_NEG):
                targets.append(head.lemma_.lower())
        if token.lemma_.lower() in {
                'stoppen', 'beëindigen', 'verbieden', 'afschaffen',
                'afbouwen', 'beperken', 'tegengaan', 'voorkomen',
                'vermijden'}:
            for child in token.children:
                if child.lemma_.lower() == 'met':
                    for gc in child.children:
                        if (gc.pos_ in ('NOUN', 'PROPN')
                                and len(gc.lemma_) > 3
                                and gc.lemma_.lower() not in STOP_NEG):
                            targets.append(gc.lemma_.lower())
    return targets


def load_all_texts(all_ones_ids: dict) -> tuple[list, dict]:
    """Load texts across all three sources, excluding all-ones items."""
    rows = []
    texts_per_party = defaultdict(list)

    # motions
    mot = pd.read_csv(
        'results/motions/macro_scores/qwen2.5-32b.csv')
    mot['id'] = mot['id'].astype(str)
    mot = mot[~mot['id'].isin(all_ones_ids['motions'])]
    mot['party'] = (mot['fractions'].str.split(';').str[0]
                    .str.strip().apply(normalize_party))
    mot = mot[mot['party'].isin(TARGET)
              & mot['normalized_text'].notna()].copy()
    for _, r in mot.iterrows():
        cleaned = clean_motion_text(r['normalized_text'])
        if cleaned:
            rows.append({'text': cleaned[:3000], 'party': r['party'],
                         'source': 'motions'})
            texts_per_party[r['party']].append(('motion', cleaned[:3000]))

    # manifestos
    man = pd.read_csv(
        'results/manifestos/macro_scores/qwen2.5-32b.csv')
    man['text'] = man['text'].astype(str)
    man = man[~man['text'].isin(all_ones_ids['manifestos'])]
    man['party'] = man['party'].apply(normalize_party)
    man = man[man['party'].isin(TARGET) & man['text'].notna()].copy()
    for _, r in man.iterrows():
        rows.append({'text': r['text'][:3000], 'party': r['party'],
                     'source': 'manifestos'})
        texts_per_party[r['party']].append(('manifesto', r['text'][:3000]))

    # speeches
    sp = pd.read_csv(
        'results/speeches/macro_scores/qwen2.5-32b.csv')
    sp['doc_id'] = sp['doc_id'].astype(str)
    sp = sp[~sp['doc_id'].isin(all_ones_ids['speeches'])]
    sp = sp.rename(columns={'party_name': 'party'})
    sp['party'] = sp['party'].apply(normalize_party)
    sp = sp[sp['party'].isin(TARGET) & sp['text'].notna()].copy()
    for _, r in sp.iterrows():
        rows.append({'text': r['text'][:3000], 'party': r['party'],
                     'source': 'speeches'})
        texts_per_party[r['party']].append(('speech', r['text'][:3000]))

    n_mot = sum(1 for r in rows if r['source'] == 'motions')
    n_man = sum(1 for r in rows if r['source'] == 'manifestos')
    n_sp  = sum(1 for r in rows if r['source'] == 'speeches')
    print(f'Loaded {len(rows)} texts '
          f'({n_mot} motions, {n_man} manifestos, {n_sp} speeches)')
    return rows, texts_per_party


def load_macro_scores() -> pd.DataFrame:
    frames = ['frame_weaponization', 'frame_economic', 'frame_crisis_urgency',
              'frame_health_environment', 'frame_security',
              'frame_moral', 'frame_scientific']
    dfs = []

    mot = pd.read_csv(
        'results/motions/macro_scores/qwen2.5-32b.csv')
    mot['party'] = (mot['fractions'].str.split(';').str[0]
                    .str.strip().apply(normalize_party))
    available = [f for f in frames if f in mot.columns]
    dfs.append(mot[['party'] + available])

    man = pd.read_csv(
        'results/manifestos/macro_scores/qwen2.5-32b.csv')
    man['party'] = man['party'].apply(normalize_party)
    available = [f for f in frames if f in man.columns]
    if available:
        dfs.append(man[['party'] + available])

    sp = pd.read_csv(
        'results/speeches/macro_scores/qwen2.5-32b.csv')
    sp = sp.rename(columns={'party_name': 'party'})
    sp['party'] = sp['party'].apply(normalize_party)
    available = [f for f in frames if f in sp.columns]
    if available:
        dfs.append(sp[['party'] + available])

    combined = pd.concat(dfs, ignore_index=True)
    available = [f for f in frames if f in combined.columns]
    macro = (combined[combined['party'].isin(TARGET)]
             .groupby('party')[available].mean().round(3))
    return macro


def find_examples(party: str,
                  target_lemma: str,
                  texts_per_party: dict,
                  max_ex: int = 2) -> list:
    examples = []
    for source, text in texts_per_party.get(party, []):
        if len(examples) >= max_ex:
            break
        doc = nlp(text)
        for sent in doc.sents:
            if len(examples) >= max_ex:
                break
            lemmas = [t.lemma_.lower() for t in sent]
            if target_lemma not in lemmas:
                continue
            has_neg = any(t.lemma_.lower() in NEG_LEMMAS for t in sent)
            if not has_neg:
                continue
            s = sent.text.strip()
            if len(s) > 30:
                examples.append(f'[{source}] {s[:250]}')
    return examples


def compute_distinctive_negations(neg_counts: dict,
                                  top_k: int = 12) -> dict:
    results = {}
    for focal in PARTIES:
        focal_c = neg_counts.get(focal, Counter())
        other_c: Counter = Counter()
        for p, c in neg_counts.items():
            if p != focal:
                other_c.update(c)

        focal_total = sum(focal_c.values())
        other_total = sum(other_c.values())

        scores = {}
        for term, f_cnt in focal_c.items():
            if f_cnt < 5:
                continue
            o_cnt  = other_c.get(term, 0)
            f_freq = (f_cnt + 0.5) / (focal_total + 1)
            o_freq = (o_cnt + 0.5) / (other_total + 1)
            scores[term] = np.log(f_freq / o_freq)

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results[focal] = [
            (term, focal_c[term], score)
            for term, score in top[:top_k]
        ]
    return results


def main():
    print('Loading all-ones IDs...')
    all_ones_ids = load_all_ones_ids()

    rows, texts_per_party = load_all_texts(all_ones_ids)

    neg_counts  = defaultdict(Counter)
    passive_all = defaultdict(lambda: {'active': 0, 'passive': 0})
    passive_src = defaultdict(
        lambda: defaultdict(lambda: {'active': 0, 'passive': 0}))

    print('\nProcessing texts...')
    for i, row in enumerate(rows):
        if i % 2000 == 0:
            print(f'  {i}/{len(rows)}')
        doc    = nlp(row['text'])
        party  = row['party']
        source = row['source']
        for sent in doc.sents:
            pkey = 'passive' if is_passive(sent) else 'active'
            passive_all[party][pkey] += 1
            passive_src[party][source][pkey] += 1
            for t in get_neg_targets(sent):
                neg_counts[party][t] += 1

    # ── Analysis 1: Distinctive negations + examples ──────────────────────────
    print('\n' + '='*65)
    print('ANALYSIS 1: Distinctive negation targets + example sentences')
    print('='*65)

    distinctive = compute_distinctive_negations(neg_counts)

    neg_rows = []
    for party in PARTIES:
        top = distinctive.get(party, [])
        total = sum(neg_counts[party].values())
        print(f'\n=== {party} (total neg tokens: {total}) ===')

        for term, count, score in top[:5]:
            print(f'  {count:4d}x  {term:<28} log_ratio={score:.3f}')
            exs = find_examples(party, term, texts_per_party, max_ex=2)
            for ex in exs:
                print(f'         > {ex}')

        neg_rows.append({
            'party': party,
            'total_neg_tokens': total,
            'distinctive_negations': ', '.join(
                t for t, _, _ in top[:12]),
        })

    pd.DataFrame(neg_rows).to_csv(
        OUT_DIR / 'negation_targets_per_party.csv', index=False)

    # ── Macro frame scores per party ──────────────────────────────────────────
    print('\n' + '='*65)
    print('MACRO FRAME SCORES PER PARTY (mean, from EXP_9)')
    print('='*65)
    try:
        macro = load_macro_scores()
        print(macro.to_string())
        macro.to_csv(OUT_DIR / 'macro_scores_per_party.csv')
    except Exception as e:
        print(f'Could not load macro scores: {e}')

    # ── Analysis 2: Active/Passive ─────────────────────────────────────────────
    print('\n' + '='*65)
    print('ANALYSIS 2: Active/Passive ratio per party')
    print('='*65)

    passive_rows = []
    for party in PARTIES:
        r     = passive_all.get(party, {'active': 0, 'passive': 0})
        total = r['active'] + r['passive']
        if total == 0:
            continue
        pct = r['passive'] / total * 100
        print(f'\n{party:<20} n={total:7d}  '
              f'active={r["active"]/total*100:.1f}%  '
              f'passive={pct:.1f}%')
        for src in ['motions', 'manifestos', 'speeches']:
            rs  = passive_src[party].get(
                src, {'active': 0, 'passive': 0})
            tot = rs['active'] + rs['passive']
            if tot == 0:
                continue
            print(f'  {src:<12} n={tot:6d}  '
                  f'passive={rs["passive"]/tot*100:.1f}%')
        passive_rows.append({
            'party': party, 'total': total,
            'n_active': r['active'], 'n_passive': r['passive'],
            'pct_passive': round(pct, 2),
        })

    pd.DataFrame(passive_rows).to_csv(
        OUT_DIR / 'passive_rate_per_party.csv', index=False)

    print('\nPairwise chi-square:')
    for pa, pb in [('PVV', 'PvdD'), ('FvD', 'GroenLinks'),
                   ('BBB', 'PvdA'), ('VVD', 'PvdD')]:
        ra = passive_all.get(pa, {'active': 0, 'passive': 0})
        rb = passive_all.get(pb, {'active': 0, 'passive': 0})
        if min(ra['active'] + ra['passive'],
               rb['active'] + rb['passive']) == 0:
            continue
        chi2, p, _, _ = chi2_contingency([
            [ra['active'], ra['passive']],
            [rb['active'], rb['passive']],
        ])
        print(f'  {pa:<20} vs {pb:<20} chi2={chi2:.2f}  p={p:.6f}')

    print(f'\nAll outputs saved to {OUT_DIR}')


if __name__ == '__main__':
    main()