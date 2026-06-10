"""
TF-IDF Lexical Distinctiveness per Party and Bloc
===========================================================
Computes TF-IDF distinctiveness scores across all three text types combined
(parliamentary motions, election manifestos, parliamentary speeches),
balanced by source to avoid speeches dominating the corpus.
Also computes bloc-level TF-IDF (Left, Center, Right).

Input:  EXP_9/19/20 scores CSVs (all-ones excluded)
Output: results/analysis/
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

OUT_DIR = Path('results/analysis')
FIG_DIR = OUT_DIR / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

nlp = spacy.load('nl_core_news_lg', disable=['ner', 'parser'])

TARGET = ['groenlinks', 'pvda', 'groenlinks-pvda', 'd66', 'cda',
          'vvd', 'pvv', 'fvd', 'bbb', 'pvdd']

TARGET_DISPLAY = {
    'groenlinks':      'GroenLinks',
    'pvda':            'PvdA',
    'groenlinks-pvda': 'GroenLinks-PvdA',
    'd66':             'D66',
    'cda':             'CDA',
    'vvd':             'VVD',
    'pvv':             'PVV',
    'fvd':             'FvD',
    'bbb':             'BBB',
    'pvdd':            'PvdD',
}

LEFT   = ['groenlinks', 'pvda', 'groenlinks-pvda', 'pvdd']
CENTER = ['d66', 'cda']
RIGHT  = ['vvd', 'pvv', 'fvd', 'bbb']

BLOC_MAP = ({p: 'Left'   for p in LEFT}  |
            {p: 'Center' for p in CENTER} |
            {p: 'Right'  for p in RIGHT})

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

BLOC_COLORS = {
    'Left':   '#2e8b57',
    'Center': '#f4a261',
    'Right':  '#e63946',
}

DUTCH_STOPS = set(stopwords.words('dutch'))

EXTRA_STOPS = {
    'constaterende', 'overwegende', 'verzoekt', 'motie', 'kamer',
    'regering', 'minister', 'staatssecretaris', 'tweede', 'eerste',
    'lid', 'leden', 'dag', 'kst', 'vergaderjaar', 'voorgesteld',
    'vergadering', 'voorzitter', 'spreekt', 'tevens', 'beraadslaging',
    'gehoord', 'gelet', 'mening', 'besluit', 'gegeven', 'verklaring',
    'indiener', 'ondertekend', 'voorstellen', 'aangenomen', 'stemming',
    'debat', 'behandeling', 'agenda', 'vergaderstuk', 'kamerstuk',
    'groenlinks', 'pvda', 'pvv', 'cda', 'd66', 'vvd', 'fvd', 'bbb',
    'pvdd', 'partij', 'fractie', 'forum', 'democratie', 'volkspartij',
    'boerburgerbeweging', 'dieren', 'arbeid', 'vrijheid',
    'jaar', 'wel', 'ook', 'nog', 'dus', 'want', 'omdat', 'tenzij',
    'reeds', 'steeds', 'altijd', 'nooit', 'soms', 'vaak', 'zeker',
    'echter', 'daarbij', 'daarmee', 'daarvoor', 'daarna', 'daarin',
    'hierbij', 'hiermee', 'hiervoor', 'hierna', 'hierin', 'hierdoor',
    'waarbij', 'waarmee', 'waarvoor', 'waarna', 'waarin', 'waardoor',
    'gaan', 'gaat', 'komen', 'komt', 'maken', 'maakt', 'moeten',
    'moet', 'kunnen', 'kan', 'willen', 'wil', 'zijn', 'wij', 'hen',
    'heel', 'zeer', 'echt', 'juist', 'snel', 'groot', 'goed', 'nieuw',
    'heer', 'mevrouw', 'nederland', 'nederlands', 'kabinet',
    'vendrik', 'halsema', 'gent', 'ouwehand', 'thieme', 'wassenberg',
    'tongeren', 'grashoff', 'moorlag', 'jacobi', 'veldhoven',
    'kops', 'klever', 'madlener', 'dijck', 'lodders', 'neppérus',
    'ziengs', 'grinwis', 'koopmans', 'mulder', 'bontenbal', 'vedder',
    'geurts', 'plas', 'eppink', 'pierik', 'eerdmans', 'jansen',
    'haga', 'blanke', 'raan', 'teunissen', 'kröger', 'thijssen',
    'gabriëls', 'kostic', 'bushoff', 'beckerman', 'tjeerd', 'remco',
    'dijkstra', 'veltman', 'eijs', 'ham', 'mos', 'tony', 'bromet',
    'klaver', 'bouchallikh', 'ouchallikh', 'lee', 'smeulders',
    'bamenga', 'paternotte', 'hagen', 'boucke', 'podt', 'schouw',
    'hachchi', 'verhoeven', 'kaya', 'boswijk', 'werf', 'sienot',
    'martels', 'harbers', 'koerhuis', 'wijngaarden', 'peter',
    'graaf', 'graus', 'wilders', 'bemmel', 'heutink', 'blank',
    'arts', 'exirel', 'omtzigt', 'bisschop', 'flach', 'vestering',
    'esch', 'nijboer', 'jan', 'vos', 'albert', 'cegerek', 'smaling',
    'eigenlijk', 'vraag', 'smaak', 'schrijver', 'cdafractie',
    'pvdafractie', 'afrikaans',
}

ALL_STOPS = DUTCH_STOPS | EXTRA_STOPS

CONTENT_POS = {'NOUN', 'ADJ'}

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


def build_corpus(df: pd.DataFrame, group_col: str) -> dict:
    """Concatenate lemmatized texts per group into one document."""
    corpus = {}
    for key, group in df.groupby(group_col):
        texts = group['text_lemma'].dropna().tolist()
        corpus[key] = ' '.join(texts)
    return corpus


def clean_motion_text(text: str) -> str:
    """
    Strip parliamentary motion headers and closing formulas to remove
    member names and boilerplate that would otherwise dominate TF-IDF.
    """
    if not isinstance(text, str):
        return ''
    text = re.sub(
        r'(?i)^.*?gehoord\s+de\s+beraadslaging[,\s]*',
        '', text, flags=re.DOTALL)
    text = re.sub(
        r'(?i)en\s+gaat\s+over\s+tot\s+de\s+orde\s+van\s+de\s+dag\.?',
        '', text)
    return ' '.join(text.split()).strip()


def lemmatize_content_words(text: str) -> str:
    """Lemmatize and retain only content nouns and adjectives."""
    if not isinstance(text, str) or not text.strip():
        return ''
    doc = nlp(text[:8000])
    return ' '.join(
        t.lemma_.lower() for t in doc
        if t.pos_ in CONTENT_POS
        and not t.is_stop
        and len(t.lemma_) > 4
        and t.lemma_.lower() not in ALL_STOPS
        and t.lemma_.isalpha()
    )


def load_and_lemmatize(source: str, all_ones_ids: dict) -> pd.DataFrame:
    """Load scores CSV, filter all-ones and non-target parties, lemmatize."""
    path, id_col = SCORE_PATHS[source]
    df = pd.read_csv(path)

    # normalise party column to lowercase
    if source == 'motions':
        df['party'] = df['fractions'].str.split(';').str[0].str.strip().str.lower()
    elif source == 'speeches':
        df['party'] = df['party_name'].str.lower()
    else:
        df['party'] = df['party'].str.lower()

    # filter all-ones
    df[id_col] = df[id_col].astype(str)
    before = len(df)
    df = df[~df[id_col].isin(all_ones_ids[source])].copy()
    print(f'  {source}: {before} -> {len(df)} rows after all-ones exclusion')

    # filter target parties
    df = df[df['party'].isin(TARGET)].copy()
    print(f'  {source}: {len(df)} rows after party filter')

    df['text_clean'] = (df['text'].apply(clean_motion_text)
                        if source == 'motions' else df['text'])
    df['text_lemma'] = df['text_clean'].apply(lemmatize_content_words)
    df['source_type'] = source

    if 'year' not in df.columns:
        df['year'] = pd.to_datetime(df['date'], utc=True).dt.year

    return df[['party', 'year', 'text_lemma', 'source_type']].copy()


def compute_tfidf(corpus: dict, top_k: int = 20) -> pd.DataFrame:
    """
    Fit TF-IDF across all group documents and return top_k terms per group.
    """
    keys = list(corpus.keys())
    docs = [corpus[k] for k in keys]

    vectorizer = TfidfVectorizer(
        max_features=15000,
        min_df=2,
        ngram_range=(1, 2),
        stop_words=list(ALL_STOPS),
        sublinear_tf=True,
        token_pattern=r'\b[a-zA-Zàáâãäåæçèéêëìíîïðñòóôõöùúûüýþÿ]{4,}\b'
    )
    matrix   = vectorizer.fit_transform(docs)
    features = vectorizer.get_feature_names_out()

    rows = []
    for i, key in enumerate(keys):
        scores  = matrix[i].toarray().flatten()
        top_idx = scores.argsort()[::-1][:top_k]
        for rank, idx in enumerate(top_idx):
            rows.append({
                'group': TARGET_DISPLAY.get(key, key),
                'rank':  rank + 1,
                'term':  features[idx],
                'score': round(scores[idx], 4),
            })
    return pd.DataFrame(rows)


def plot_tfidf_grid(tfidf_df: pd.DataFrame, group_order: list,
                    colors: dict, title: str, filename: str,
                    top_k: int = 15):
    """Horizontal bar chart grid, one subplot per group."""
    groups  = [g for g in group_order if g in tfidf_df['group'].unique()]
    n_cols  = 3
    n_rows  = -(-len(groups) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(18, n_rows * 3.5),
                              constrained_layout=True)
    axes = axes.flatten()

    for i, group in enumerate(groups):
        ax    = axes[i]
        pdata = tfidf_df[tfidf_df['group'] == group].head(top_k)
        color = colors.get(group, '#555555')
        ax.barh(pdata['term'][::-1], pdata['score'][::-1],
                color=color, alpha=0.85)
        ax.set_title(group, fontweight='bold', color=color)
        ax.set_xlabel('TF-IDF score')
        ax.grid(True, alpha=0.3, axis='x')
        ax.tick_params(axis='y', labelsize=9)

    for j in range(len(groups), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.01)
    plt.savefig(FIG_DIR / filename, bbox_inches='tight')
    plt.close()
    print(f'Saved: {filename}')


def print_top_terms(tfidf_df: pd.DataFrame, label: str, top_k: int = 10):
    print(f'\n=== Top {top_k} distinctive terms ({label}) ===')
    for group in tfidf_df['group'].unique():
        pdata = tfidf_df[tfidf_df['group'] == group].head(top_k)
        terms = ', '.join(pdata['term'].tolist())
        print(f'  {group:<20} {terms}')


def main():
    print('Loading all-ones IDs...')
    all_ones_ids = load_all_ones_ids()

    print('Loading and lemmatizing sources...')
    dfs = []
    for source in ['motions', 'manifestos', 'speeches']:
        dfs.append(load_and_lemmatize(source, all_ones_ids))

    combined = pd.concat(dfs, ignore_index=True)
    combined['bloc'] = combined['party'].map(BLOC_MAP)
    print(f'Combined: {len(combined)} rows')

    # ── Party-level TF-IDF ────────────────────────────────────────────────────
    print('\nComputing party-level TF-IDF...')
    party_corpus = build_corpus(combined, 'party')
    party_tfidf  = compute_tfidf(party_corpus, top_k=20)

    party_tfidf.to_csv(OUT_DIR / 'tfidf_combined.csv', index=False)
    print_top_terms(party_tfidf, 'combined', top_k=10)
    plot_tfidf_grid(
        party_tfidf,
        group_order=list(TARGET_DISPLAY.values()),
        colors=PARTY_COLORS,
        title='Most Distinctive Terms per Party — All Text Types Combined',
        filename='tfidf_combined.pdf',
        top_k=15,
    )

    # ── Bloc-level contrastive TF-IDF ─────────────────────────────────────────
    print('\nComputing bloc-level contrastive TF-IDF...')
    bloc_df = combined[combined['bloc'].notna()].copy()

    vectorizer = TfidfVectorizer(
        max_features=15000, min_df=1,
        ngram_range=(1, 2), stop_words=list(ALL_STOPS),
        sublinear_tf=True,
        token_pattern=r'\b[a-zA-Zàáâãäåæçèéêëìíîïðñòóôõöùúûüýþÿ]{4,}\b'
    )

    blocs = ['Left', 'Center', 'Right']
    docs  = [' '.join(bloc_df[bloc_df['bloc'] == b]['text_lemma'].dropna())
             for b in blocs]
    matrix   = vectorizer.fit_transform(docs)
    features = vectorizer.get_feature_names_out()

    rows = []
    for i, bloc in enumerate(blocs):
        focal_mean  = matrix[i].toarray().flatten()
        others_mean = np.array([
            matrix[j].toarray().flatten()
            for j in range(len(blocs)) if j != i
        ]).mean(axis=0)
        diff    = focal_mean - others_mean
        top_idx = diff.argsort()[::-1][:20]
        for rank, idx in enumerate(top_idx):
            if diff[idx] > 0:
                rows.append({'group': bloc, 'rank': rank+1,
                             'term': features[idx],
                             'score': round(diff[idx], 4)})

    bloc_tfidf = pd.DataFrame(rows)
    bloc_tfidf.to_csv(OUT_DIR / 'tfidf_bloc.csv', index=False)
    print_top_terms(bloc_tfidf, 'blocs', top_k=10)
    plot_tfidf_grid(
        bloc_tfidf,
        group_order=['Left', 'Center', 'Right'],
        colors=BLOC_COLORS,
        title='Most Distinctive Terms per Ideological Bloc — Contrastive TF-IDF',
        filename='tfidf_bloc.pdf',
        top_k=15,
    )


if __name__ == '__main__':
    main()