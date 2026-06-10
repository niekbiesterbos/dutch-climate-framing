"""
Micro Actor/Patient Extraction via spaCy
===================================================
Extracts actor and patient spans from LOME Statement and Cogitation
clauses using Dutch dependency parsing.

Input:  results/motions/micro_scores/lome_roles.csv
Output: results/motions/micro_scores/spacy_spans.csv
"""

import pandas as pd
import spacy
from pathlib import Path


INPUT_CSV  = Path("results/motions/micro_scores/lome_roles.csv")
OUTPUT_CSV = Path("results/motions/micro_scores/spacy_spans.csv")

RELEVANT_FRAMES = {"Statement", "Cogitation"}
RELEVANT_ROLES  = {"Message", "Topic"}

ACTOR_DEPS   = {"nsubj", "nsubj:pass", "agent"}
PATIENT_DEPS = {"obj", "iobj", "obl", "nsubj:pass"}

# Noise: stopwords, procedural terms, abstract nouns
NOISE_EXACT = {
    "motie", "kamer", "beraadslaging", "de kamer", "de beraadslaging",
    "de regering", "het kabinet", "nederland", "tweede kamer", "parlement",
    "regering", "kabinet", "minister", "nederland", "nederlandse",
    "Europese",  
}

NOISE_LEMMAS = {
    "feit", "probleem", "deel", "gebruik", "belang", "reactie", "beeld",
    "mate", "geval", "moment", "manier", "wijze", "punt", "reden", "vraag",
    "mogelijkheid", "kans", "gevolg", "effect", "basis", "kader", "lijst",
    "termijn", "periode", "jaar", "tijd", "dag", "week", "maand",
    "ding", "iets", "niets", "alles", "aanleiding", "achtergrond",
    "context", "situatie", "stand", "sprake", "aantal", "doel",
    "transitie", "energietransitie", "kost", "rekening", "risico",
    "onderzoek", "alternatief", "investering", "energie", "besparing",
    "druk", "uitvoering", "rol", "bijdrage", "praktijk", "toekomst",
    "verwachting", "lijn", "eind", "ontwikkeling", "kwaliteit",
    "doelstelling", "miljoen", "miljard", "maatregel", "ambitie", "plan",
    "bouw", "toelating", "ruimte", "akkoord", "coalitieakkoord",
    "regeerakkoord", "klimaatakkoord", "energiebesparing", "voorstel",
    "duidelijkheid", "keer", "afspraak", "inzet", "wet", "regeling",
    "prioriteit", "geld", "invoering", "energieakkoord", "raad",
}

def clean_span(text: str) -> str:
    """Strip leading conjunctions and determiners like 'dat', 'de', 'het'."""
    text = text.strip(" ,;.")
    for prefix in ("dat ", "die ", "wat ", "welke ", "waarbij "):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
    return text.strip()


def is_noise(token, text: str) -> bool:
    """Return True if this span should be filtered out."""
    if text.lower().strip(" ,;.") in NOISE_EXACT:
        return True
    if token.lemma_.lower() in NOISE_LEMMAS:
        return True
    if len(text.strip()) < 4:
        return True
    return False


def extract_spans(text: str, nlp) -> tuple[list[str], list[str]]:
    """
    Extract actor and patient noun chunks from a clause via dependency parsing.
    Returns (actors, patients) as lists of cleaned strings.
    """
    if not isinstance(text, str) or len(text.strip()) < 5:
        return [], []

    doc = nlp(text[:500])
    actors, patients = [], []

    for token in doc:
        if token.pos_ not in {"NOUN", "PROPN"}:
            continue

        # Find the noun chunk this token heads
        chunk_text = token.text
        for nc in doc.noun_chunks:
            if nc.root == token:
                chunk_text = nc.text
                break

        chunk_text = clean_span(chunk_text)

        if is_noise(token, chunk_text):
            continue

        if token.dep_ in ACTOR_DEPS:
            actors.append(chunk_text)
        elif token.dep_ in PATIENT_DEPS:
            patients.append(chunk_text)

    return actors, patients


def main():
    print("Loading spaCy model...")
    nlp = spacy.load("nl_core_news_lg")

    print(f"Loading LOME roles: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    mask = (
        df["frame"].isin(RELEVANT_FRAMES) &
        df["role"].isin(RELEVANT_ROLES) &
        df["filler"].notna()
    )
    clauses = df[mask].drop_duplicates(subset=["motie_id", "frame", "role", "filler"]).copy()
    print(f"Clauses to process: {len(clauses)}")

    actors_list, patients_list = [], []
    for i, text in enumerate(clauses["filler"]):
        actors, patients = extract_spans(text, nlp)
        actors_list.append("|".join(actors) if actors else "")
        patients_list.append("|".join(patients) if patients else "")
        if i % 5000 == 0:
            print(f"  {i}/{len(clauses)}")

    clauses["actor_spans"]   = actors_list
    clauses["patient_spans"] = patients_list

    clauses = clauses[
        (clauses["actor_spans"] != "") | (clauses["patient_spans"] != "")
    ].copy()

    print(f"\nClauses with spans: {len(clauses)}")
    print(f"With actor:   {(clauses['actor_spans'] != '').sum()}")
    print(f"With patient: {(clauses['patient_spans'] != '').sum()}")

    print("\nTop 30 actor spans:")
    actors_exploded = clauses["actor_spans"].str.split("|").explode()
    actors_exploded = actors_exploded[actors_exploded.str.len() > 3]
    print(actors_exploded.value_counts().head(30).to_string())

    print("\nTop 30 patient spans:")
    patients_exploded = clauses["patient_spans"].str.split("|").explode()
    patients_exploded = patients_exploded[patients_exploded.str.len() > 3]
    print(patients_exploded.value_counts().head(30).to_string())

    clauses.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()