import pandas as pd
import spacy
from pathlib import Path

nlp = spacy.load("nl_core_news_lg")

df = pd.read_csv("results/motions/micro_scores/lome_roles.csv")
content = df[
    (df["frame"] == "Attempt_suasion") &
    (df["role"] == "Content")
].drop_duplicates(subset=["motie_id", "filler"]).copy()

content["year"] = pd.to_datetime(content["date"], utc=True, errors="coerce").dt.year
content["party"] = content["fractions"].str.split(";").str[0].str.strip()

PATIENT_TAXONOMY = {
    # Nature/Environment
    "natuur": "Nature", "dier": "Nature", "dieren": "Nature", "water": "Nature",
    "gebied": "Nature", "stof": "Nature", "uitstoot": "Nature", "gaswinning": "Nature",
    "waddenzee": "Nature", "noordzee": "Nature", "biodiversiteit": "Nature",
    "ecosysteem": "Nature", "soort": "Nature", "habitat": "Nature",
    "veenweide": "Nature", "stikstof": "Nature", "emissie": "Nature",

    # Industry
    "bedrijf": "Industry", "bedrijven": "Industry", "industrie": "Industry",
    "sector": "Industry", "product": "Industry", "vergunning": "Industry",
    "regelgeving": "Industry", "subsidie": "Industry", "mkb": "Industry",
    "onderneming": "Industry", "ondernemer": "Industry", "werkgever": "Industry",

    # Agriculture
    "landbouw": "Agriculture", "boer": "Agriculture", "boeren": "Agriculture",
    "gewas": "Agriculture", "veeteelt": "Agriculture", "veehouder": "Agriculture",
    "agrariër": "Agriculture", "tuinder": "Agriculture", "visser": "Agriculture",
    "visserij": "Agriculture",

    # Citizens
    "woning": "Citizens", "woningen": "Citizens", "kost": "Citizens",
    "zorg": "Citizens", "burger": "Citizens", "burgers": "Citizens",
    "huishouden": "Citizens", "gezin": "Citizens", "huurder": "Citizens",
    "consument": "Citizens", "inwoner": "Citizens", "energierekening": "Citizens",

    # Energy
    "energie": "Energy", "verduurzaming": "Energy", "energietransitie": "Energy",
    "netbeheerder": "Energy", "energiebedrijf": "Energy", "windpark": "Energy",
    "zonnepaneel": "Energy", "warmtepomp": "Energy", "elektriciteit": "Energy",
    "waterstof": "Energy", "co2": "Energy", "klimaat": "Energy",
}

# Wat we weggooien — policy instrumenten, geen patients
SKIP = {
    "maatregel", "onderzoek", "plan", "aanpak", "voorstel", "afspraak",
    "kader", "besluit", "beleid", "doel", "doelstelling", "overleg",
    "gebruik", "middel", "vorm", "wijze", "basis", "geval", "mogelijkheid",
    "effect", "gevolg", "bijdrage", "ontwikkeling", "project", "gesprek",
    "termijn", "kaart", "ruimte", "land", "nederland", "jaar", "kamer",
    "regering", "kabinet", "overheid", "minister"
}

def extract_patient(text):
    if not isinstance(text, str):
        return None
    doc = nlp(text[:400])
    # Loop over tokens, zoek lemma in taxonomy
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma in SKIP:
            continue
        if lemma in PATIENT_TAXONOMY:
            return PATIENT_TAXONOMY[lemma]
    return None

content["patient"] = content["filler"].apply(extract_patient)
out = content[content["patient"].notna()].copy()

print(f"Content rows met patient: {len(out)} / {len(content)}")
print(f"Coverage: {len(out)/len(content)*100:.1f}%")

print("\n=== Patient verdeling ===")
print(out["patient"].value_counts())

print("\n=== Patient per partij ===")
pivot = out.groupby(["party", "patient"]).size().unstack(fill_value=0)
print(pivot)

print("\n=== Patient per jaar ===")
yearly = out.groupby(["year", "patient"]).size().unstack(fill_value=0)
print(yearly)

# Aggregeer naar motieniveau — één rij per motie
motie_out = out.groupby("motie_id").agg(
    party=("party", "first"),
    year=("year", "first"),
    title=("title", "first"),
    patient=("patient", lambda x: x.value_counts().index[0])  # meest voorkomende
).reset_index()

motie_out.to_csv(
    "results/motions/micro_scores/EXP_18_patient_per_motie.csv",
    index=False
)
print(f"\nSaved. {len(motie_out)} moties met patient.")
print(motie_out["patient"].value_counts())