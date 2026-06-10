"""
ParlaMint Speeches to CSV
====================================
Parses Okky's ParlaMint-NL JSONL files and aggregates sentences
into utterances (one row per speaker turn per debate).

Input:  data/speeches/ParlaMint-Okky/ParlaMint-NL-*.jsonl
Output: results/speeches/utterances.csv
"""

import json
import re
import pandas as pd
from pathlib import Path


INPUT_DIR  = Path("data/speeches/ParlaMint-Okky")
OUTPUT_CSV = Path("results/speeches/utterances.csv")

PARTY_MAP = {
    "Volkspartij voor Vrijheid en Democratie":          "VVD",
    "Partij voor de Vrijheid":                          "PVV",
    "Christen-Democratisch Appèl":                      "CDA",
    "Democraten 66":                                    "D66",
    "GroenLinks":                                       "GroenLinks",
    "Partij van de Arbeid":                             "PvdA",
    "Socialistische Partij":                            "SP",
    "ChristenUnie":                                     "ChristenUnie",
    "Partij voor de Dieren":                            "PvdD",
    "Staatkundig Gereformeerde Partij":                 "SGP",
    "Forum voor Democratie":                            "FvD",
    "BoerBurgerBeweging":                               "BBB",
    "Volt Nederland":                                   "Volt",
    "Juiste Antwoord 2021":                             "JA21",
    "Bij1":                                             "BIJ1",
    "Nieuw Sociaal Contract":                           "NSC",
    "GroenLinks-PvdA":                                  "GroenLinks-PvdA",
    "50PLUS":                                           "50PLUS",
    "DENK":                                             "DENK",
}


def normalize_party(party_name: str) -> str:
    """Map full Dutch party name to standard abbreviation."""
    return PARTY_MAP.get(party_name, party_name)


def parse_jsonl(jsonl_file: Path, year: int) -> list[dict]:
    """Parse a single JSONL file into a list of sentence-level rows."""
    rows = []
    with open(jsonl_file, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", d["sent_id"])
            rows.append({
                "sent_id":     d["sent_id"],
                "doc_id":      d["doc_id"],
                "year":        year,
                "date":        date_match.group(1) if date_match else "",
                "text":        d["text"],
                "party_name":  normalize_party(d["party_name"]),
                "orientation": d.get("weak_grouped_party_orientation", ""),
                "utt_topic":   "|".join(d.get("utt_topic", [])),
                "utt_keyword": "|".join(d.get("utt_keyword", [])),
                "frame_list":  "|".join(d.get("frame_list", [])),
            })
    return rows


def load_all_sentences(input_dir: Path) -> pd.DataFrame:
    """Load and concatenate all JSONL files into a sentence-level DataFrame."""
    all_rows = []
    for jsonl_file in sorted(input_dir.glob("ParlaMint-NL-*.jsonl")):
        year = int(re.search(r"(\d{4})", jsonl_file.stem).group(1))
        print(f"  Loading {jsonl_file.name}...")
        all_rows.extend(parse_jsonl(jsonl_file, year))
    df = pd.DataFrame(all_rows)
    print(f"Total sentences loaded: {len(df)}")
    return df


def aggregate_to_utterances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentence-level rows to utterance level.
    One utterance = one speaker turn in one debate (doc_id).
    Text is joined, frame_list is deduplicated.
    Filters out utterances with fewer than 3 sentences.
    """
    utterances = df.groupby("doc_id").agg(
        text=("text", " ".join),
        year=("year", "first"),
        date=("date", "first"),
        party_name=("party_name", "first"),
        orientation=("orientation", "first"),
        utt_topic=("utt_topic", "first"),
        utt_keyword=("utt_keyword", "first"),
        frame_list=("frame_list", lambda x: "|".join(dict.fromkeys("|".join(x).split("|")))),
        n_sentences=("text", "count"),
    ).reset_index()

    before = len(utterances)
    utterances = utterances[utterances["n_sentences"] >= 3].reset_index(drop=True)
    print(f"Filtered {before - len(utterances)} utterances with < 3 sentences.")

    return utterances


def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print("Loading sentences...")
    sentences = load_all_sentences(INPUT_DIR)

    print("\nAggregating to utterances...")
    utterances = aggregate_to_utterances(sentences)

    print(f"Total utterances: {len(utterances)}")
    print("\nUtterances per party per year:")
    print(utterances.groupby(["year", "party_name"]).size().unstack(fill_value=0).to_string())

    utterances.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()