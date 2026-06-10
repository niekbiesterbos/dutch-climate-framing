import re
import sys
import pandas as pd
from pathlib import Path

# =============================================================================
# Config
# =============================================================================

MODE = sys.argv[1] if len(sys.argv) > 1 else "motions"
assert MODE in {"motions", "manifestos", "speeches"}, f"Unknown mode: {MODE}"

BASE   = Path(os.environ.get("LOME_BASE", "/scratch/s4744497"))
THESIS = BASE / "thesis"

CONFIGS = {
    "motions": {
        "lome_dir":    BASE / "lome/motions",
        "file_glob":   "motion_*.txt",
        "meta_path":   THESIS / "results/motions/climate_motions.csv",
        "output_path": THESIS / "results/motions/micro_scores/lome_roles.csv",
        "meta_cols":   ["id", "title", "date", "fractions", "predicted_topic"],
        "id_prefix":   "motion_",
    },
    "manifestos": {
        "lome_dir":    BASE / "lome/manifestos/output",
        "file_glob":   "manifesto_*.txt",
        "meta_path":   THESIS / "data/manifestos/relevant_phrases.csv",
        "output_path": THESIS / "results/manifestos/micro_scores/lome_roles.csv",
        "meta_cols":   ["party", "year", "text"],
        "id_prefix":   "manifesto_",
    },
    "speeches": {
        "lome_dir":    BASE / "lome/speeches/output",
        "file_glob":   "speech_*.txt",
        "meta_path":   THESIS / "results/speeches/utterances.csv",
        "output_path": THESIS / "results/speeches/micro_scores/lome_roles.csv",
        "meta_cols":   ["utterance_id", "party", "date", "speaker"],
        "id_prefix":   "speech_",
    },
}

cfg = CONFIGS[MODE]
cfg["output_path"].parent.mkdir(parents=True, exist_ok=True)

# Frames relevant for agency and framing analysis
AGENCY_FRAMES = {
    "Attempt_suasion", "Request", "Statement", "Cogitation",
    "Causation", "Compliance", "Leadership", "Being_obligated",
    "Intentionally_act", "Activity_stop", "Bringing",
    "Cause_expansion", "Preventing_or_letting",
    "Ratification", "Removing", "Placing", "Scrutiny",
}

# Roles that carry actor or patient information
AGENCY_ROLES = {
    "Speaker", "Addressee", "Content", "Message", "Topic",
    "Agent", "Actor", "Leader", "Cognizer", "Cause",
    "Theme", "Affected", "Victim", "Goal", "Item",
}


# =============================================================================
# Parsing
# =============================================================================

def parse_lome_file(path: Path, doc_id: str) -> list[dict]:
    rows = []
    current_frame = None
    current_trigger = None

    for line in path.read_text(encoding="utf-8").split("\n"):
        frame_match = re.match(r"^Frame: (\S+) \| Trigger: '(.+)'$", line.strip())
        if frame_match:
            current_frame   = frame_match.group(1)
            current_trigger = frame_match.group(2)
            continue

        role_match = re.match(r"^\s+└─ (\w+): '(.+)' \[(.+)\]$", line)
        if role_match and current_frame in AGENCY_FRAMES:
            role        = role_match.group(1)
            filler      = role_match.group(2)
            entity_type = role_match.group(3)
            if role in AGENCY_ROLES:
                rows.append({
                    "doc_id":      doc_id,
                    "frame":       current_frame,
                    "trigger":     current_trigger,
                    "role":        role,
                    "filler":      filler,
                    "entity_type": entity_type,
                    "source":      MODE,
                })

    return rows


# =============================================================================
# Metadata loading
# =============================================================================

def load_metadata(cfg: dict) -> dict:
    meta = pd.read_csv(cfg["meta_path"], usecols=cfg["meta_cols"])

    if MODE == "motions":
        meta["id"] = meta["id"].astype(str).str.strip()
        return meta.set_index("id").to_dict("index")

    if MODE == "manifestos":
        meta = meta.reset_index()
        meta["file_idx"] = meta["index"].astype(str).str.zfill(4)
        return meta.set_index("file_idx").to_dict("index")

    if MODE == "speeches":
        meta["utterance_id"] = meta["utterance_id"].astype(str).str.strip()
        return meta.set_index("utterance_id").to_dict("index")


def attach_metadata(row: dict, stem: str, meta_lookup: dict) -> dict:
    if MODE == "motions":
        doc_id = row["doc_id"]
        m = meta_lookup.get(doc_id, {})
        return {**row,
                "motie_id":        doc_id,
                "title":           m.get("title", ""),
                "date":            m.get("date", ""),
                "fractions":       m.get("fractions", ""),
                "predicted_topic": m.get("predicted_topic", "")}

    if MODE == "manifestos":
        file_idx = stem.split("_")[1]
        m = meta_lookup.get(file_idx, {})
        return {**row,
                "manifesto_id": stem,
                "party":        m.get("party", ""),
                "year":         m.get("year", ""),
                "phrase_text":  m.get("text", "")}

    if MODE == "speeches":
        doc_id = row["doc_id"]
        m = meta_lookup.get(doc_id, {})
        return {**row,
                "utterance_id": doc_id,
                "party":        m.get("party", ""),
                "date":         m.get("date", ""),
                "speaker":      m.get("speaker", "")}


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"Mode: {MODE}")

    meta_lookup = load_metadata(cfg)

    files = sorted(cfg["lome_dir"].glob(cfg["file_glob"]))
    print(f"Processing {len(files)} files...")

    all_rows = []
    for i, path in enumerate(files):
        stem   = path.stem
        doc_id = stem.replace(cfg["id_prefix"], "")
        rows   = parse_lome_file(path, doc_id)
        rows   = [attach_metadata(row, stem, meta_lookup) for row in rows]
        all_rows.extend(rows)

        if i % 500 == 0:
            print(f"  {i}/{len(files)}")

    df = pd.DataFrame(all_rows)
    df.to_csv(cfg["output_path"], index=False)

    print(f"\nSaved {len(df)} rows to {cfg['output_path']}")
    print(df["frame"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()