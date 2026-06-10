"""
Reconstruct EXP_18 results from checkpoint without rerunning the model.
"""

import json
import pickle
import re
import pandas as pd
from pathlib import Path

INPUT_CSV  = Path("results/motions/micro_scores/lome_roles.csv")
OUTPUT_CSV = Path("results/motions/micro_scores/EXP_18_actor_patient.csv")
CKPT_PATH  = Path("results/motions/micro_scores/ckpt_v2_qwen2.5-32b.pkl")

ACTOR_ROLES   = {"Speaker", "Cognizer", "Agent", "Leader", "Cause", "Actor"}
PATIENT_ROLES = {"Addressee", "Theme", "Patient", "Goal", "Affected", "Victim"}
CLAUSE_ROLES  = {"Message", "Topic", "Content"}

INCLUDE_FRAMES = {
    "Statement", "Cogitation",
    "Leadership", "Attempt_suasion", "Scrutiny",
    "Intentionally_act", "Activity_stop", "Bringing",
    "Cause_expansion", "Preventing_or_letting",
    "Ratification", "Removing", "Placing",
}

TAXONOMY = [
    "Government", "Industry", "Agriculture", "Citizens", "Nature",
    "Science", "International", "Energy", "CivilSociety", "Other",
]

import os
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

print("Loading checkpoint...")
with open(CKPT_PATH, "rb") as f:
    ckpt = pickle.load(f)
print(f"Checkpoint entries: {len(ckpt)}")

print("Loading LOME roles...")
df = pd.read_csv(INPUT_CSV)
df = df[df["frame"].isin(INCLUDE_FRAMES)].copy()
df["party"] = df["fractions"].str.split(";").str[0].str.strip()
df["year"]  = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.year
df["is_actor_role"]   = df["role"].isin(ACTOR_ROLES)
df["is_patient_role"] = df["role"].isin(PATIENT_ROLES)
df["is_clause_role"]  = df["role"].isin(CLAUSE_ROLES)

direct_rows = df[df["is_actor_role"] | df["is_patient_role"]].copy()
clause_rows = df[df["is_clause_role"]].copy()

print(f"Direct rows: {len(direct_rows)} | Clause rows: {len(clause_rows)}")

results = []

# Reconstruct direct spans
span_items = []
for i, row in direct_rows.iterrows():
    role = "actor" if row["is_actor_role"] else "patient"
    key  = f"span_{i}_{role}"
    span_items.append((key, row["filler"], role, i, row))

for key, span, role, i, row in span_items:
    if key not in ckpt or ckpt[key] is None:
        continue
    cat = ckpt[key]["category"]
    results.append({
        "motie_id":         row["motie_id"],
        "frame":            row["frame"],
        "trigger":          row["trigger"],
        "source":           "direct",
        "actor_span":       span if role == "actor" else None,
        "actor_category":   cat  if role == "actor" else None,
        "patient_span":     span if role == "patient" else None,
        "patient_category": cat  if role == "patient" else None,
        "party":            row["party"],
        "year":             row["year"],
        "date":             row["date"],
        "fractions":        row["fractions"],
    })

# Reconstruct clause results
clause_items = []
for i, row in clause_rows.iterrows():
    key = f"clause_{i}"
    clause_items.append((key, row["filler"], row["frame"], row["trigger"], i, row))

for key, filler, frame, trigger, i, row in clause_items:
    if key not in ckpt or ckpt[key] is None:
        continue
    parsed = ckpt[key]
    results.append({
        "motie_id":         row["motie_id"],
        "frame":            row["frame"],
        "trigger":          row["trigger"],
        "source":           "llm_clause",
        "actor_span":       parsed["actor_span"],
        "actor_category":   parsed["actor_category"],
        "patient_span":     parsed["patient_span"],
        "patient_category": parsed["patient_category"],
        "party":            row["party"],
        "year":             row["year"],
        "date":             row["date"],
        "fractions":        row["fractions"],
    })

out = pd.DataFrame(results)
out = out[out["actor_category"].notna() | out["patient_category"].notna()].copy()

out.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV} ({len(out)} rows)")

print("\n=== Actor distribution ===")
print(out["actor_category"].value_counts().to_string())

print("\n=== Patient distribution ===")
print(out["patient_category"].value_counts().to_string())

print("\n=== Patient per party ===")
target = ["GroenLinks","PvdA","D66","CDA","VVD","PVV","FVD","BBB","PvdD"]
sub = out[out["party"].isin(target)]
print(sub.groupby(["party","patient_category"]).size().unstack(fill_value=0).to_string())