#!/usr/bin/env python3
"""
Text normalization pipeline for Dutch parliamentary motions
Optimized for RoBERTa-based classification.

Purpose:
    - Cleans OCR and boilerplate artifacts while keeping natural sentence structure.
    - Preserves stopwords, grammar, punctuation, and casing (RoBERTa benefits from them).
    - Removes all header content before "gehoord de beraadslaging;".
    - Can optionally combine title + text for classifier training input.

Usage:
    python normalize_motions_roberta.py
"""

import re
import unicodedata
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

# ---------------------------------------------------------------------
# --- Regex patterns for cleaning ---
# ---------------------------------------------------------------------

RE_URL = re.compile(r"https?://\S+")
RE_NUMERIC = re.compile(r"\b\d+\b")
RE_MONTH = re.compile(
    r"\b(januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december)\b",
    re.IGNORECASE,
)
RE_AFTER_END = re.compile(
    r"en gaat over tot de orde van de dag\..*", re.IGNORECASE | re.DOTALL
)
RE_FOOTER_LINES = re.compile(
    r"(Tweede Kamer der Staten-Generaal|Vergaderjaar|Sdu Uitgevers|’s[-’]Gravenhage|KST\d+|ISSN|tkkst|XII|XVI)",
    re.IGNORECASE,
)
RE_MOTIE_HEADER = re.compile(r"(?im)^\s*nr\.\s*.*$")
RE_MULTISPACE = re.compile(r"\s{2,}")

# Remove extremely repetitive procedural boilerplate but preserve general sentence meaning
RE_REMOVE_TERMS_LIGHT = re.compile(
    r"(?<!\w)(?:kst|vergaderjaar|tweede kamer|eerste kamer|staten-?generaal|bijlage|bijlagen|nota|"
    r"commissie|stuk|artikel|regeling|vergadering|bijlage|rijksbegroting|"
    r"Sdu Uitgevers|’s[-’]Gravenhage)(?!\w)",
    re.IGNORECASE,
)

RE_NR = re.compile(r"(?<!\w)n\s*\.?\s*r\s*\.?(?!\w)", re.IGNORECASE)

# Match everything before "gehoord de beraadslaging;"
RE_BEFORE_DEBATE = re.compile(
    r"^.*?(gehoord de beraadslaging\s*[;,:])", re.IGNORECASE | re.DOTALL
)

# ---------------------------------------------------------------------
# --- Core cleaning helpers ---
# ---------------------------------------------------------------------


def remove_control_bytes(text: str) -> str:
    """Remove ASCII control bytes (0x00–0x1F, 0x7F–0x9F) from OCR text."""
    data = text.encode("utf-8", "ignore")
    filtered = bytes(b for b in data if b >= 0x20 or b in (0x09, 0x0A, 0x0D))
    return filtered.decode("utf-8", "ignore")


def replace_controls_with_space(text: str) -> str:
    """Replace Unicode control and formatting characters with a space."""
    return ''.join(' ' if unicodedata.category(ch).startswith('C') else ch for ch in text)


# ---------------------------------------------------------------------
# --- Cleaning for RoBERTa ---
# ---------------------------------------------------------------------

def clean_for_roberta(text: str) -> str:
    """Light cleaning for RoBERTa classifier input (keeps grammar, punctuation, and stopwords)."""
    if not isinstance(text, str):
        return ""

    # Normalize Unicode and remove hidden control characters
    text = remove_control_bytes(text)
    text = unicodedata.normalize("NFKC", text)
    text = replace_controls_with_space(text)
    text = re.sub(r"[\u00AD\u200B\u200C\u200D\u2060\uFEFF]", "", text)

    # Remove everything before "gehoord de beraadslaging;"
    match = RE_BEFORE_DEBATE.search(text)
    if match:
        text = text[match.start(1):]  # start from "gehoord de beraadslaging;"

    # Remove clear OCR or structural noise
    text = RE_AFTER_END.sub(" ", text)
    text = RE_FOOTER_LINES.sub(" ", text)
    text = RE_MOTIE_HEADER.sub(" ", text)
    text = RE_NR.sub(" ", text)
    text = RE_URL.sub(" ", text)

    # Remove obvious metadata or repetitive procedural junk
    text = RE_REMOVE_TERMS_LIGHT.sub(" ", text)

    # Collapse whitespace but keep punctuation
    text = RE_MULTISPACE.sub(" ", text).strip()

    return text
