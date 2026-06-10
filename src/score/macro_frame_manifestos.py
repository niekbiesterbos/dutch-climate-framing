"""
Macro-Frame Annotation on Party Manifestos
====================================================
Scores climate phrases extracted from party manifestos (EXP_16)
on 7 macro-frames using a Likert 1-5 schema.

Input:  data/manifestos/relevant_phrases.csv
Output: results/manifestos/macro_scores/{MODEL_SHORT}.csv

Models: Qwen2.5-14B-Instruct, Qwen2.5-32B-Instruct
"""

import os
import gc
import shutil
import re
import time
import json
import pickle
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


USER         = "s4744497"
SCRATCH_CACHE = f"/scratch/{USER}/hf_cache"
os.environ["HF_HOME"] = SCRATCH_CACHE
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

MODELS = [
    ("google/gemma-3-12b-it", "gemma-3-12b-it"),
    ("google/gemma-3-27b-it", "gemma-3-27b-it"),
]

DATA_PATH      = "data/manifestos/relevant_phrases.csv"
TEXT_COL       = "text"
BATCH_SIZE     = 32
MAX_NEW_TOKENS = 8
OUT_DIR        = Path("results/manifestos/macro_scores")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG_PATH   = OUT_DIR / "run_log.jsonl"

LIKERT_DEFINITIONS = (
    "1 = Absent    -- No recognizable signal of this frame.\n"
    "2 = Marginal  -- A passing reference, but the frame does not structure the argument.\n"
    "3 = Present   -- The frame is clearly present but not the central organizing logic.\n"
    "4 = Prominent -- The frame is a primary lens; multiple sentences organized around it.\n"
    "5 = Dominant  -- The frame fully structures the text's argument."
)

FRAMES = {
    "economic": {
        "label": "F1 - Economic",
        "decision_rule": (
            "Ask: is the central justification in this text economic? "
            "If the text argues that climate policy is good or bad because of what it costs or earns, score high. "
            "If economic terms appear as context for a regulatory request, score low."
        ),
        "anchors": {
            1: "No mention of costs, jobs, investment, innovation, competitiveness, or economic consequences of climate policy.",
            2: "An economic term appears once as context or background but does not justify the argument.",
            3: "Economic consequences are explicitly used to support or oppose the policy across at least two sentences, but another frame is primary.",
            4: "The economic logic is one of two central pillars. Multiple sentences organized around economic costs, competitiveness, or opportunities.",
            5: "The entire argument is structured around economic logic. Every reason given is economic. No other frame plays a meaningful role.",
        },
        "examples": {
            1: "Parliament requests the government to designate additional Natura 2000 areas. -- No economic dimension.",
            2: "The text notes the energy transition will involve transition costs, then focuses on a renewable energy roadmap. -- Cost mentioned once, not used as argument.",
            3: "The text argues the subsidy scheme will create 5,000 jobs and stimulate clean-tech innovation. -- Economic benefit argued but core request is regulatory.",
            4: "The text argues ETS reform will cost industry EUR 2 billion and threaten 12,000 jobs. -- Economic framing dominates alongside one other frame.",
            5: "The text argues climate levies destroy purchasing power, make industry uncompetitive, and must be replaced by market instruments. -- Pure economic reasoning.",
        },
        "llm_question": "To what extent does this text frame climate policy through the logic of economic consequences -- costs, competitiveness, jobs, or innovation opportunities -- as the primary reason to act or not act?",
    },
    "moral": {
        "label": "F2 - Moral & Ethical",
        "decision_rule": (
            "Ask: does the text explicitly state that someone has a moral obligation or that the situation is unjust? "
            "Implied concern for nature or health is not moral framing. "
            "Fairness between nations (rich vs poor, historical emitters vs vulnerable nations) is Moral."
        ),
        "anchors": {
            1: "No moral vocabulary. No duty, obligation, fairness, justice, responsibility, or rights in relation to climate.",
            2: "A single moral phrase appears but is not elaborated or used as a justification.",
            3: "Moral reasoning is explicitly used as one justification across at least two sentences, but other frames are equally prominent.",
            4: "Moral framing is the primary lens. The text repeatedly names obligations or injustices as the main reasons for action.",
            5: "The entire argument is organized around moral obligation. Every clause names a duty, right, or injustice.",
        },
        "examples": {
            1: "The text requests an investigation into hydrogen storage capacity. -- Purely technical; no moral dimension.",
            2: "The text briefly states the Netherlands has a responsibility to lead on climate, then focuses on ETS reform. -- Responsibility named once, dropped immediately.",
            3: "The text argues the Netherlands bears a special obligation to lead and future generations must not inherit a destabilized climate. -- Moral argument present alongside governance framing.",
            4: "The text argues repeatedly that wealthy nations have a historical debt to the Global South and inaction violates intergenerational justice. -- Moral reasoning central throughout.",
            5: "The text argues we owe it to our children, to developing nations, and to future generations. Acting is a moral imperative that overrides economic considerations. -- Pure moral logic.",
        },
        "llm_question": "To what extent does this text ground its climate argument in explicit moral duty, fairness, justice, or responsibility -- naming who owes what to whom?",
    },
    "scientific": {
        "label": "F3 - Scientific",
        "decision_rule": (
            "Ask: is evidence being used as a reason to act, or just as decoration? "
            "A year mentioned as a target deadline is not scientific framing. "
            "A report cited to show that current policy is insufficient is scientific framing."
        ),
        "anchors": {
            1: "No empirical data, research citations, or quantitative claims.",
            2: "A single number or factual reference appears but is not used as an argument.",
            3: "One or two empirical claims are explicitly used as justification, but the text is not organized primarily around evidence.",
            4: "Multiple empirical sources or quantitative claims are cited and used as primary justifications.",
            5: "The entire argument is structured around empirical evidence. Every claim is grounded in cited research or data.",
        },
        "examples": {
            1: "The text requests the government to promote sustainable agriculture. -- No data or evidence.",
            2: "The text states CO2 emissions must fall and references the 2030 target. -- A target year appears but no evidence cited.",
            3: "The text cites PBL findings that current policy is insufficient to meet 2030 targets. -- One empirical source cited and used as argument.",
            4: "The text cites IPCC thresholds, PBL policy gap analysis, and KNMI projections to demonstrate inadequate policy. -- Evidence-based reasoning is primary.",
            5: "Every clause cites a specific IPCC AR6 conclusion or PBL projection, and the request follows directly from these findings. -- Pure evidence-based reasoning.",
        },
        "llm_question": "To what extent does this text ground its argument in empirical evidence, quantitative data, or cited research -- treating evidence as the reason for action?",
    },
    "security": {
        "label": "F4 - Security",
        "decision_rule": (
            "Economic dependence on foreign energy (higher import costs) is Economic. "
            "Geopolitical vulnerability, strategic autonomy, and national security are Security. "
            "Does the text invoke risk to the nation's sovereignty or safety, or just to its wallet?"
        ),
        "anchors": {
            1: "No connection between climate or energy policy and national security or geopolitical risk.",
            2: "Energy supply reliability is mentioned without security framing. Concern is about costs, not sovereignty.",
            3: "National security or strategic autonomy is explicitly invoked as one justification alongside other frames.",
            4: "Security or strategic autonomy is a primary lens. The text repeatedly frames energy policy as a national security matter.",
            5: "The entire argument is structured around national security. Every justification is geopolitical.",
        },
        "examples": {
            1: "The text requests a feasibility study on hydrogen infrastructure. -- No security framing.",
            2: "The text notes dependence on imported gas increases energy costs. -- Economic dependence, not strategic vulnerability.",
            3: "The text argues reducing dependence on Russian gas serves Dutch strategic interests. -- Security argument present alongside economic framing.",
            4: "The text frames wind and solar expansion as a NATO energy security matter and requests a national energy security plan. -- Security framing central.",
            5: "Every clause connects the energy transition to national defence and fossil dependence to a security liability. -- Pure security logic.",
        },
        "llm_question": "To what extent does this text frame climate or energy policy through the lens of national security, strategic autonomy, or geopolitical stability?",
    },
    "health_environment": {
        "label": "F5 - Health & Environment",
        "decision_rule": (
            "The text must describe specific, named impacts and use them as reasons to act. "
            "Abstract environmental concern scores low; concrete impact evidence scores high."
        ),
        "anchors": {
            1: "No tangible health or environmental impacts described.",
            2: "A general environmental or health reference appears without specificity.",
            3: "One or two specific, named impacts are described and used as justification.",
            4: "Multiple concrete health or environmental impacts are described as primary justifications.",
            5: "The entire argument is organized around cataloguing specific health and environmental impacts.",
        },
        "examples": {
            1: "The text requests implementation of ETS reform. -- No health or environmental impact argument.",
            2: "The text notes the Noordzee is under environmental pressure. -- General concern, no specific impact.",
            3: "The text cites declining marine biodiversity in the North Sea and names protected species at risk. -- Specific impact cited as argument.",
            4: "The text describes heat stress mortality, rising respiratory disease, flooding risk, and biodiversity collapse. -- Health and environmental impacts are primary.",
            5: "Every clause describes a specific harm: excess heat mortality, freshwater ecosystem collapse, coastal erosion. -- Pure health/environment logic.",
        },
        "llm_question": "To what extent does this text emphasize concrete, tangible impacts of climate change on human health, biodiversity, ecosystems, or quality of life -- treating these impacts as the primary reason to act?",
    },
    "crisis_urgency": {
        "label": "F6 - Crisis & Urgency",
        "decision_rule": (
            "A target year mentioned as an administrative deadline scores 1-2. "
            "A target year invoked as a closing window with irreversible consequences scores 3-4. "
            "Is delay framed as procedurally inconvenient, or as catastrophically consequential?"
        ),
        "anchors": {
            1: "No urgency framing. Deadlines appear as administrative facts, not closing windows.",
            2: "A deadline is mentioned as approaching but without crisis vocabulary or catastrophic framing.",
            3: "Urgency is explicitly framed as consequential: delay leads to concrete significant harm.",
            4: "Crisis and urgency framing is central. The text repeatedly invokes tipping points or irreversibility.",
            5: "The entire argument is structured around existential urgency. Every justification is about the imminence of catastrophe.",
        },
        "examples": {
            1: "'The Netherlands aims to be climate neutral by 2050.' -- Target year as planning fact, no urgency.",
            2: "The text requests faster implementation, noting the 2030 deadline is approaching. -- Administrative urgency, not crisis framing.",
            3: "The text argues current trajectories will miss the 1.5C window and further damage becomes irreversible. -- Urgency substantive and consequential.",
            4: "The text invokes climate tipping points, ecosystem collapse, and the closing 1.5C window across multiple clauses. -- Crisis framing dominant.",
            5: "Every clause frames delay as civilizational risk: tipping points crossed, feedback loops triggered. -- Pure crisis logic.",
        },
        "llm_question": "To what extent does this text frame climate change as an imminent crisis with a closing window -- treating delay itself as catastrophic and invoking urgency as the primary reason to act now?",
    },
    "weaponization": {
        "label": "F7 - Weaponization",
        "decision_rule": (
            "This frame requires EXPLICIT political or ideological contestation. "
            "A group being affected by policy is NOT conflict framing -- that is context. "
            "Conflict framing requires the text to argue the policy is politically or ideologically wrong. When in doubt, score 1-2."
        ),
        "anchors": {
            1: "No political or ideological contestation. Climate policy is not framed as politically driven.",
            2: "A stakeholder group is mentioned as affected but without political or ideological framing.",
            3: "The text explicitly argues the cost distribution of climate policy is politically unfair. The political nature is named.",
            4: "Political or ideological conflict is a primary lens. The text repeatedly frames climate policy as ideologically driven.",
            5: "The entire argument is organized around political or ideological conflict. Climate policy is framed as illegitimate overreach.",
        },
        "examples": {
            1: "The text requests more ambitious climate targets for 2030. -- No conflict framing.",
            2: "The text notes stricter emission norms will affect livestock farmers and requests a transition package. -- Farmers mentioned as affected, not as political victims.",
            3: "The text argues nitrogen policy disproportionately burdens farmers while exempting aviation, explicitly framing this as politically motivated. -- Political unfairness named.",
            4: "The text frames climate policy as a left-wing project that systematically burdens rural communities. -- Political conflict framing dominant.",
            5: "Every clause frames climate measures as ideological warfare: farmers sacrificed to an urban green agenda. -- Pure conflict logic.",
        },
        "llm_question": "To what extent does this text frame climate policy as a site of explicit political or ideological conflict -- identifying political losers, naming ideological overreach, or framing climate measures as a zero-sum struggle?",
    },
}

FRAME_KEYS = list(FRAMES.keys())


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_prompt(frame_key: str, text: str) -> str:
    """Build a scoring prompt for a single frame and text."""
    f = FRAMES[frame_key]
    anchors  = "\n".join(f"  {s} -- {d}" for s, d in f["anchors"].items())
    examples = "\n".join(f"  Score {s}: {e}" for s, e in f["examples"].items())
    return (
        f"You are an expert political scientist specializing in climate discourse analysis.\n"
        f"Score the following Dutch political manifesto excerpt on the '{f['label']}' frame.\n\n"
        f"## Decision Rule\n{f['decision_rule']}\n\n"
        f"## Likert Scale\n{LIKERT_DEFINITIONS}\n\n"
        f"## Scoring Anchors\n{anchors}\n\n"
        f"## Calibration Examples\n{examples}\n\n"
        f"## Text to Score (Dutch)\n{text}\n\n"
        f"## Question\n{f['llm_question']}\n\n"
        f"Apply the decision rule first. Then identify which anchor (1-5) best matches. "
        f"Respond with a SINGLE integer from 1 to 5, no explanation."
    )


def parse_score(response: str) -> tuple[int, bool]:
    """Extract a 1-5 score from the model response."""
    match = re.search(r"\b([1-5])\b", response.strip())
    if match:
        return int(match.group(1)), False
    return -1, True


def log_event(event: dict):
    """Append a timestamped event to the run log."""
    with open(RUN_LOG_PATH, "a") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), **event}) + "\n")


def load_checkpoint(ckpt_dir: Path, frame_key: str) -> dict:
    path = ckpt_dir / f"{frame_key}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def save_checkpoint(ckpt_dir: Path, frame_key: str, scores: dict):
    with open(ckpt_dir / f"{frame_key}.pkl", "wb") as f:
        pickle.dump(scores, f)


def score_frame(
    frame_key: str,
    df: pd.DataFrame,
    model,
    tokenizer,
    ckpt_dir: Path,
    model_short: str,
) -> list[int]:
    """Score all phrases on a single frame, with checkpointing."""
    label = FRAMES[frame_key]["label"]
    ckpt  = load_checkpoint(ckpt_dir, frame_key)
    total = len(df)

    if len(ckpt) == total:
        print(f"  {label}: complete checkpoint found, skipping.")
        return [ckpt[i] for i in range(total)]

    print(f"  {label}: {len(ckpt)}/{total} done, resuming...")
    texts       = df[TEXT_COL].tolist()
    todo        = [i for i in range(total) if i not in ckpt]
    parse_fails = 0

    for batch_start in tqdm(range(0, len(todo), BATCH_SIZE), desc=f"  {label}", unit="batch"):
        batch_idx   = todo[batch_start: batch_start + BATCH_SIZE]
        batch_texts = [texts[i] for i in batch_idx]
        t0 = time.time()

        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": build_prompt(frame_key, t)}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for t in batch_texts
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        responses = tokenizer.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        batch_fails = 0
        for pos, (row_i, response) in enumerate(zip(batch_idx, responses)):
            score, failed = parse_score(response)
            if failed:
                score = -1
                batch_fails += 1
                parse_fails += 1
            ckpt[row_i] = score
            if batch_start == 0 and pos < 3:
                print(f"\n  [SAMPLE row={row_i}] '{response.strip()}' -> {score}")

        log_event({
            "event":           "batch_done",
            "model":           model_short,
            "frame":           frame_key,
            "batch_start_row": batch_idx[0],
            "batch_size":      len(batch_idx),
            "parse_failures":  batch_fails,
            "elapsed_s":       round(time.time() - t0, 2),
        })

        save_checkpoint(ckpt_dir, frame_key, ckpt)

    print(f"  {label}: done. Parse failures: {parse_fails}/{total}")
    return [ckpt[i] for i in range(total)]


def load_model(model_name: str, model_short: str):
    """Load a quantized model and tokenizer."""
    print(f"\nLoading {model_name}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=SCRATCH_CACHE)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=SCRATCH_CACHE,
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"  {model_short} loaded.")
    return model, tokenizer

    # Remove model from HF cache to free disk space
    model_cache_name = model_name.replace("/", "--")
    cache_path = Path(SCRATCH_CACHE) / f"models--{model_cache_name}"
    if cache_path.exists():
        shutil.rmtree(cache_path)
        print(f"  Removed cache: {cache_path}")
    
    print(f"  {model_short} unloaded.\n")


def run_model(model_name: str, model_short: str, df: pd.DataFrame):
    """Score all frames for a single model and save output."""
    ckpt_dir = OUT_DIR / f"checkpoints_{model_short}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(model_name, model_short)
    df_out = df.copy()

    log_event({"event": "run_start", "model": model_short, "n_rows": len(df_out)})

    for frame_key in FRAME_KEYS:
        df_out[frame_key] = score_frame(frame_key, df_out, model, tokenizer, ckpt_dir, model_short)

    df_out["dominant_frame"]       = df_out[FRAME_KEYS].idxmax(axis=1)
    df_out["dominant_frame_score"] = df_out[FRAME_KEYS].max(axis=1)
    df_out["n_frames_present"]     = df_out[FRAME_KEYS].apply(lambda r: (r >= 3).sum(), axis=1)
    df_out["has_parse_failure"]    = df_out[FRAME_KEYS].apply(lambda r: (r == -1).any(), axis=1)

    scores_path = OUT_DIR / f"scores_{model_short}.csv"
    df_out.to_csv(scores_path, index=False)
    print(f"\nSaved: {scores_path}")

    print(f"\nFINAL REPORT — {model_short} (N={len(df_out)})")
    for k in FRAME_KEYS:
        valid = df_out[k][df_out[k] != -1]
        dist  = " | ".join(f"{i}:{(valid == i).sum()}" for i in range(1, 6))
        print(f"  {FRAMES[k]['label']:<30}  mean={valid.mean():.2f}  [{dist}]")

    log_event({"event": "run_complete", "model": model_short, "output": str(scores_path)})

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  {model_short} unloaded.\n")

def main():
    print(f"Loading data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df[df[TEXT_COL].notna() & (df[TEXT_COL].str.strip().str.len() > 0)].copy()
    df = df.reset_index(drop=False).rename(columns={"index": "original_index"})
    print(f"{len(df)} phrases loaded.")

    for model_name, model_short in MODELS:
        run_model(model_name, model_short, df)

    print("All models done.")


if __name__ == "__main__":
    main()