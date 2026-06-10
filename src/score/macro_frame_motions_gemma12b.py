"""
Macro-Frame Annotation via LLM
=======================================
Scores all climate motions on 7 macro-frames (Likert 1-5) per motion,
based on the annotation scheme of Biesterbos (2025) v4.

Frame definitions, decision rules, anchors, and prompts are in English.
Motion texts remain in Dutch.

Checkpointing per frame in results/motions/macro_scores/checkpoints_{MODEL_SHORT}/
"""

import os
import re
import time
import json
import pickle
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# 1. CONFIG & PATHS
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

MODEL_NAME  = "google/gemma-3-12b-it"
MODEL_SHORT = "gemma-3-12b-it"
DATA_PATH   = "results/motions/climate_motions.csv"
TEXT_COL    = "normalized_text"
BATCH_SIZE  = 32
MAX_NEW_TOKENS = 8
OUT_DIR  = Path("results/motions/macro_scores")
CKPT_DIR = OUT_DIR / f"checkpoints_{MODEL_SHORT}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

RUN_LOG_PATH = OUT_DIR / "run_log.jsonl"

# ---------------------------------------------------------------------------
# 2. FRAME DEFINITIONS -- Biesterbos (2025) v4, behaviorally anchored
# ---------------------------------------------------------------------------
FRAMES = {
    "economic": {
        "label": "F1 - Economic",
        "decision_rule": (
            "Ask: is the central justification in this text economic? "
            "If the text argues that climate policy is good or bad because of what it costs or earns, score high. "
            "If economic terms appear as context for a regulatory request, score low."
        ),
        "anchors": {
            1: "No mention of costs, jobs, investment, innovation, competitiveness, or economic consequences of climate policy. The text is about something else entirely.",
            2: "An economic term (cost, subsidy, levy, jobs) appears once as context or background. It does not justify the request; it merely acknowledges that an economic dimension exists.",
            3: "Economic consequences are explicitly used to support or oppose the policy request across at least two sentences, but the text has another primary frame.",
            4: "The economic logic is one of two central pillars of the argument. Multiple sentences or overwegende clauses are organized around economic costs, competitiveness effects, or economic opportunities.",
            5: "The entire argument, from constaterende through verzoekt, is structured around economic logic. Every reason given for action is economic. No other frame plays a meaningful role.",
        },
        "examples": {
            1: 'Parliament requests the government to designate additional Natura 2000 areas before 2025. -- No economic dimension whatsoever.',
            2: 'The motion notes that the energy transition will involve transition costs, and requests the government to publish a renewable energy roadmap. -- Cost mentioned once, not used as argument.',
            3: 'The motion argues the subsidy scheme will create 5,000 regional jobs and stimulate clean-tech innovation, while requesting the government to extend the SDE++ scheme. -- Economic benefit argued but core request is regulatory.',
            4: 'The motion argues that ETS reform will cost Dutch industry EUR 2 billion, disadvantage Dutch firms relative to German competitors, and threaten 12,000 manufacturing jobs. -- Economic framing dominates alongside one other frame.',
            5: 'The motion argues that climate levies destroy household purchasing power, make Dutch industry globally uncompetitive, increase energy poverty, and must be replaced by market-based instruments. -- Pure economic reasoning throughout.',
        },
        "llm_question": "To what extent does this text frame climate policy through the logic of economic consequences -- costs, competitiveness, jobs, or innovation opportunities -- as the primary reason to act or not act?",
    },
    "moral": {
        "label": "F2 - Moral & Ethical",
        "decision_rule": (
            "Ask: does the text explicitly state that someone has a moral obligation or that the situation is unjust? "
            "Implied concern for nature or health is not moral framing. "
            "Fairness about distribution of costs between domestic groups is Economic (F1), not Moral. "
            "Fairness between nations (rich vs poor countries, historical emitters vs vulnerable nations) is Moral."
        ),
        "anchors": {
            1: "No moral vocabulary present. The text does not invoke duty, obligation, fairness, justice, responsibility, or rights in relation to climate. Concern for nature or health alone does not qualify.",
            2: "A single moral phrase appears ('our responsibility', 'fair share', 'we owe it') but it is not elaborated or used as a justification. It functions as rhetorical decoration, not as an argument.",
            3: "Moral reasoning is explicitly used as one justification for action across at least two sentences. The text names a specific obligation and treats it as a genuine reason to act -- but other frames are equally or more prominent.",
            4: "Moral or ethical framing is the primary lens. The text repeatedly names obligations, duties, or injustices as the main reasons for action. Other frames may be present but the moral logic drives the argument.",
            5: "The entire argument is organized around moral obligation. Every clause names a duty, a right, or an injustice. No other frame structures the reasoning.",
        },
        "examples": {
            1: 'Parliament requests an investigation into hydrogen storage capacity by 2026. -- Purely technical; no moral dimension.',
            2: 'The motion briefly states that the Netherlands has a responsibility to lead on climate, then focuses entirely on the ETS reform mechanism. -- Responsibility named once, dropped immediately.',
            3: 'The motion argues that as a historically high-emitting nation, the Netherlands bears a special obligation to lead, and that future generations must not inherit a destabilized climate. -- Moral argument present and substantive, alongside governance framing.',
            4: 'The motion argues repeatedly that wealthy nations have a historical debt to the Global South, that inaction violates intergenerational justice, and that the Netherlands must act out of moral obligation regardless of economic cost. -- Moral reasoning central throughout.',
            5: 'The motion argues that we owe it to our children, to developing nations bearing the worst impacts they did not cause, and to the generations who will inherit a destabilized world. Acting is a moral imperative that overrides economic considerations. -- Pure moral logic.',
        },
        "llm_question": "To what extent does this text ground its climate argument in explicit moral duty, fairness, justice, or responsibility -- naming who owes what to whom?",
    },
    "scientific": {
        "label": "F3 - Scientific",
        "decision_rule": (
            "Ask: is evidence being used as a reason to act, or just as decoration? "
            "A year mentioned as a target deadline is not scientific framing. "
            "A PBL report cited to show that current policy is insufficient is scientific framing. "
            "The evidence must function as an argument, not as background."
        ),
        "anchors": {
            1: "No empirical data, research citations, or quantitative claims. The argument is not grounded in evidence. Targets or years mentioned without supporting data do not qualify.",
            2: "A single number, percentage, or factual reference appears but is not used as an argument. It provides context or decorates a policy request without serving as a reason for action.",
            3: "One or two empirical claims or citations are explicitly used as justification for the policy request. The evidence functions as a reason to act, not merely as background, but the text is not organized primarily around evidence.",
            4: "Multiple empirical sources, datasets, or quantitative claims are cited and used as primary justifications. The text builds its case through evidence across several overwegende clauses.",
            5: "The entire argument is structured around empirical evidence. Every claim is grounded in cited research or data. The policy request follows logically from the evidence presented.",
        },
        "examples": {
            1: 'Parliament requests the government to promote sustainable agriculture in cooperation with the sector. -- No data or evidence of any kind.',
            2: 'The motion states that CO2 emissions must fall and references the 2030 reduction target, then requests policy measures. -- A target year appears but no evidence is cited to justify it.',
            3: 'The motion cites PBL findings that current policy is insufficient to meet 2030 targets, using this to justify stricter measures. -- One empirical source cited and used as argument.',
            4: 'The motion cites IPCC emission thresholds, PBL policy gap analysis, and KNMI temperature projections across four overwegende clauses to demonstrate that current policy is inadequate. -- Evidence-based reasoning is the primary mode of argument.',
            5: 'Every overwegende clause cites a specific IPCC AR6 conclusion, PBL projection, or SCP dataset, and the verzoekt follows directly from these empirical findings. -- Pure evidence-based reasoning; no other frame plays a role.',
        },
        "llm_question": "To what extent does this text ground its argument in empirical evidence, quantitative data, or cited research from scientific bodies or official reports -- treating evidence as the reason for action?",
    },
    "security": {
        "label": "F4 - Security",
        "decision_rule": (
            "Economic dependence on foreign energy (higher import costs) is Economic (F1). "
            "Geopolitical vulnerability, strategic autonomy, and national security are Security (F5). "
            "The distinction: does the text invoke risk to the nation's sovereignty or safety, or just to its wallet?"
        ),
        "anchors": {
            1: "No connection between climate or energy policy and national security, strategic autonomy, or geopolitical risk. Energy may be mentioned but purely in economic or technical terms.",
            2: "Energy supply reliability or import dependence is mentioned without security framing. The concern is about costs or availability, not about national sovereignty or geopolitical risk.",
            3: "National security or strategic autonomy is explicitly invoked as one justification for energy or climate policy, alongside other frames. The security argument is substantive but not the sole driver.",
            4: "Security or strategic autonomy is a primary lens. The text repeatedly frames energy or climate policy as a national security matter, invoking geopolitical vulnerability, NATO obligations, or supply chain resilience.",
            5: "The entire argument is structured around national security or strategic autonomy. Every justification is geopolitical. No other frame plays a meaningful role.",
        },
        "examples": {
            1: 'Parliament requests a feasibility study on hydrogen infrastructure expansion. -- No security framing; purely technical/economic.',
            2: 'The motion notes that dependence on imported gas increases energy costs and requests measures to boost domestic production. -- Economic dependence, not strategic vulnerability.',
            3: 'The motion argues that reducing dependence on Russian gas serves both economic resilience and Dutch strategic interests within the EU. -- Security argument present and explicit, alongside economic framing.',
            4: 'The motion frames wind and solar expansion explicitly as a matter of NATO energy security, argues that fossil dependence creates strategic vulnerability to authoritarian supplier states, and requests a national energy security plan. -- Security framing central across multiple clauses.',
            5: 'Every clause connects the energy transition to national defence: fossil dependence is framed as a security liability, renewables as strategic infrastructure, and delay as a threat to NATO obligations. -- Pure security logic.',
        },
        "llm_question": "To what extent does this text frame climate or energy policy through the lens of national security, strategic autonomy, or geopolitical stability -- treating energy independence as a strategic, not merely economic, imperative?",
    },
    "health_environment": {
        "label": "F5 - Health & Environment",
        "decision_rule": (
            "Mentioning that something is 'bad for nature' or 'affects the environment' is not enough for a high score. "
            "The text must describe specific, named impacts and use them as reasons to act. "
            "Named ecosystems with documented decline data score 3. Named species plus quantified impact data score 4. "
            "Abstract environmental concern scores 1-2; concrete impact evidence scores 3-5."
        ),
        "anchors": {
            1: "No tangible health or environmental impacts described. The text does not invoke effects on human health, specific ecosystems, biodiversity, or quality of life as reasons for action.",
            2: "A general environmental or health reference appears ('nature is under pressure', 'affects public health') without specificity. No named species, ecosystem, health condition, or community is invoked.",
            3: "One or two specific, named impacts are described and used as justification. The impact is concrete -- a named ecosystem, health condition, or community effect -- and explicitly cited as a reason for action.",
            4: "Multiple concrete health or environmental impacts are described and used as primary justifications across several clauses. The text is substantially organized around demonstrating harm to people or nature.",
            5: "The entire argument is organized around cataloguing and connecting specific health and environmental impacts to the policy request. Every justification is an impact on people or nature. No other frame plays a role.",
        },
        "examples": {
            1: 'Parliament requests the government to implement the ETS reform by 2025 and report back within six months. -- No health or environmental impact argument.',
            2: 'The motion notes that the Noordzee is under environmental pressure and requests an action programme for marine protection. -- Named ecosystem but no specific impact data; general concern only.',
            3: 'The motion cites the Natuurbalans 2008 finding that marine biodiversity in the North Sea is declining, names specific protected species at risk, and uses this to justify a marine protection programme. -- Specific environmental impact cited and used as argument.',
            4: 'The motion describes heat stress mortality among the elderly, rising respiratory disease rates from air pollution, flooding risk to coastal communities, and biodiversity collapse in delta ecosystems -- all used as reasons to demand accelerated climate action. -- Health and environmental impacts are the primary mode of argument.',
            5: 'Every overwegende clause describes a specific harm: excess heat mortality rates, particulate matter illness burden, freshwater ecosystem collapse, and coastal erosion affecting named municipalities -- each directly used to justify the intervention. -- Pure health/environment logic.',
        },
        "llm_question": "To what extent does this text emphasize concrete, tangible impacts of climate change or climate policy on human health, biodiversity, ecosystems, or quality of life -- treating these impacts as the primary reason to act?",
    },
    "crisis_urgency": {
        "label": "F6 - Crisis & Urgency",
        "decision_rule": (
            "A target year (2030, 2050) mentioned as an administrative deadline scores 1-2. "
            "A target year invoked as a closing window that will lead to irreversible consequences if missed scores 3-4. "
            "The distinction: is delay framed as procedurally inconvenient, or as catastrophically consequential?"
        ),
        "anchors": {
            1: "No urgency framing. Deadlines or targets appear as administrative facts or planning horizons, not as indicators of a closing window. The text does not invoke irreversibility, tipping points, or catastrophic consequences of delay.",
            2: "A deadline is mentioned and described as approaching, but without crisis vocabulary or catastrophic framing. The urgency is procedural ('we need to act soon') rather than existential ('failure means irreversible harm').",
            3: "Urgency is explicitly framed as consequential: delay is described as leading to concrete, significant harm. A closing window, tipping point, or irreversibility is invoked as a substantive reason to act -- but alongside other frames.",
            4: "Crisis and urgency framing is central. The text repeatedly invokes tipping points, irreversibility, or the catastrophic consequences of delay as primary reasons for immediate action.",
            5: "The entire argument is structured around existential urgency. Every justification is about the imminence of catastrophe and the impossibility of further delay. No other frame structures the reasoning.",
        },
        "examples": {
            1: '"The Netherlands aims to be climate neutral by 2050 in line with the European Green Deal." -- Target year as planning fact, no urgency framing.',
            2: 'The motion requests faster implementation, noting that the 2030 deadline is approaching and that current policy timelines are too slow. -- Administrative urgency, not crisis framing.',
            3: 'The motion argues that accelerated action is needed because current trajectories will miss the 1.5C window, invoking the Urgenda ruling to argue the state is legally required to act before further damage becomes irreversible. -- Urgency argument substantive and consequential.',
            4: 'The motion invokes climate tipping points, the irreversibility of ecosystem collapse, the Urgenda legal obligation, and the closing 1.5C window across four overwegende clauses, arguing that each year of delay multiplies eventual harm. -- Crisis framing dominant.',
            5: 'Every clause frames delay as civilizational risk: tipping points already being crossed, feedback loops being triggered, and legal obligations being violated. The verzoekt calls for emergency measures with immediate effect. -- Pure crisis logic throughout.',
        },
        "llm_question": "To what extent does this text frame climate change as an imminent crisis with a closing window for action -- treating delay itself as catastrophic and invoking urgency as the primary reason to act now?",
    },
    "conflict_strategy": {
        "label": "F7 - Conflict & Strategy",
        "decision_rule": (
            "This frame requires EXPLICIT political or ideological contestation. "
            "A group being affected by policy (farmers, fishermen, businesses) is NOT conflict framing -- that is context. "
            "Conflict framing requires the text to argue that the policy is politically or ideologically wrong, "
            "that specific groups are being deliberately harmed by political choices, "
            "or that the policy represents ideological overreach. When in doubt, score 1-2."
        ),
        "anchors": {
            1: "No political or ideological contestation. The text does not frame climate policy as politically driven, ideologically motivated, or as a zero-sum struggle. A group affected by policy without ideological framing scores 1.",
            2: "A stakeholder group is mentioned as being affected by climate policy (farmers, fishermen, businesses, regions) but without any political or ideological framing. Their impact is noted as a practical consideration, not as political victimization.",
            3: "The text explicitly argues that the cost distribution of climate policy is politically unfair -- that a specific group is bearing disproportionate costs due to political choices. The political nature of the grievance is named, not just implied.",
            4: "Political or ideological conflict is a primary lens. The text repeatedly frames climate policy as politically contestable -- as driven by ideology, as producing deliberately unfair outcomes, or as part of a broader political struggle.",
            5: "The entire argument is organized around political or ideological conflict. Climate policy is framed throughout as illegitimate overreach, elite imposition on ordinary citizens, or a zero-sum struggle. No other frame plays a role.",
        },
        "examples": {
            1: 'Parliament requests the government to formulate more ambitious climate targets for 2030. -- No conflict framing whatsoever.',
            2: 'The motion notes that stricter emission norms will affect livestock farmers and requests a transition support package. -- Farmers mentioned as affected group, not as political victims of ideological overreach.',
            3: 'The motion argues that nitrogen policy disproportionately burdens farmers while exempting aviation and shipping, explicitly framing this as a politically motivated unequal treatment. -- Political unfairness named explicitly.',
            4: 'The motion frames climate policy as a left-wing ideological project that systematically burdens rural communities and manufacturing workers while exempting urban elites. -- Political conflict framing dominant.',
            5: 'Every clause frames climate measures as ideological warfare: farmers are being sacrificed to an urban green agenda, working-class energy bills are subsidizing elite virtue signalling, and the government is using climate as cover for a power grab. -- Pure conflict logic from start to finish.',
        },
        "llm_question": "To what extent does this text frame climate policy as a site of explicit political or ideological conflict -- identifying specific political losers, naming ideological overreach, or framing climate measures as a zero-sum struggle between groups?",
    },
}

FRAME_KEYS = list(FRAMES.keys())

LIKERT_DEFINITIONS = (
    "1 = Absent    -- No recognizable signal of this frame. Core concepts and logic entirely absent.\n"
    "2 = Marginal  -- A passing reference, but the frame does not structure the argument.\n"
    "3 = Present   -- The frame is clearly present and contributes, but is not the central organizing logic.\n"
    "4 = Prominent -- The frame is a primary lens; multiple sentences or rhetorical structure organized around it.\n"
    "5 = Dominant  -- The frame fully structures the text's argument."
)

# ---------------------------------------------------------------------------
# 3. PROMPT BUILDER
# ---------------------------------------------------------------------------
def build_prompt(frame_key: str, text: str) -> str:
    f = FRAMES[frame_key]
    anchor_lines  = "\n".join(f"  {score} -- {desc}" for score, desc in f["anchors"].items())
    example_lines = "\n".join(f"  Score {score}: {ex}" for score, ex in f["examples"].items())
    return (
        f"You are an expert political scientist specializing in climate discourse analysis.\n"
        f"Score the following Dutch parliamentary motion on the '{f['label']}' frame.\n\n"
        f"## Decision Rule\n{f['decision_rule']}\n\n"
        f"## Likert Scale\n{LIKERT_DEFINITIONS}\n\n"
        f"## Scoring Anchors\n{anchor_lines}\n\n"
        f"## Calibration Examples\n{example_lines}\n\n"
        f"## Motion to Score (Dutch)\n{text}\n\n"
        f"## Question\n{f['llm_question']}\n\n"
        f"Apply the decision rule first. Then identify which anchor (1-5) best matches. "
        f"Respond with a SINGLE integer from 1 to 5, no explanation."
    )


def parse_score(response: str) -> tuple[int, bool]:
    match = re.search(r"\b([1-5])\b", response.strip())
    if match:
        return int(match.group(1)), False
    return -1, True


# ---------------------------------------------------------------------------
# 4. LOGGING
# ---------------------------------------------------------------------------
def log_event(event: dict):
    with open(RUN_LOG_PATH, "a") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), **event}) + "\n")


# ---------------------------------------------------------------------------
# 5. MODEL LOADING
# ---------------------------------------------------------------------------
print(f"[{datetime.now():%H:%M:%S}] Loading model: {MODEL_NAME}")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=SCRATCH_CACHE)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir=SCRATCH_CACHE,
    attn_implementation="sdpa",
)
model.eval()
print(f"[{datetime.now():%H:%M:%S}] Model loaded.")

# ---------------------------------------------------------------------------
# 6. DATA LOADING
# ---------------------------------------------------------------------------
print(f"[{datetime.now():%H:%M:%S}] Loading data: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

if TEXT_COL not in df.columns:
    raise ValueError(f"Column '{TEXT_COL}' not found. Available: {list(df.columns)}")

df = df[df[TEXT_COL].notna() & (df[TEXT_COL].str.strip().str.len() > 0)].copy()
df = df.reset_index(drop=False).rename(columns={"index": "original_index"})

N = len(df)
print(f"[{datetime.now():%H:%M:%S}] {N} motions loaded after filtering.")
log_event({"event": "data_loaded", "n_rows": N, "text_col": TEXT_COL, "model": MODEL_NAME})

# ---------------------------------------------------------------------------
# 7. CHECKPOINTED FRAME SCORING
# ---------------------------------------------------------------------------
def load_checkpoint(frame_key: str) -> dict:
    path = CKPT_DIR / f"{frame_key}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def save_checkpoint(frame_key: str, scores: dict):
    with open(CKPT_DIR / f"{frame_key}.pkl", "wb") as f:
        pickle.dump(scores, f)


def score_frame_full(frame_key: str, df: pd.DataFrame) -> list[int]:
    label = FRAMES[frame_key]["label"]
    ckpt  = load_checkpoint(frame_key)
    total = len(df)

    if len(ckpt) == total:
        print(f"  {label}: complete checkpoint found, skipping.")
        return [ckpt[i] for i in range(total)]

    print(f"  {label}: {len(ckpt)}/{total} done, continuing...")
    texts          = df[TEXT_COL].tolist()
    todo_indices   = [i for i in range(total) if i not in ckpt]
    parse_failures = 0

    for batch_start in tqdm(range(0, len(todo_indices), BATCH_SIZE), desc=f"  {label}", unit="batch"):
        batch_idx   = todo_indices[batch_start: batch_start + BATCH_SIZE]
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
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            )

        responses = tokenizer.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
        )

        batch_failures = 0
        for pos, (row_i, response) in enumerate(zip(batch_idx, responses)):
            score, failed = parse_score(response)
            if failed:
                score = -1
                batch_failures += 1
                parse_failures += 1
            ckpt[row_i] = score
            if batch_start == 0 and pos < 3:
                print(f"\n[SAMPLE row={row_i}] response: '{response.strip()}' -> score: {score}")

        log_event({
            "event": "batch_done", "frame": frame_key,
            "batch_start_row": batch_idx[0], "batch_size": len(batch_idx),
            "parse_failures": batch_failures, "elapsed_s": round(time.time() - t0, 2),
        })

        for row_i in batch_idx:
            df.at[row_i, f"frame_{frame_key}"] = ckpt[row_i]
        if batch_start % 10 == 0:
            df.to_csv(OUT_DIR / f"scores_interim_{MODEL_SHORT}.csv", index=False)

        save_checkpoint(frame_key, ckpt)

    print(f"  {label}: done. Parse failures: {parse_failures}/{total}")
    log_event({"event": "frame_done", "frame": frame_key, "parse_failures": parse_failures})
    return [ckpt[i] for i in range(total)]


# ---------------------------------------------------------------------------
# 8. EXECUTION
# ---------------------------------------------------------------------------
print(f"\n[{datetime.now():%H:%M:%S}] Start scoring: {N} motions x {len(FRAME_KEYS)} frames")
log_event({"event": "run_start", "n_rows": N, "frames": FRAME_KEYS, "batch_size": BATCH_SIZE})

for frame_key in FRAME_KEYS:
    df[f"frame_{frame_key}"] = score_frame_full(frame_key, df)

print(f"\n[{datetime.now():%H:%M:%S}] All frames scored.")

# ---------------------------------------------------------------------------
# 9. DERIVED METADATA
# ---------------------------------------------------------------------------
frame_cols = [f"frame_{k}" for k in FRAME_KEYS]

df["dominant_frame_key"]   = df[frame_cols].idxmax(axis=1).str.replace("frame_", "")
df["dominant_frame_label"] = df["dominant_frame_key"].map({k: FRAMES[k]["label"] for k in FRAME_KEYS})
df["dominant_frame_score"] = df[frame_cols].max(axis=1)
df["prominent_frames"]     = df.apply(
    lambda row: "|".join(k for k in FRAME_KEYS if row[f"frame_{k}"] >= 4), axis=1
)
df["n_frames_present"]  = df[frame_cols].apply(lambda r: (r >= 3).sum(), axis=1)
df["has_parse_failure"] = df[frame_cols].apply(lambda r: (r == -1).any(), axis=1)

n_failures = df["has_parse_failure"].sum()
print(f"[INFO] Motions with 1+ parse failure: {n_failures}/{N}")
log_event({"event": "parse_failures_total", "n_motions_affected": int(n_failures)})

# ---------------------------------------------------------------------------
# 10. SAVE
# ---------------------------------------------------------------------------
scores_path = OUT_DIR / f"scores_{MODEL_SHORT}.csv"
df.to_csv(scores_path, index=False)
print(f"[INFO] Scores saved: {scores_path}  ({len(df)} rows)")

id_col = next((c for c in ["motion_id", "id", "doc_id", "motie_id", "original_index"] if c in df.columns), "original_index")
fiv_cols = [id_col] + frame_cols + ["dominant_frame_key", "dominant_frame_label",
                                     "dominant_frame_score", "prominent_frames",
                                     "n_frames_present", "has_parse_failure"]
fiv_df = df[fiv_cols].rename(columns={f"frame_{k}": FRAMES[k]["label"] for k in FRAME_KEYS})
fiv_path = OUT_DIR / f"fivs_{MODEL_SHORT}.csv"
fiv_df.to_csv(fiv_path, index=False)
print(f"[INFO] FIVs saved: {fiv_path}  ({len(fiv_df)} rows)")

# ---------------------------------------------------------------------------
# 11. FINAL REPORT
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print(f"FINAL REPORT EXP_9  (N={N}, model={MODEL_SHORT})")
print("=" * 65)
for k in FRAME_KEYS:
    col   = f"frame_{k}"
    valid = df[col][df[col] != -1]
    dist  = " | ".join(f"{i}:{(valid == i).sum()}" for i in range(1, 6))
    print(f"{FRAMES[k]['label']:<30}  mean={valid.mean():.2f}  [{dist}]")
print("-" * 65)
print("Dominant frame distribution:")
print(df["dominant_frame_label"].value_counts().to_string())
print("=" * 65)

log_event({"event": "run_complete", "output_scores": str(scores_path), "output_fivs": str(fiv_path)})
print(f"\n[{datetime.now():%H:%M:%S}] Done. Output in: {OUT_DIR}")