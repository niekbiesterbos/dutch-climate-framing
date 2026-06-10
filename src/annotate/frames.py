"""
Shared macro-frame definitions (Biesterbos 2025 v4).

Seven behaviorally anchored frames scored 1–5:
  1 = Absent   2 = Marginal   3 = Present   4 = Prominent   5 = Dominant
"""

FRAMES = {
    "economic": {
        "label": "F1 — Economic",
        "decision_rule": (
            "Is the central justification economic? Score high when the text argues "
            "that climate policy is good or bad because of what it costs or earns. "
            "Score low when economic terms appear only as context for a regulatory request."
        ),
        "anchors": {
            1: "No mention of costs, jobs, investment, competitiveness, or economic consequences.",
            2: "An economic term appears once as context or background; not used as argument.",
            3: "Economic consequences justify the argument across ≥2 sentences, but not the primary frame.",
            4: "Economic logic is one of two central pillars; multiple sentences organised around it.",
            5: "Entire argument structured around economic logic; no other frame plays a meaningful role.",
        },
    },
    "moral": {
        "label": "F2 — Moral & Ethical",
        "decision_rule": (
            "Does the text explicitly name a moral obligation, duty, or injustice? "
            "Implied concern for nature/health is NOT moral framing. "
            "Fairness between nations (historical emitters vs. vulnerable nations) qualifies; "
            "fairness about domestic cost distribution belongs to Economic (F1)."
        ),
        "anchors": {
            1: "No moral vocabulary; no duty, obligation, fairness, justice, or rights invoked.",
            2: "A single moral phrase appears but is not elaborated or used as justification.",
            3: "Moral reasoning explicitly used as one justification across ≥2 sentences.",
            4: "Moral framing is the primary lens; obligations/injustices drive the argument.",
            5: "Entire argument organised around moral obligation; no other frame structures the reasoning.",
        },
    },
    "scientific": {
        "label": "F3 — Scientific",
        "decision_rule": (
            "Does the text invoke empirical evidence — data, measurements, projections, "
            "scientific consensus — as a primary reason to act or not act? "
            "Mentioning 'science says X' once as backdrop does not qualify."
        ),
        "anchors": {
            1: "No scientific evidence, data, or research cited.",
            2: "A scientific fact or study mentioned once as background; not used as argument.",
            3: "Empirical evidence explicitly used to justify action across ≥2 sentences.",
            4: "Scientific evidence is the primary justification; multiple data points invoked.",
            5: "Entire argument built on scientific evidence; every claim backed by data or research.",
        },
    },
    "security": {
        "label": "F4 — Security",
        "decision_rule": (
            "Is geopolitical stability, national interest, energy security, or defence "
            "the central lens? Distinguish from Economic: security framing concerns threats "
            "to state sovereignty and stability, not competitiveness."
        ),
        "anchors": {
            1: "No security or geopolitical framing present.",
            2: "A security term appears once as context; not developed as argument.",
            3: "Security concerns justify action across ≥2 sentences alongside other frames.",
            4: "Security is the primary lens; national interest or energy independence drives the argument.",
            5: "Entire argument structured around security threats; no other frame is relevant.",
        },
    },
    "health_environment": {
        "label": "F5 — Health & Environment",
        "decision_rule": (
            "Are direct health impacts (illness, mortality) or ecosystem damage "
            "(biodiversity loss, habitat destruction) the central justification? "
            "General concern for 'the environment' without concrete impacts scores low."
        ),
        "anchors": {
            1: "No concrete health impacts or ecosystem damage mentioned.",
            2: "Health or environmental harm mentioned once as backdrop.",
            3: "Health/environmental harms explicitly justify action across ≥2 sentences.",
            4: "Health or ecosystem framing is the primary lens; specific impacts named.",
            5: "Entire argument structured around concrete health or ecological harm.",
        },
    },
    "crisis_urgency": {
        "label": "F6 — Crisis & Urgency",
        "decision_rule": (
            "Does the text frame climate as an existential crisis, tipping point, "
            "or irreversible threat that demands immediate action? "
            "Urgency language without catastrophic framing scores lower."
        ),
        "anchors": {
            1: "No crisis, emergency, or urgency language present.",
            2: "A single urgency phrase appears but is not developed.",
            3: "Crisis framing explicitly invoked across ≥2 sentences alongside other frames.",
            4: "Existential threat or urgency is the primary driver of the argument.",
            5: "Entire argument organised around crisis and irreversibility; action framed as last chance.",
        },
    },
    "weaponization": {
        "label": "F7 — Weaponization",
        "decision_rule": (
            "Is climate used as a political conflict instrument — attacking opponents, "
            "framing the debate as a fight between groups, or delegitimising the other side? "
            "Critical policy evaluation is NOT weaponization; personal attacks and polarising "
            "framing are."
        ),
        "anchors": {
            1: "No adversarial or conflict framing present.",
            2: "A mildly critical reference to opponents appears but is not central.",
            3: "Climate explicitly used as a political weapon across ≥2 sentences.",
            4: "Conflict framing is one of two central pillars; opponents attacked or delegitimised.",
            5: "Entire text is organised as a political attack using climate as the instrument.",
        },
    },
}

FRAME_KEYS = list(FRAMES.keys())
LIKERT_SCALE = "1=Absent  2=Marginal  3=Present  4=Prominent  5=Dominant"
