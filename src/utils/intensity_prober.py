import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm

# Zorg dat je API key in je environment staat: export OPENAI_API_KEY='jouw-key'
client = OpenAI()


def analyze_motion_nisbet(text):
    # De Questionnaire: we vragen specifiek naar de 7 frames
    # Dit is de 'Adaptation' stap van de Mitran-methode
    system_prompt = """
    Je bent een politiek analist. Beantwoord voor de gegeven motie de volgende vragenlijst 
    om de intensiteit van de 7 Nisbet-frames te bepalen op een schaal van 1 (niet aanwezig) tot 5 (zeer dominant).

    VRAGENLIJST:
    1. Economic: In hoeverre benadrukt de tekst financiële kosten, banen, innovatie of concurrentie?
    2. Moral: In hoeverre beroept de tekst zich op rechtvaardigheid, ethiek of morele plicht?
    3. Scientific: In hoeverre wordt er verwezen naar wetenschappelijk bewijs, data of onderzoek?
    4. Governance: In hoeverre gaat het over wetgeving, beleidsinstrumenten of instituties?
    5. Security: In hoeverre speelt energie-onafhankelijkheid of nationale veiligheid een rol?
    6. Health/Env: In hoeverre worden ecosystemen of de volksgezondheid als argument gebruikt?
    7. Crisis: In hoeverre wordt er geduid op een naderende dreiging of extreme tijdsdruk?

    OUTPUT: Geef alleen een JSON object terug met de scores (1-5).
    Formaat: {"economic": int, "moral": int, "scientific": int, "governance": int, "security": int, "health": int, "crisis": int}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except:
        return None

# Normalisatie functie: zet 1-5 om naar 0.0-1.0
# Formule: (x - min) / (max - min) -> (x - 1) / 4


def normalize_score(score):
    if score is None:
        return 0.0
    return (score - 1) / (5 - 1)


# Inlezen en subset maken
df = pd.read_csv(
    "data/formatted_data/all_motions_2008_2025_with_text_and_normalized.csv")
df_subset = df.head(1000).copy()

# Data verzamelen
results_list = []
for index, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
    raw_scores = analyze_motion_nisbet(str(row['normalized_text'])[:4000])

    if raw_scores:
        # Hier gebeurt de normalisatie naar de "Frame Intensity Vector"
        normalized_vector = {f"norm_{k}": normalize_score(
            v) for k, v in raw_scores.items()}
        # Combineer originele scores + genormaliseerde scores
        results_list.append({**raw_scores, **normalized_vector})
    else:
        results_list.append(
            {f: 0 for f in ["economic", "norm_economic"]})  # fallback

# Toevoegen aan dataframe
res_df = pd.DataFrame(results_list)
df_final = pd.concat([df_subset.reset_index(drop=True), res_df], axis=1)

df_final.to_csv(
    "data/formatted_data/motions_nisbet_probed_1000.csv", index=False)
