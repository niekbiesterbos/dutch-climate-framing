import os
import pandas as pd
import re
from lxml import etree

# Paden instellen
BASE_PATH = 'data/speeches/ParlaMint-NL'
OUTPUT_FILE = 'results/motions/parlamint_seed_motions.csv'

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

THEMA_MAPPING = {
    'argic': 'climate_agriculture_energy', 'envir': 'climate_agriculture_energy',
    'energ': 'climate_agriculture_energy', 'lands': 'climate_agriculture_energy',
    'macro': 'economy_finance', 'domes': 'economy_finance', 'trade': 'economy_finance',
    'labor': 'labor_social_security', 'welfa': 'labor_social_security',
    'healt': 'health_care', 'educa': 'education_science_culture',
    'techn': 'education_science_culture', 'cultu': 'education_science_culture',
    'trans': 'housing_infrastructure', 'housi': 'housing_infrastructure',
    'lawcr': 'security_justice_defense', 'defen': 'security_justice_defense',
    'civil': 'rights_democracy', 'gover': 'rights_democracy',
    'immig': 'migration_integration', 'inter': 'foreign_european_policy'
}


def extract_strict_motions(file_path):
    try:
        # Datum extractie uit bestandsnaam
        date_match = re.search(
            r'\d{4}-\d{2}-\d{2}', os.path.basename(file_path))
        file_path_match = re.search(r'ParlaMint-NL_.*', os.path.basename(file_path))
        file_date = date_match.group(0) if date_match else "unknown"
        file_path_cleaned = file_path_match.group(0) if file_path_match else "unknown"
        
        tree = etree.parse(file_path)
        root = tree.getroot()
        # Namespace voor xml:id is essentieel voor doc_id
        ns = {'tei': 'http://www.tei-c.org/ns/1.0',
              'xml': 'http://www.w3.org/XML/1.98/namespace'}

        motions_data = []

        for u in root.xpath('.//tei:u', namespaces=ns):
            # Gebruik de volledige namespace URI voor de xml:id
            u_id = u.get(
                '{http://www.w3.org/XML/1.98/namespace}id') or "unknown"
            speaker = u.get('who', 'unknown')
            ana_str = u.get('ana', '')
            raw_topic = ana_str.split(
                'topic:')[-1] if 'topic:' in ana_str else 'other'
            mapped_theme = THEMA_MAPPING.get(raw_topic, 'other')

            # We pakken alle kinderen van de utterance (notes en segs)
            elements = u.xpath('./*', namespaces=ns)

            current_motion_text = []
            is_collecting = False

            for elem in elements:
                # Haal alle tekst op, ook uit geneste tags
                text = "".join(elem.xpath('.//text()')).strip()
                if not text:
                    continue

                # START MARKER: "Motie" of "De Kamer,"
                # Dit markeert het begin van de formele motie-tekst
                if text.lower() == "motie" or text.lower().startswith("de kamer,"):
                    # Als we al bezig waren (bijv. bij opeenvolgende moties), sla de vorige op
                    if is_collecting and current_motion_text:
                        full_txt = " ".join(current_motion_text).strip()
                        motions_data.append({
                            'doc_id': file_path_cleaned,
                            'date': file_date,
                            'speaker': speaker,
                            'hoofd_thema': mapped_theme,
                            'text': full_txt
                        })
                        current_motion_text = []

                    is_collecting = True
                    # We voegen "De Kamer," vaak wel toe voor de volledigheid van de tekst
                    if text.lower().startswith("de kamer,"):
                        current_motion_text.append(text)
                    continue

                # STOP MARKER: "gaat over tot de orde van de dag"
                # Dit is de formele afsluiting van ELKE motie
                if is_collecting and "gaat over tot de orde van de dag" in text.lower():
                    current_motion_text.append(text)
                    full_txt = " ".join(current_motion_text).strip()
                    motions_data.append({
                        'doc_id': file_path_cleaned,
                        'date': file_date,
                        'speaker': speaker,
                        'hoofd_thema': mapped_theme,
                        'text': full_txt
                    })
                    # Reset state
                    is_collecting = False
                    current_motion_text = []
                    continue

                # COLLECTIE: Voeg tekst toe als we binnen een motie-blok zitten
                if is_collecting:
                    # Sla de 'gehoord de beraadslaging' note niet over, die hoort bij de tekst
                    current_motion_text.append(text)

        return motions_data
    except Exception as e:
        print(f"Error in {file_path}: {e}")
        return []


# Loop door de mappen (2014-2022) [cite: 39]
all_results = []
for year in range(2014, 2023):
    year_path = os.path.join(BASE_PATH, str(year))
    if os.path.exists(year_path):
        print(f"Processing year: {year}")
        files = [f for f in os.listdir(year_path) if f.endswith('.xml')]
        for file in sorted(files):
            all_results.extend(extract_strict_motions(
                os.path.join(year_path, file)))


if all_results:
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(
        f"Gereed! {len(df)} moties succesvol geëxtraheerd naar {OUTPUT_FILE}")
else:
    print("Geen data gevonden. Controleer je BASE_PATH.")
