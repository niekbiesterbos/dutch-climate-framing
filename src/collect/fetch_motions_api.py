import requests
import pandas as pd
import urllib.parse

# ----------------------------------------
# Configuratie
# ----------------------------------------

BASE_URL = "https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0/Document"
OUTPUT_FILE = "all_motions_2008_2025.csv"

SELECT = "Id,Titel,Datum,Soort,DocumentNummer"
EXPAND = "Zaak($expand=ZaakActor)"


# ----------------------------------------
# Hulpfuncties
# ----------------------------------------

def build_url(base: str, filter_expr: str, select_fields: str, expand_expr: str) -> str:
    params = {
        "$filter": filter_expr,
        "$select": select_fields,
        "$expand": expand_expr
    }
    return base + "?" + urllib.parse.urlencode(params, safe="(),': ")


def fetch_all_motions():
    """Haalt alle moties op binnen de opgegeven datumbereik met automatische paginatie."""
    filter_expr = (
        "Verwijderd eq false and Soort eq 'Motie' "
    )

    url = build_url(BASE_URL, filter_expr, SELECT, EXPAND)
    results = []

    while url:
        print(f"Ophalen: {url}")
        r = requests.get(url, timeout=120)
        if not r.ok:
            print(f"Fout {r.status_code}: {r.text[:300]}...")
            break
        data = r.json()
        results.extend(data.get("value", []))
        url = data.get("@odata.nextLink")
    return results


def extract_indieners_en_fracties(doc):
    """Extract all indieners and medeindieners with names and factions."""
    zaken = doc.get("Zaak")
    if not zaken:
        return []
    if isinstance(zaken, dict):
        zaken = [zaken]

    all_actors = []
    for zaak in zaken:
        actors = zaak.get("ZaakActor")
        if not actors:
            continue
        if isinstance(actors, dict):
            actors = [actors]
        for actor in actors:
            rel = (actor.get("Relatie") or "").lower()
            if rel in ("indiener", "medeindiener"):
                all_actors.append({
                    "Naam": actor.get("ActorNaam"),
                    "Fractie": actor.get("ActorFractie"),
                    "Relatie": actor.get("Relatie")
                })
    return all_actors


def construct_motie_url(doc):
    """Maak geldige motie-URL met id=Zaak.Nummer en did=DocumentNummer."""
    zaken = doc.get("Zaak")
    if not zaken:
        return None
    zaak = zaken[0] if isinstance(zaken, list) else zaken
    zaaknummer = zaak.get("Nummer")
    did = doc.get("DocumentNummer")
    if zaaknummer and did:
        return f"https://www.tweedekamer.nl/kamerstukken/moties/detail?id={zaaknummer}&did={did}"
    return None


# ----------------------------------------
# Hoofdlogica
# ----------------------------------------

def main():
    docs = fetch_all_motions()
    if not docs:
        print("Geen resultaten — controleer IP-registratie of query-filter.")
        return

    rows = []
    for d in docs:
        actors = extract_indieners_en_fracties(d)
        fracties = sorted({a["Fractie"] for a in actors if a["Fractie"]})
        url = construct_motie_url(d)
        zaak = d.get("Zaak", [{}])[0] if d.get("Zaak") else {}
        rows.append({
            "Id": d.get("Id"),
            "Titel": d.get("Titel"),
            "Datum": d.get("Datum"),
            "Indieners": actors,
            "Fracties": "; ".join(fracties) if fracties else None,
            "ZaakNummer": zaak.get("Nummer"),
            "ZaakTitel": zaak.get("Titel"),
            "ZaakOnderwerp": zaak.get("Onderwerp"),
            "Kabinetsappreciatie": zaak.get("Kabinetsappreciatie"),
            "Url": url,
        })

    if not rows:
        print("Geen moties gevonden.")
        return

    df = pd.DataFrame(rows)
    df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce")
    df.drop_duplicates(subset=["Url", "Id"], inplace=True)
    df.sort_values("Datum", inplace=True)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n✅ {len(df)} unieke moties opgeslagen in {OUTPUT_FILE}")


def inspect_sample():
    filter_expr = "Verwijderd eq false and Soort eq 'Motie'"
    url = build_url(BASE_URL, filter_expr, SELECT, EXPAND)
    r = requests.get(url, timeout=60)
    data = r.json()
    docs = data.get("value", [])

    for doc in docs[:5]:
        import json
        print(json.dumps(doc, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
