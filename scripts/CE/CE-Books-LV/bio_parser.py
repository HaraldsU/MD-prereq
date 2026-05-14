#!/usr/bin/env python3
"""
Parse BIO_VSK_MG.pdf into structured JSON with sections and mapped index concepts.
Uses pdftotext (poppler) for fast extraction.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

PDF_PATH = Path("~/Downloads/prereq/datasets/CE-Books-LV/BIO_VSK_MG.pdf").expanduser()
OUTPUT_PATH = "outputs/bio_vsk_mg.json"

# ── TOC entries: (section_number, section_name, start_page) ──
TOC_RAW = [
    ("1", "Ievads", 6),
    ("2", "Dzīvības izcelšanās", 8),
    ("3", "Sarežģīta specifiska uzbūve — šūna", 11),
    ("3.1", "Šūnu teorija", 13),
    ("3.2", "Šūnu līdzība un atšķirības", 14),
    ("3.3", "Šūnas ķīmija", 15),
    ("3.4", "Prokariotu šūna", 18),
    ("3.5", "Eikariotu šūna", 19),
    ("3.6", "Šūnu diferenciācija", 39),
    ("3.7", "Audi", 40),
    ("3.8", "Šūnu pētīšanas metodes", 46),
    ("4", "Vielu un enerģijas maiņa — vielmaiņa", 51),
    ("4.1", "Fermenti", 53),
    ("4.2", "Katabolisms un anabolisms", 55),
    ("4.3", "Fotosintēze", 57),
    ("4.4", "Enerģija", 58),
    ("4.5", "Olbaltummaiņa, taukmaiņa un ogļhidrātmaiņa", 59),
    ("5", "Nemainīga iekšējā vide — homeostāze", 65),
    ("5.1", "Iekšējā vide", 67),
    ("5.2", "Atgriezeniskā saite", 68),
    ("5.3", "Termoregulācija", 69),
    ("5.4", "Osmotiskā regulācija", 72),
    ("5.5", "Glikozes līmeņa regulēšana asinīs", 73),
    ("5.6", "Augu adaptācijas", 74),
    ("6", "Sarežģītas dzīvības norises — fizioloģiskie procesi", 77),
    ("6.1", "Elpošana un gāzu maiņa", 79),
    ("6.2", "Barošanās un gremošana", 86),
    ("6.3", "Vielu transports un imunitāte", 95),
    ("6.4", "Balsts un kustības", 114),
    ("6.5", "Vielu izvadīšana un osmotiskā regulācija", 125),
    ("6.6", "Organisma darbības un iekšējās vides regulācija", 131),
    ("6.7", "Vides kairinājumu uztveršana un analīze", 150),
    ("7", "Vairošanās un attīstība — reprodukcija", 161),
    ("7.1", "Bezdzimumvairošanās", 163),
    ("7.2", "Dzimumvairošanās", 166),
    ("7.3", "Šūnu dalīšanās", 167),
    ("7.4", "Šūnas dzīves cikls", 175),
    ("7.5", "Gametoģenēze", 176),
    ("7.6", "Embrionālā attīstība", 185),
    ("8", "Iedzimtība un mainība — ģenētika", 193),
    ("8.1", "Ģenētikas vēsture", 195),
    ("8.2", "Iedzimtība", 197),
    ("8.3", "Molekulārā ģenētika", 207),
    ("8.4", "Gēnu mijiedarbība", 220),
    ("8.5", "Gēnu saistība", 229),
    ("8.6", "Ģenealoģijas metode", 234),
    ("8.7", "Mainība", 248),
    ("8.8", "Mutācijas", 252),
    ("8.9", "Populāciju ģenētika", 263),
    ("8.10", "Biotehnoloģija", 271),
    ("9", "Specifiska uzvedība — etoloģija", 283),
    ("9.1", "Iedzimtās uzvedības formas", 285),
    ("9.2", "Iegūtās uzvedības formas", 288),
    ("9.3", "Sabiedriskās uzvedības formas", 291),
    ("10", "Mijiedarbība ar citiem organismiem — ekoloģija", 299),
    ("10.1", "Populāciju ekoloģija", 301),
    ("10.2", "Biocenožu ekoloģija", 308),
    ("10.3", "Ekosistēmas ekoloģija", 313),
    ("10.4", "Dabas aizsardzība", 326),
    ("10.5", "Lauksaimnieciskā ražošana", 330),
    ("10.6", "Bioloģiskās daudzveidības saglabāšana", 333),
    ("10.7", "Dabas piesārņošana", 348),
    ("11", "Pārmaiņas laikā — evolūcija", 357),
    ("11.1", "Evolūcijas teorijas", 359),
    ("11.2", "Dabiskā izlase", 361),
    ("11.3", "Neodarvinisma teorijas", 363),
    ("11.4", "Evolūcijas pierādījumi", 366),
    ("11.5", "Mikropasaules evolūcija", 368),
]

GLOSSARY_START = 376
INDEX_START = 385
LAST_PAGE = 392


def get_text(start_pg, end_pg):
    """Extract text for book pages [start_pg, end_pg] using pdftotext."""
    r = subprocess.run(
        ['pdftotext', '-f', str(start_pg), '-l', str(end_pg), PDF_PATH, '-'],
        capture_output=True, text=True
    )
    return r.stdout


def parse_page_numbers(s):
    """Parse page number strings like '14, 19', '171–174', '253, 255, 256'."""
    pages = []
    parts = re.split(r',', s)
    for part in parts:
        part = part.strip()
        range_match = re.match(r'(\d+)\s*[–\-]\s*(\d+)', part)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            pages.extend(range(start, end + 1))
        elif re.match(r'^\d+$', part):
            pages.append(int(part))
    return pages


# ── Manual overrides for main entries ──
# When a main entry name in the index should be renamed.
# Key: raw entry text → corrected concept name
MAIN_ENTRY_OVERRIDES = {
    # add overrides here for entries that need manual renaming
    # (comma-modifier entries like "difūzija, atvieglotā" are now handled automatically)
}

# ── Manual overrides for all — and — — sub-entries ──
# Key: (parent_entry, depth, sub_word) → resolved concept name
# Fill in "TODO" entries with the correct Latvian concept name
SUB_ENTRIES = {
    ("aminoskābes", 1, "aizstājamās"): "Aizstājamās aminoskābes",
    ("aminoskābes", 1, "neaizstājamās"): "Neaizstājamās aminoskābes",
    ("atgriezeniskā saite, negatīvā", 1, "pozitīvā"): "Pozitīvā atgriezeniskā saite",
    ("augsnes erozija", 1, "piesārņojums"): "Augsnes piesārņojums",
    ("dabas aizsardzība", 1, "aizsardzības dokumenti"): "Dabas aizsardzības dokumenti",
    ("dabas aizsardzība", 1, "liegums"): "Dabas liegums",
    ("dabas aizsardzība", 1, "parks"): "Dabas parks",
    ("dabas aizsardzība", 1, "pieminekļi"): "Dabas pieminekļi",
    ("dabas aizsardzība", 1, "piesārņošana"): "Dabas piesārņošana",
    ("dabas rezervāts", 2, "Grīņu"): "Grīņu dabas rezervāts",
    ("dabas rezervāts", 2, "Krustkalnu"): "Krustkalnu dabas rezervāts",
    ("dabas rezervāts", 2, "Moricsalas"): "Moricsalas dabas rezervāts",
    ("dabas rezervāts", 2, "Teiču"): "Teiču dabas rezervāts",
    ("DNS", 1, "\u201cpirkstu nospiedumi\u201d"): "DNS pirkstu nospiedumi",
    ("DNS", 1, "replikācija"): "DNS replikācija",
    ("ekosistēmas biomasas piramīda", 1, "enerģijas piramīda"): "Ekosistēmas enerģijas piramīda",
    ("ekosistēmas biomasas piramīda", 1, "lieluma piramīda"): "Ekosistēmas lieluma piramīda",
    ("ekosistēmas biomasas piramīda", 1, "produktivitāte"): "Ekosistēmas produktivitāte",
    ("ekosistēmas biomasas piramīda", 1, "struktūra"): "Ekosistēmas struktūra",
    ("elpošanas orgāni", 1, "regulācija"): "Elpošanas regulācija",
    ("endoplazmatiskais tīkls", 2, "gludais"): "Gludais endoplazmatiskais tīkls",
    ("endoplazmatiskais tīkls", 2, "graudainais"): "Graudainais endoplazmatiskais tīkls",
    ("erozija", 1, "agrotehniskā"): "Agrotehniskā erozija",
    ("erozija", 1, "augsnes"): "Augsnes erozija",
    ("gēnu banka", 1, "ekspresija"): "Gēnu ekspresija",
    ("gēnu banka", 1, "mijiedarbība"): "Gēnu mijiedarbība",
    ("gēnu banka", 1, "mutācijas"): "Gēnu mutācijas",
    ("gēnu banka", 1, "saistība"): "Gēnu saistība",
    ("gēnu banka", 1, "terapija"): "Gēnu terapija",
    ("gremošanas fermenti", 1, "orgānu sistēma"): "Gremošanas orgānu sistēma",
    ("ģenētiskā daudzveidība", 1, "inženierija"): "Ģenētiskā inženierija",
    ("hromosomu aberācijas", 1, "mutācijas"): "Hromosomu mutācijas",
    ("hromosomu aberācijas", 1, "segregācija"): "Hromosomu segregācija",
    ("lizosoma, primārā", 1, "sekundārā"): "Sekundārā lizosoma",
    ("muskuļi", 1, "gludie"): "Gludie muskuļi",
    ("muskuļi", 1, "skeleta"): "Skeleta muskuļi",
    ("mutācijas", 1, "ģeneratīvās"): "Ģeneratīvās mutācijas",
    ("mutācijas", 1, "somatiskās"): "Somatiskās mutācijas",
    ("nervu impulss", 1, "sistēma"): "Nervu sistēma",
    ("nervu impulss", 2, "centrālā"): "Centrālā nervu sistēma",
    ("nervu impulss", 2, "parasimpātiskā"): "Parasimpātiskā nervu sistēma",
    ("nervu impulss", 2, "perifērā"): "Perifērā nervu sistēma",
    ("nervu impulss", 2, "simpātiskā"): "Simpātiskā nervu sistēma",
    ("nervu impulss", 2, "somatiskā"): "Somatiskā nervu sistēma",
    ("nervu impulss", 2, "veģetatīvā"): "Veģetatīvā nervu sistēma",
    ("plazmatiskā membrāna", 2, "pārveidojumi"): "Plazmatiskās membrānas pārveidojumi",
    ("salīdzinošā anatomija", 1, "embrioloģija"): "Salīdzinošā embrioloģija",
    ("sēņu audzēšana", 1, "dzīves cikls"): "Sēņu dzīves cikls",
    ("sēņu audzēšana", 1, "hifas"): "Sēņu hifas",
    ("sēņu audzēšana", 1, "klasifikācija"): "Sēnu klasifikācija",
    ("sēņu audzēšana", 1, "valsts"): "Sēņu valsts",
    ("sirds toņi", 1, "uzbūve"): "Sirds uzbūve",
    ("smadzeņu garoza", 1, "puslodes"): "Smadzeņu puslodes",
    ("smadzeņu garoza", 1, "saiklis"): "Smadzeņu saiklis",
    ("smadzeņu garoza", 1, "tilts"): "Smadzeņu tilts",
    ("ūdens", 1, "aprite"): "Ūdens aprite",
    ("ūdens", 1, "erozija"): "Ūdens erozija",
    ("ūdens", 1, "pārvietošanās augā"): "Ūdens pārvietošanās augā",
    ("ūdens", 1, "piesārņojums"): "Ūdens piesārņojums",
}


def parse_index():
    """Parse the Alfabētiskais rādītājs into a list of (concept, [pages]).
    
    Normal entries are parsed automatically.
    Sub-entries (lines starting with —) are resolved via the SUB_ENTRIES dictionary.
    Any entry with value "TODO" is skipped with a warning.
    """
    raw_text = get_text(INDEX_START, LAST_PAGE)

    entries = []
    current_main = None
    skipped = []

    for line in raw_text.split("\n"):
        line = line.strip()
        if not line or line == "ALFABĒTISKAIS RĀDĪTĀJS":
            continue
        if re.match(r'^[A-ZĀČĒĢĪĶĻŅŠŪŽ]$', line):
            continue
        if re.match(r'^\d{1,3}$', line):
            continue

        # Sub-entry starting with —
        if line.startswith("—"):
            depth = 0
            tmp = line
            while tmp.startswith("—"):
                depth += 1
                tmp = tmp[1:].lstrip(" ")
            m = re.match(r'^(.+?)\s+([\d,\s–\-]+)$', tmp)
            if not m:
                continue
            sub_word = m.group(1).strip()
            pages = parse_page_numbers(m.group(2))

            key = (current_main, depth, sub_word)
            concept = SUB_ENTRIES.get(key)

            if concept is None:
                # Key not in dictionary at all — print it so user can add it
                skipped.append(key)
            elif concept == "TODO":
                skipped.append(key)
            else:
                entries.append((concept, pages))
            continue

        # Normal entry
        m = re.match(r'^(.+?)\s+([\d,\s–\-]+)$', line)
        if m:
            raw_concept = m.group(1).strip()
            pages = parse_page_numbers(m.group(2))
            current_main = raw_concept

            # Check for comma-modifier pattern: "concept, modifier"
            # e.g. "difūzija, atvieglotā 22" → two concepts:
            #   1. "difūzija"
            #   2. "atvieglotā difūzija"
            cm = re.match(r'^(.+?),\s+([a-zāčēģīķļņšūžA-ZĀČĒĢĪĶĻŅŠŪŽ]\S*)$', raw_concept)
            if cm:
                base = cm.group(1).strip()
                modifier = cm.group(2).strip()
                entries.append((base, pages))
                entries.append((modifier + " " + base, pages))
            else:
                # Apply main entry override if exists
                concept = MAIN_ENTRY_OVERRIDES.get(raw_concept, raw_concept)
                entries.append((concept, pages))
        else:
            current_main = line

    if skipped:
        print(f"  WARNING: {len(skipped)} sub-entries skipped (TODO or missing):")
        for k in skipped:
            print(f"    {k}")

    return entries


def page_to_section_idx(page):
    """Given a book page number, find the TOC section index it belongs to."""
    for i in range(len(TOC_RAW) - 1, -1, -1):
        if page >= TOC_RAW[i][2]:
            # Check upper bound
            if i + 1 < len(TOC_RAW):
                if page < TOC_RAW[i + 1][2]:
                    return i
            else:
                if page < GLOSSARY_START:
                    return i
            # If page >= next section start, keep searching backwards?
            # No - if page >= this section and >= next section, 
            # the later match (higher i) is correct. But we're going backwards,
            # so first match is the latest section that starts <= page.
            # We need to verify page < next section start.
            # If not, this page is actually in a later section, but we already
            # passed it. This shouldn't happen since we go from end to start.
            pass
    return None


def main():
    print("Extracting section texts...", flush=True)

    sections = []
    for i, (sec_num, sec_name, start_pg) in enumerate(TOC_RAW):
        if i + 1 < len(TOC_RAW):
            end_pg = TOC_RAW[i + 1][2] - 1
        else:
            end_pg = GLOSSARY_START - 1

        text = get_text(start_pg, end_pg)
        # Clean up form feed characters
        text = text.replace("\f", "\n").strip()

        sections.append({
            "section_number": sec_num,
            "section_name": sec_name,
            "section_text": text,
            "gtc": []
        })
        print(f"  {sec_num}: {sec_name} ({end_pg - start_pg + 1} pages, {len(text)} chars)", flush=True)

    # ── Parse index ──
    print("\nParsing Alfabētiskais rādītājs...", flush=True)
    index_entries = parse_index()
    print(f"  Found {len(index_entries)} index entries", flush=True)

    # Map concepts to sections
    mapping_count = 0
    for concept, pages in index_entries:
        matched = set()
        for pg in pages:
            idx = page_to_section_idx(pg)
            if idx is not None:
                matched.add(idx)
        for sec_idx in matched:
            sections[sec_idx]["gtc"].append(concept)
            mapping_count += 1

    print(f"  Mapped {mapping_count} concept-section pairs", flush=True)

    # Convert gtc to comma-separated strings
    for sec in sections:
        # sec["gtc"] = ", ".join(sorted(sec["gtc"]))
        sec["gtc"] = sorted(sec["gtc"])

    # Write output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Written to {OUTPUT_PATH}", flush=True)
    print(f"Sections: {len(sections)}", flush=True)

    # Summary
    for sec in sections:
        # gc = len(sec["gtc"].split(", ")) if sec["gtc"] else 0
        gc = len(sec["gtc"])
        print(f"  {sec['section_number']:>5} | {sec['section_name'][:50]:<50} | {gc:>3} concepts", flush=True)


if __name__ == "__main__":
    main()
