#!/usr/bin/env python3
"""
Parse FIZ_12_MG.pdf into structured JSON with sections and mapped index concepts.
Uses pdftotext (poppler) for fast extraction.
"""

import json
import re
import subprocess
import sys

PDF_PATH = "/home/dust/Downloads/prereq/datasets/CE-Books-LV/FIZ_12_MG.pdf"
OUTPUT_PATH = "outputs/fiz_12_mg.json"

# ── TOC entries: (section_number, section_name, start_page) ──
# Extracted from SATURS (pages 3-4). Sections from "Ievads" equivalent
# (first chapter) to before "Fizikas terminu skaidrojums" (glossary).
# Note: This book doesn't have an "Ievads" — it starts with chapter 1.
# Sub-sections like "Kopsavilkums" and "Uzdevumi" are NOT separate sections.

TOC_RAW = [
    ("1.1", "Maiņstrāvas iegūšana", 8),
    ("1.2", "Maiņstrāvas raksturlielumu momentānās vērtības", 11),
    ("1.3", "Maiņstrāvas raksturlielumu efektīvās vērtības", 14),
    ("1.4", "Enerģijas pārvērtības maiņstrāvas ķēdē", 15),
    ("1.5", "Aktīvā pretestība maiņstrāvas ķēdē", 16),
    ("1.6", "Induktīvā pretestība maiņstrāvas ķēdē", 18),
    ("1.7", "Kapacitīvā pretestība maiņstrāvas ķēdē", 20),
    ("1.8", "Pilnā pretestība. Oma likums maiņstrāvas ķēdei", 23),
    ("1.9", "Maiņstrāvas pilnā jauda. Jaudas koeficients", 26),
    ("1.10", "Trīsfāzu maiņstrāva", 28),
    ("1.11", "Transformatori. Elektroenerģijas pārvades līnijas", 29),
    ("1.12", "Elektroenerģija Latvijā", 32),
    ("2.1", "Svārstību kontūrs", 48),
    ("2.2", "Nerimstošas elektriskās svārstības", 50),
    ("2.3", "Elektromagnētisko viļņu iegūšana", 51),
    ("2.4", "Elektromagnētiskais šķērsvilnis", 52),
    ("2.5", "Elektromagnētisko viļņu ātrums", 54),
    ("2.6", "Elektromagnētisko viļņu skala", 56),
    ("2.7", "Radiosignāla pārraide un uztveršana", 58),
    ("2.8", "Radioviļņu izplatīšanās ap Zemi", 61),
    ("2.9", "Kosmiskais radiostarojums. Radioteleskopi", 63),
    ("2.10", "Ultraīsviļņi. Televīzija. Radiolokācija", 65),
    ("2.11", "Satelītu sakari. Globālās pozicionēšanas sistēmas", 67),
    ("3.1", "Gaismas elektromagnētiskie viļņi", 80),
    ("3.2", "Gaismas stari. Heigensa princips", 82),
    ("3.3", "Gaismas atstarošanās", 84),
    ("3.4", "Gaismas laušana", 86),
    ("3.5", "Gaismas pilnā iekšējā atstarošanās. Gaismas vadi", 88),
    ("3.6", "Gaismas dispersija. Staru gaita prizmā", 90),
    ("3.7", "Varavīksne", 92),
    ("4.1", "Gaismas avota stiprums. Gaismas plūsma", 108),
    ("4.2", "Apgaismojums", 110),
    ("4.3", "Elektriskie gaismas avoti. Apgaismojums darba vietā", 112),
    ("4.4", "Ēnas. Aptumsumi", 114),
    ("4.5", "Attēla veidošanās. Attēls plakanā spogulī", 116),
    ("4.6", "Sfēriski spoguļi", 119),
    ("4.7", "Apgaismojuma un attēlu iegūšana ar sfēriskiem spoguļiem", 120),
    ("4.8", "Sfēriskas lēcas", 122),
    ("4.9", "Attēlu iegūšana ar sfēriskām lēcām", 124),
    ("4.10", "Lēcas optiskais stiprums. Lēcu kļūdas", 126),
    ("4.11", "Cilvēka acs. Redze", 128),
    ("4.12", "Acs optiskie defekti un to korekcija", 129),
    ("4.13", "Lupa. Mikroskops. Tālskatis", 131),
    ("4.14", "Teleskopi. Kosmosa izpēte ar teleskopiem", 134),
    ("5.1", "Gaismas interference. Koherenti gaismas viļņi", 150),
    ("5.2", "Interferences maksimumu un minimumu nosacījumi", 152),
    ("5.3", "Interference plānās kārtiņās. Ņūtona gredzeni. Dzidrinātā optika", 154),
    ("5.4", "Interferometri", 157),
    ("5.5", "Gaismas difrakcija. Heigensa – Frenela princips", 158),
    ("5.6", "Gaismas difrakcija spraugā", 160),
    ("5.7", "Difrakcijas režģis", 162),
    ("5.8", "Hologrāfija", 164),
    ("5.9", "Polarizācija. Optiski aktīvas vielas", 166),
    ("6.1", "Gaismas kvanti. Planka konstante", 180),
    ("6.2", "Fotoefekts", 182),
    ("6.3", "Einšteina vienādojums fotoefektam", 184),
    ("6.4", "Gaismas spiediens", 185),
    ("6.5", "Emisijas spektri. Atoma enerģijas līmeņi", 186),
    ("6.6", "Spontānais un inducētais starojums", 189),
    ("6.7", "Vielas absorbcijas spektri. Atoma jonizācija", 191),
    ("6.8", "Lāzeri", 192),
    ("6.9", "Lāzeru iekārtas", 194),
    ("6.10", "Siltumstarojums", 195),
    ("6.11", "Luminiscence", 197),
    ("6.12", "Rentgenstarojums", 198),
    ("7.1", "No kā sastāv atoms?", 211),
    ("7.2", "Atoma kodola atklāšana", 213),
    ("7.3", "Atoma planetārais modelis. Bora teorija", 214),
    ("7.4", "Kvantu mehānika. Orbitālais kvantu skaitlis", 216),
    ("7.5", "Magnētiskais kvantu skaitlis. Magnētiskās mijiedarbības atomā", 217),
    ("7.6", "Elektrona spins", 219),
    ("7.7", "Pauli princips. Elektronu konfigurācija vairākelektronu atomos", 221),
    ("7.8", "De Brojī vilnis. Elektrona viļņu daba", 224),
    ("7.9", "Elektrona mākonis atomā", 226),
    ("7.10", "Atoma orbitāles", 228),
    ("8.1", "Kodoldaļiņas — protoni un neitroni. Izotopi", 238),
    ("8.2", "Kodolspēks", 240),
    ("8.3", "Kodola saites enerģija", 240),
    ("8.4", "Alfa un beta radioaktivitāte", 243),
    ("8.5", "Gamma starojums", 245),
    ("8.6", "Pussabrukšanas periods. Radioaktīvās sabrukšanas likums", 247),
    ("8.7", "Jonizējošā starojuma aktivitāte un absorbēšanas doza. Daļiņu reģistrācija", 249),
    ("8.8", "Radioaktīvo izotopu izmantošana", 251),
    ("8.9", "Kodolreakcijas", 254),
    ("8.10", "Kodolu dalīšanās reakcijas. Vadāma ķēdes reakcija", 255),
    ("8.11", "Kodolreaktors. Kodolenerģētika", 257),
    ("8.12", "Kodolsintēzes reakcijas", 259),
    ("9.1", "Planētas un zvaigznes", 276),
    ("9.2", "Galaktikas un Visums", 279),
    ("9.3", "Visuma evolūcija. Habla likums", 280),
    ("9.4", "Zvaigžņu evolūcija", 283),
    ("9.5", "Visuma apgūšanas perspektīvas", 284),
    ("9.6", "Lielu ātrumu un enerģiju fizika", 286),
    ("9.7", "Elementārdaļiņas. Fermioni un bozoni. Daļiņas un antidaļiņas", 289),
    ("9.8", "Fundamentālas mijiedarbības", 291),
    ("9.9", "Fundamentālās daļiņas. Kvarki un leptoni", 293),
    ("9.10", "Mijiedarbību nesējkvanti", 295),
    ("9.11", "Neitrīno un kosmiskie stari", 297),
]

GLOSSARY_START = 313  # Fizikas terminu skaidrojums
INDEX_START = 319     # Alfabētiskais rādītājs
LAST_PAGE = 320


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


def parse_index():
    """Parse the Alfabētiskais rādītājs into a list of (concept, [pages]).

    Normal entries are parsed automatically.
    Sub-entries (lines starting with –) are prepended to the parent concept.
    E.g. "Atoma modelis" + "– planetārais" → "planetārais Atoma modelis"
    """
    raw_text = get_text(INDEX_START, LAST_PAGE)

    entries = []
    current_main = None

    for line in raw_text.split("\n"):
        line = line.strip()
        if not line or line == "ALFABĒTISKAIS RĀDĪTĀJS":
            continue
        if re.match(r'^[A-ZĀČĒĢĪĶĻŅŠŪŽ]$', line):
            continue
        if re.match(r'^\d{1,3}$', line):
            continue

        # Sub-entry starting with – (en-dash)
        if line.startswith("–"):
            tmp = line.lstrip("– ").strip()
            m = re.match(r'^(.+?)\s+([\d,\s–\-]+)$', tmp)
            if not m or not current_main:
                continue
            sub_word = m.group(1).strip()
            pages = parse_page_numbers(m.group(2))
            concept = sub_word + " " + current_main
            entries.append((concept, pages))
            continue

        # Normal entry
        m = re.match(r'^(.+?)\s+([\d,\s–\-]+)$', line)
        if m:
            raw_concept = m.group(1).strip()
            pages = parse_page_numbers(m.group(2))
            current_main = raw_concept
            entries.append((raw_concept, pages))
        else:
            current_main = line

    return entries


def page_to_section_idx(page):
    """Given a book page number, find the TOC section index it belongs to."""
    for i in range(len(TOC_RAW) - 1, -1, -1):
        if page >= TOC_RAW[i][2]:
            if i + 1 < len(TOC_RAW):
                if page < TOC_RAW[i + 1][2]:
                    return i
            else:
                if page < GLOSSARY_START:
                    return i
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

        # When two sections start on the same page, end_pg < start_pg.
        # Include the shared page in both sections.
        if end_pg < start_pg:
            end_pg = start_pg

        text = get_text(start_pg, end_pg)
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

    # Sort gtc lists
    for sec in sections:
        sec["gtc"] = sorted(sec["gtc"])

    # Write output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Written to {OUTPUT_PATH}", flush=True)
    print(f"Sections: {len(sections)}", flush=True)

    for sec in sections:
        gc = len(sec["gtc"])
        print(f"  {sec['section_number']:>5} | {sec['section_name'][:50]:<50} | {gc:>3} concepts", flush=True)


if __name__ == "__main__":
    main()
