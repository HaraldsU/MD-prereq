#!/usr/bin/env python3
"""
Parse KIM_12_MG.pdf into structured JSON with sections and mapped index concepts.
Uses pdftotext (poppler) for fast extraction.
"""

import json
import re
import subprocess
from pathlib import Path

PDF_PATH = Path("~/Downloads/prereq/datasets/CE-Books-LV/KIM_12_MG.pdf").expanduser()
OUTPUT_PATH = "/home/dust/Downloads/prereq/scripts/CE-Books-LV/outputs/chem_vsk_mg.json"

# ── TOC entries: (section_number, section_name, start_page) ──
# No parent sections (1, 2, 3...), no "Uzdevumi", no "Eksperimenti".
# The ToC has a typo: "6.1" appears twice; the second one is actually "6.2".

TOC_RAW = [
    # Chapter 1
    ("1.1", "Ogļūdeņraži un to atvasinājumi", 8),
    ("1.2", "Ogļūdeņražu hidroksilatvasinājumu un karbonilatvasinājumu nomenklatūra un izomērija", 10),
    ("1.3", "Ūdeņraža saišu ietekme uz ogļūdeņražu hidroksilatvasinājumu īpašībām", 15),
    ("1.4", "Spirtu un fenolu aizvietošanas un atšķelšanas reakcijas", 18),
    ("1.4.1", "Ogļūdeņražu hidroksilatvasinājumu un karbonilatvasinājumu izmantošana", 21),
    ("1.5", "Organisko vielu pārvērtības oksidēšanās–reducēšanās reakcijās. No spirta līdz karbonskābei", 22),
    ("1.5.1", "Vai etanola lietošana ikdienā ir problēma?", 25),
    # Chapter 2
    ("2.1", "Karbonskābes — ogļūdeņražu atvasinājumi", 34),
    ("2.2", "Karbonskābes, izomērija un nomenklatūra", 35),
    ("2.3", "Karbonskābes. Fizikālās un ķīmiskās īpašības", 39),
    ("2.4", "Aizvietotās karbonskābes", 42),
    ("2.5", "Karbonskābju funkcionālie atvasinājumi", 45),
    ("2.5.1", "Karbonskābju un to atvasinājumu izmantošana", 49),
    # Chapter 3
    ("3.1", "Dabas vielas", 56),
    ("3.2", "Tauki un eļļas", 57),
    ("3.2.1", "Lipīdi", 60),
    ("3.3", "Ogļhidrāti", 61),
    ("3.3.1", "Saldā dzīve", 66),
    ("3.4", "Olbaltumvielas", 67),
    ("3.5", "Nukleīnskābes", 71),
    # Chapter 4
    ("4.1", "Materiālu daudzveidība", 78),
    ("4.2", "Sintētisko organisko polimēru raksturlielumi un iegūšana", 80),
    ("4.3", "Polimērmateriālu īpašības", 83),
    ("4.3.1", "Biodegradējamie polimēri", 87),
    ("4.4", "Ziepes un sintētiskās virsmaktīvās vielas", 89),
    ("4.4.1", "Virsmaktīvās vielas mazgāšanas un kosmētikas līdzekļu sastāvā", 93),
    # Chapter 5
    ("5.1", "Ķīmijas tehnoloģijas", 98),
    ("5.1.1", "Ķīmijas un farmācijas rūpniecība Latvijā", 99),
    ("5.2", "Etanola ražošanas tehnoloģijas", 100),
    ("5.3", "Celulozes ražošanas tehnoloģijas", 103),
    ("5.4", "Silikātu tehnoloģijas", 105),
    ("5.5", "Vides tehnoloģijas", 109),
    ("5.5.1", "Vai Latvijā šķiro atkritumus?", 115),
    ("5.6", "Aprēķini ķīmiskajā rūpniecībā", 116),
    # Chapter 6
    ("6.1", "Vielu savstarpējā saikne", 122),
    ("6.2", "Globālās, reģionālās un lokālās vides problēmas", 125),
    ("6.2.1", "Kas ir noturīgie organiskie piesārņotāji (NOP)?", 130),
    ("6.3", "Ilgtspējīga attīstība", 132),
    ("6.3.1", "Ķīmija profesijas izvēlei", 136),
]

# Pages to exclude from section text extraction:
# "Svarīgākais nodaļā" pages, "Uzdevumi" pages, "Eksperimenti" pages
# These appear at the end of each chapter.
EXCLUDED_PAGES = {
    # Chapter 1: Svarīgākais 28, Uzdevumi 29-30, Eksperimenti 31-32
    28, 29, 30, 31, 32,
    # Chapter 2: Svarīgākais 51, Uzdevumi 52, Eksperimenti 53-54
    51, 52, 53, 54,
    # Chapter 3: Svarīgākais 74, Uzdevumi 75, Eksperimenti 76
    74, 75, 76,
    # Chapter 4: Svarīgākais 94, Uzdevumi 95, Eksperimenti 96
    94, 95, 96,
    # Chapter 5: Svarīgākais 118, Uzdevumi 119, Eksperimenti 120
    118, 119, 120,
    # Chapter 6: Svarīgākais 138, Uzdevumi 139
    138, 139,
}

GLOSSARY_START = 140  # Pielikumi start (then Terminu skaidrojums at 155)
INDEX_START = 157
LAST_PAGE = 158


def get_text(start_pg, end_pg, exclude_pages=None):
    """Extract text for book pages [start_pg, end_pg] using pdftotext.
    Optionally skip specific pages."""
    texts = []
    for pg in range(start_pg, end_pg + 1):
        if exclude_pages and pg in exclude_pages:
            continue
        r = subprocess.run(
            ['pdftotext', '-f', str(pg), '-l', str(pg), PDF_PATH, '-'],
            capture_output=True, text=True
        )
        if r.stdout.strip():
            texts.append(r.stdout)
    return "\n".join(texts)


def get_text_simple(start_pg, end_pg):
    """Extract text without exclusions (for index parsing)."""
    r = subprocess.run(
        ['pdftotext', '-f', str(start_pg), '-l', str(end_pg), PDF_PATH, '-'],
        capture_output=True, text=True
    )
    return r.stdout


def parse_page_numbers(s):
    """Parse page number strings like '14, 19', '171–174'."""
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


# ── Manual sub-entry dictionary ──
# Key: (parent_entry, sub_word) → resolved concept name
# Fill in "TODO" entries with the correct concept name.
SUB_ENTRIES = {
    ("alkohols", "pārvērtības organismā"): "Alkohola pārvērtības organismā",
    ("aminoskābes", "bipolārs jons"): "Bipolārs jons",
    ("atkritumi", "apsaimniekošana"): "Apsaimniekošanas atkritumi",
    ("atkritumi", "bīstamie"): "Bīstamie atkritumi",
    ("atkritumi", "ražošanas"): "Ražošanas atkritumi",
    ("atkritumi", "sadzīves"): "Sadzīves atkritumi",
    ("atkritumi", "šķirošana"): "Šķirošanas atkritumi",
    ("DNS", "dubultspirāle"): "DNS dubultspirāle",
    ("DNS", "telpiskā struktūra"): "DNS telpiskā struktūra",
    ("etanols", "bioetanols"): "Bioetanols",
    ("etanols", "ražošana"): "Etanola ražošana",
    ("fenoli", "formaldehīda sveķi"): "Fenola formaldehīda sveķi",
    ("izomēri", "funkcionālās grupas vietas"): "Funkcionālās grupas vietas izomērija",
    ("izomēri", "oglekļa virknes"): "Ogļekļa virknes izomēri",
    ("izomēri", "starp savienojumu klasēm"): "Izomēri starp savienojumu klasēm",
    ("karbonskābes", "funkcionālie atvasinājumi"): "Karbonskābju funkcionālie atvasinājumi",
    ("karbonskābes", "sāļi"): "Karbonskābju sāļi",
    ("karbonskābes", "vienvērtīgās aromātiskās"): "Vienvērtīgās aromātiskās karbonskābes",
    ("karbonskābes", "vienvērtīgās nepiesātinātās"): "Vienvērtīgās nepiesātinātās karbonskābes",
    ("karbonskābes", "vienvērtīgās piesātinātās"): "Vienvērtīgās piesātinātās karbonskābes",
    ("lipīdi", "fosfolipīdi"): "Fosfolipīdi",
    ("lipīdi", "steroīdi"): "Steroīdi",
    ("lipīdi", "vaski"): "Vaski",
    ("notekūdeņi", "attīrīšana"): "Notekūdeņu attīrīšana",
    ("nukleīnskābes", "pirmējā struktūra"): "Nukleīnskābes pirmējā struktūra",
    ("nukleīnskābes", "telpiskā struktūra"): "Nukleīnskābes telpiskā struktūra",
    ("ogļhidrāti", "disaharīdi"): "Disaharīdi",
    ("ogļhidrāti", "laktoze"): "Laktoze",
    ("ogļhidrāti", "maltoze"): "Maltoze",
    ("ogļhidrāti", "saharoze"): "Saharoze",
    ("ogļhidrāti", "monosaharīdi"): "Monosaharīdi",
    ("ogļhidrāti", "fruktoze"): "Fruktoze",
    ("ogļhidrāti", "glikoze"): "Glikoze",
    ("ogļhidrāti", "polisaharīdi"): "Polisaharīdi",
    ("ogļhidrāti", "celuloze"): "Celuloze",
    ("ogļhidrāti", "ciete"): "Ciete",
    ("ogļūdeņraži", "hidroksilatvasinājumi"): "Ogļūdeņražu hidroksilatvasinājumi",
    ("ogļūdeņraži", "karbonilatvasinājumi"): "Ogļūdeņražu karbonilatvasinājumi",
    ("oksidēšana", "daļēja"): "Daļēja oksidēšana",
    ("oksidēšana", "pilnīga"): "Pilnīga oksidēšana",
    ("olbaltumvielas", "ceturtējā struktūra"): "Olbaltumvielu ceturtējā struktūra",
    ("olbaltumvielas", "otrējā struktūra"): "Olbaltumvielu otrējā struktūra",
    ("olbaltumvielas", "pirmējā struktūra"): "Olbaltumvielu pirmējā struktūra",
    ("olbaltumvielas", "trešējā struktūrā"): "Olbaltumvielu trešējā struktūra",
    ("polimēri", "biodegradējamie"): "Biodegradējamie polimēri",
    ("polimēri", "termoplastiskie"): "Termoplastiskie polimēri",
    ("polimēri", "termoreaktīvie"): "Termoreaktīvie polimēri",
    ("ražošana", "celulozes"): "Celulozes ražošana",
    ("ražošana", "cementa"): "Cementa ražošana",
    ("ražošana", "etanola"): "Etanola ražošana",
    ("ražošana", "keramikas"): "Keramikas ražošana",
    ("ražošana", "stikla"): "Stikla ražošana",
    ("reakcija", "ķēdes"): "Ķēdes reakcija",
    ("reakcija", "pārziepjošanas"): "Pārziepjošanas reakcija",
    ("reakcija", "polikondensācijas"): "Polikondensācijas reakcija",
    ("reakcija", "polimerizācijas"): "Polimerizācijas reakcija",
    ("rūgšana", "alkoholiskā"): "Alkoholiskā rūgšana",
    ("rūgšana", "citronskābā"): "Citronskābā rūgšana",
    ("rūgšana", "etiķskābā"): "Etiķskābā rūgšana",
    ("rūgšana", "pienskābā"): "Pienskābā rūgšana",
    ("rūgšana", "sviestskābā"): "Sviestskābā rūgšana",
    ("spirti", "aromātiskie"): "Aromātiskie spirti",
    ("spirti", "daudzvērtīgie"): "Daudzvērtīgie spirti",
    ("spirti", "vienvērtīgie"): "Vienvērtīgie spirti",
    ("tauki", "hidrogenēšana"): "Tauku hidrogenēšana",
    ("tauki", "hidrolīze"): "Tauku hidrolīze",
    ("taukskābes", "neaizstājamās"): "Neaizstājamās taukskābes",
    ("vides problēmas", "globālas"): "Globālās vides problēmas",
    ("vides problēmas", "lokālas"): "Lokālās vides problēmas",
    ("vides problēmas", "reģionālas"): "Reģionālās vides problēmas",
    ("ziepes", "mazgājošā darbība"): "Ziepju mazgājošā darbība",
}

# Set of sub-entry words for detection (since indentation is lost in pdftotext)
# These are the raw sub-entry lines as they appear in non-layout extraction
SUB_ENTRY_LINES = set()
for (parent, sub), _ in SUB_ENTRIES.items():
    SUB_ENTRY_LINES.add(sub)


def parse_index():
    """Parse the Alfabētiskais rādītājs into a list of (concept, [pages]).

    This book uses indentation for sub-entries which is lost in pdftotext.
    We use the SUB_ENTRIES dictionary to identify and resolve sub-entries.
    """
    raw_text = get_text_simple(INDEX_START, LAST_PAGE)

    entries = []
    current_main = None
    skipped = []
    # Track if we're inside a known parent's sub-entries
    expecting_subs_for = None

    for line in raw_text.split("\n"):
        line = line.strip()
        if not line or line == "ALFABĒTISKAIS RĀDĪTĀJS":
            continue
        if re.match(r'^[A-ZĀČĒĢĪĶĻŅŠŪŽ]$', line):
            continue
        if re.match(r'^\d{1,3}$', line):
            continue
        # Skip "skat." cross-references
        if line.startswith("skat."):
            continue

        # Try to parse as "concept pages"
        m = re.match(r'^(.+?)\s+([\d,\s]+)$', line)
        if not m:
            # Could be a continuation line (e.g. "skat. ogļhidrāti")
            # or a multi-line entry — skip
            continue

        concept = m.group(1).strip()
        pages = parse_page_numbers(m.group(2))

        # Check if this is a known sub-entry
        if current_main and (current_main, concept) in SUB_ENTRIES:
            resolved = SUB_ENTRIES[(current_main, concept)]
            if resolved == "TODO":
                skipped.append((current_main, concept))
            else:
                entries.append((resolved, pages))
            continue

        # Otherwise it's a main entry
        current_main = concept
        entries.append((concept, pages))

    if skipped:
        print(f"  WARNING: {len(skipped)} sub-entries skipped (TODO or missing):")
        for k in skipped:
            print(f"    {k}")

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

        # When two sections start on the same page
        if end_pg < start_pg:
            end_pg = start_pg

        text = get_text(start_pg, end_pg, exclude_pages=EXCLUDED_PAGES)
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
        # sec["gtc"] = sorted(sec["gtc"])
        sec["gtc"] = sorted(set(sec["gtc"]))

    # Write output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Written to {OUTPUT_PATH}", flush=True)
    print(f"Sections: {len(sections)}", flush=True)

    for sec in sections:
        gc = len(sec["gtc"])
        print(f"  {sec['section_number']:>5} | {sec['section_name'][:55]:<55} | {gc:>3} concepts", flush=True)


if __name__ == "__main__":
    main()
