import requests
import json
import unicodedata
from pathlib import Path
from bs4 import BeautifulSoup

def extract_concepts(term):
# {{{
    parts = []
    for chunk in term.split('/'):
        for sub in chunk.split(','):
            cleaned = sub.strip()
            if cleaned:
                parts.append(cleaned)
    return parts
# }}}

def get_index_link_map():
# {{{
    url = "https://nlp.stanford.edu/IR-book/html/htmledition/index-1.html"
    soup = BeautifulSoup(requests.get(url).text, 'html5lib')
    index = {}
    base = "https://nlp.stanford.edu/IR-book/html/htmledition/"
    for dt in soup.find_all('dt'):
        strong = dt.find('strong')
        if not strong:
            continue
        term = clean(strong.get_text(strip=True))
        concepts = extract_concepts(term)
        dd = dt.find_next_sibling('dd')
        if not dd:
            continue
        seen = set()
        for a in dd.find_all('a'):
            href = base + a.get('href', '').split('#')[0]
            if href not in seen:
                seen.add(href)
                if href not in index:
                    index[href] = set()
                index[href].update(concepts)
    return index
# }}}

def find_terms_by_url(index, url):
# {{{
    if url in index:
        return index[url]
    return set()
# }}}

def clean(text):
# {{{
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
# }}}

def write_json(obj, name: str):
# {{{
    with open('outputs/' + name + '.json', 'w') as f:
        json.dump(obj, f, ensure_ascii = True, indent = 2)
        # json.dump(obj, f, indent = 2)
# }}}

def make_index_gtc_full_section_json():
# {{{
    iir = Path('~/Downloads/prereq/scripts/IIR/iir_sections_full.json').expanduser()
    index = get_index_link_map()
    index = {clean(k): v for k, v in index.items()}
    new_objects = []

    with open (iir, 'r') as f:
        data = json.load(f)

        for obj in data:
            section_number = obj['section_number']
            section_name = obj['section_name']
            section_text = obj['section_text']
            source_urls = obj['source_urls']
            concepts = []
            
            for url in source_urls:
                # print (url, find_terms_by_url(index, url))
                c = find_terms_by_url(index, url)
                concepts.append(c)

            concepts_flat = [item for s in concepts for item in s]
            # print(concepts_flat)
            # print()

            new_objects.append({
                'section_number': section_number,
                'section_name': section_name,
                'section_text': section_text,
                'gtc': concepts_flat,
                'source_urls': source_urls,
            })

    write_json(new_objects, 'iir_sections_full_index_test_test')
# }}}

make_index_gtc_full_section_json()

