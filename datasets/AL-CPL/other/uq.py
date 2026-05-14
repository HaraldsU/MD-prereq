import json
import sys
import time
import re
import requests
from bs4 import BeautifulSoup

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "ConceptChecker/1.0 (academic research; contact@example.com)"
})

def get_unique_concepts(filepath):
# {{{
    with open(filepath) as f:
        data = json.load(f)

    concepts = set()

    for obj in data:
        concepts.add(obj["concept_A"])
        concepts.add(obj["concept_B"])

    return sorted(concepts)
# }}}

def get_latvian_titles_batch(titles):
# {{{
    """Query up to 50 titles at once via Wikipedia API."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": "|".join(t.replace("_", " ") for t in titles),
        "prop": "langlinks",
        "lllang": "lv",
        "lllimit": "500",
        "format": "json",
    }
    try:
        resp = SESSION.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"  [WARNING] Batch API error: {e}", file=sys.stderr)
        return {}

    results = {}
    pages = data.get("query", {}).get("pages", {})
    normalized = {n["from"]: n["to"] for n in data.get("query", {}).get("normalized", [])}

    for page in pages.values():
        title = page.get("title", "")
        langlinks = page.get("langlinks", [])
        lv = langlinks[0]["*"] if langlinks else None
        results[title] = lv

    mapped = {}
    for orig in titles:
        lookup = orig.replace("_", " ")
        lookup = normalized.get(lookup, lookup)
        mapped[orig] = results.get(lookup)
    return mapped
# }}}

def search_akadterm(en_term):
# {{{
    """Search akadterm.lv for a Latvian translation of an English term."""
    search = en_term.replace("_", " ")
    url = "https://www.akadterm.lv/term.php"
    params = {"term": search, "list": search, "lang": "EN"}
    try:
        resp = SESSION.get(url, params=params, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except (requests.RequestException, ValueError) as e:
        print(f"  [WARNING] AkadTerm error for '{en_term}': {e}", file=sys.stderr)
        return None

    # Look for exact match: find EN term then grab LV translation
    text = soup.get_text()
    # The page lists numbered entries like:
    # 1. EN <term>
    # LV <latvian term>
    # We want entries where the EN line matches our term exactly
    lines = text.split("\n")
    for i, line in enumerate(lines):
        line = line.strip()
        # Match lines like "EN cone" or "EN cone; ..."
        if line.startswith("EN "):
            en_terms = [t.strip().lower() for t in line[3:].split(";")]
            # Remove annotations like "(IETEICAMS)" etc
            en_terms = [re.sub(r'\s*\(.*?\)', '', t).strip() for t in en_terms]
            if search.lower() in en_terms:
                # Next line(s) should have "LV ..."
                for j in range(i + 1, min(i + 5, len(lines))):
                    lv_line = lines[j].strip()
                    if lv_line.startswith("LV "):
                        lv_term = lv_line[3:].split(";")[0].strip()
                        lv_term = re.sub(r'\s*\(.*?\)', '', lv_term).strip()
                        if lv_term:
                            return lv_term
    return None
# }}}

def load_results(filepath):
# {{{
    """Load previously saved results from a semicolon-separated file."""
    results = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if "; " in line:
                concept, lv = line.split("; ", 1)
                results[concept] = lv if lv != "None" else None
    return results
# }}}

def check_latvian_articles(filepath, prev_results=None):
# {{{
    if prev_results:
        results = load_results(prev_results)
    else:
        concepts = get_unique_concepts(filepath)
        results = {}
        batch_size = 50
        for i in range(0, len(concepts), batch_size):
            batch = concepts[i:i + batch_size]
            batch_results = get_latvian_titles_batch(batch)
            results.update(batch_results)
            time.sleep(1)

    # For all None results, try akadterm
    none_concepts = [c for c, v in results.items() if v is None]
    print(f"Checking {len(none_concepts)} concepts on akadterm.lv...", file=sys.stderr)

    for i, concept in enumerate(none_concepts):
        lv = search_akadterm(concept)
        if lv:
            results[concept] = lv
            print(f"  [{i+1}/{len(none_concepts)}] FOUND: {concept} -> {lv}", file=sys.stderr)
        else:
            print(f"  [{i+1}/{len(none_concepts)}] not found: {concept}", file=sys.stderr)
        time.sleep(0.5)  # be nice

    for concept in sorted(results.keys()):
        lv = results[concept]
        print(f"{concept}; {lv if lv else 'None'}")
# }}}

def fetch_wikipedia_article(title: str, retries = 5):
# {{{
    """Fetch a Wikipedia article summary via the free API."""
    url = 'https://lv.wikipedia.org/api/rest_v1/page/summary/' + title

    for attempt in range(retries):
        try:
            resp = requests.get(url, headers={'User-Agent': 'MyApp/1.0'}, timeout=10)

            if resp.status_code == 200:
                return resp.json().get('extract')
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue

            return None
        except requests.RequestException:
            time.sleep(2 ** attempt)

    return None
# }}}

if __name__ == "__main__":
# {{{
    path = sys.argv[1] if len(sys.argv) > 1 else "data.json"
    # prev = sys.argv[2] if len(sys.argv) > 2 else None
    # check_latvian_articles(path, prev)

    # print(len(get_unique_concepts(path)))
    concepts = get_unique_concepts(path)
    count = 0
    
    for i, concept in enumerate(concepts):
        wiki = fetch_wikipedia_article(concept)

        if i % 10 == 0:
            print(f'{i}/{len(concepts)}')

        if wiki == None:
            count += 1

    print('count = ', count)

    # print(get_unique_concepts(path))
# }}}

