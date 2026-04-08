import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json, re, time

BASE = "https://nlp.stanford.edu/IR-book/html/htmledition/"
TOC  = BASE + "irbook.html"
STOP_AFTER_CHAPTER = 16

# 1. Get ordered list of section links from the TOC
toc_html = requests.get(TOC).text
soup = BeautifulSoup(toc_html, "html.parser")
links = [urljoin(BASE, a["href"])
         for a in soup.select("a[href$='.html']")
         if "irbook" not in a["href"]]

# Regex: leading "1", "1.2", "12.3.4", etc. followed by the title
HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.*)$")
sections = []

for url in links:
    page = BeautifulSoup(requests.get(url).text, "html.parser")

    for tag in page(["script", "style", "nav"]):
        tag.decompose()

    heading = page.find(["h1", "h2", "h3", "h4"])
    heading_txt = heading.get_text(" ", strip=True) if heading else ""
    m = HEADING_RE.match(heading_txt)

    if m:
        section_number, section_name = m.group(1), m.group(2).strip()
    else:
        section_number, section_name = "", heading_txt

    # Stop once we pass chapter 16
    if section_number:
        chapter = int(section_number.split(".")[0])

        if chapter > STOP_AFTER_CHAPTER:
            break

    # Remove the heading from the body so text doesn't duplicate it
    if heading:
        heading.decompose()

    text = page.get_text("\n", strip=True)
    print(url)

    sections.append({
        "section_number": section_number,
        "section_name":   section_name,
        "text":           text,
        "source_url":     url,
    })
    time.sleep(0.5)  # be polite

with open("iir_book.json", "w", encoding="utf-8") as f:
    json.dump(sections, f, ensure_ascii=False, indent=2)

print(f"Saved {len(sections)} sections to iir_book.json")

