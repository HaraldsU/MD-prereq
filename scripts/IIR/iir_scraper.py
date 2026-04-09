"""
Scrape Introduction to Information Retrieval (Manning, Raghavan, Schütze)
from https://nlp.stanford.edu/IR-book/html/htmledition/

Produces a JSON file with one object per (numbered) section.
Subsubsection text is folded into its parent section's text.
Scope: Chapter 1 "Boolean retrieval" (section 1.1) through Chapter 16 "Flat
clustering" section "Model-based clustering" (16.9), inclusive.
"""

import json
import time
import re
import requests
from bs4 import BeautifulSoup

BASE = "https://nlp.stanford.edu/IR-book/html/htmledition/"
HEADERS = {
    "User-Agent": "iir-dataset-scraper/1.0 (personal research; contact: you@example.com)"
}

# TOC: (chapter_number, chapter_name, chapter_slug, [(section_name, section_slug, [subsubsection_slug, ...]), ...])
# Slugs are the filename without the "-1.html" suffix (we append it when building URLs).
TOC = [
    (1, "Boolean retrieval", "boolean-retrieval", [
        ("An example information retrieval problem", "an-example-information-retrieval-problem", []),
        ("A first take at building an inverted index", "a-first-take-at-building-an-inverted-index", []),
        ("Processing Boolean queries", "processing-boolean-queries", []),
        ("The extended Boolean model versus ranked retrieval", "the-extended-boolean-model-versus-ranked-retrieval", []),
        ("References and further reading", "references-and-further-reading", []),
    ]),
    (2, "The term vocabulary and postings lists", "the-term-vocabulary-and-postings-lists", [
        ("Document delineation and character sequence decoding", "document-delineation-and-character-sequence-decoding",
            ["obtaining-the-character-sequence-in-a-document", "choosing-a-document-unit"]),
        ("Determining the vocabulary of terms", "determining-the-vocabulary-of-terms",
            ["tokenization", "dropping-common-terms-stop-words",
             "normalization-equivalence-classing-of-terms", "stemming-and-lemmatization"]),
        ("Faster postings list intersection via skip pointers", "faster-postings-list-intersection-via-skip-pointers", []),
        ("Positional postings and phrase queries", "positional-postings-and-phrase-queries",
            ["biword-indexes", "positional-indexes", "combination-schemes"]),
        ("References and further reading", "references-and-further-reading-2", []),
    ]),
    (3, "Dictionaries and tolerant retrieval", "dictionaries-and-tolerant-retrieval", [
        ("Search structures for dictionaries", "search-structures-for-dictionaries", []),
        ("Wildcard queries", "wildcard-queries",
            ["general-wildcard-queries", "k-gram-indexes-for-wildcard-queries"]),
        ("Spelling correction", "spelling-correction",
            ["implementing-spelling-correction", "forms-of-spelling-correction", "edit-distance",
             "k-gram-indexes-for-spelling-correction", "context-sensitive-spelling-correction"]),
        ("Phonetic correction", "phonetic-correction", []),
        ("References and further reading", "references-and-further-reading-3", []),
    ]),
    (4, "Index construction", "index-construction", [
        ("Hardware basics", "hardware-basics", []),
        ("Blocked sort-based indexing", "blocked-sort-based-indexing", []),
        ("Single-pass in-memory indexing", "single-pass-in-memory-indexing", []),
        ("Distributed indexing", "distributed-indexing", []),
        ("Dynamic indexing", "dynamic-indexing", []),
        ("Other types of indexes", "other-types-of-indexes", []),
        ("References and further reading", "references-and-further-reading-4", []),
    ]),
    (5, "Index compression", "index-compression", [
        ("Statistical properties of terms in information retrieval", "statistical-properties-of-terms-in-information-retrieval",
            ["heaps-law-estimating-the-number-of-terms", "zipfs-law-modeling-the-distribution-of-terms"]),
        ("Dictionary compression", "dictionary-compression",
            ["dictionary-as-a-string", "blocked-storage"]),
        ("Postings file compression", "postings-file-compression",
            ["variable-byte-codes", "gamma-codes"]),
        ("References and further reading", "references-and-further-reading-5", []),
    ]),
    (6, "Scoring, term weighting and the vector space model", "scoring-term-weighting-and-the-vector-space-model", [
        ("Parametric and zone indexes", "parametric-and-zone-indexes",
            ["weighted-zone-scoring", "learning-weights", "the-optimal-weight-g"]),
        ("Term frequency and weighting", "term-frequency-and-weighting",
            ["inverse-document-frequency", "tf-idf-weighting"]),
        ("The vector space model for scoring", "the-vector-space-model-for-scoring",
            ["dot-products", "queries-as-vectors", "computing-vector-scores"]),
        ("Variant tf-idf functions", "variant-tf-idf-functions",
            ["sublinear-tf-scaling", "maximum-tf-normalization",
             "document-and-query-weighting-schemes", "pivoted-normalized-document-length"]),
        ("References and further reading", "references-and-further-reading-6", []),
    ]),
    (7, "Computing scores in a complete search system", "computing-scores-in-a-complete-search-system", [
        ("Efficient scoring and ranking", "efficient-scoring-and-ranking",
            ["inexact-top-k-document-retrieval", "index-elimination", "champion-lists",
             "static-quality-scores-and-ordering", "impact-ordering", "cluster-pruning"]),
        ("Components of an information retrieval system", "components-of-an-information-retrieval-system",
            ["tiered-indexes", "query-term-proximity",
             "designing-parsing-and-scoring-functions", "putting-it-all-together"]),
        ("Vector space scoring and query operator interaction", "vector-space-scoring-and-query-operator-interaction", []),
        ("References and further reading", "references-and-further-reading-7", []),
    ]),
    (8, "Evaluation in information retrieval", "evaluation-in-information-retrieval", [
        ("Information retrieval system evaluation", "information-retrieval-system-evaluation", []),
        ("Standard test collections", "standard-test-collections", []),
        ("Evaluation of unranked retrieval sets", "evaluation-of-unranked-retrieval-sets", []),
        ("Evaluation of ranked retrieval results", "evaluation-of-ranked-retrieval-results", []),
        ("Assessing relevance", "assessing-relevance",
            ["critiques-and-justifications-of-the-concept-of-relevance"]),
        ("A broader perspective: System quality and user utility", "a-broader-perspective-system-quality-and-user-utility",
            ["system-issues", "user-utility", "refining-a-deployed-system"]),
        ("Results snippets", "results-snippets", []),
        ("References and further reading", "references-and-further-reading-8", []),
    ]),
    (9, "Relevance feedback and query expansion", "relevance-feedback-and-query-expansion", [
        ("Relevance feedback and pseudo relevance feedback", "relevance-feedback-and-pseudo-relevance-feedback",
            ["the-rocchio-algorithm-for-relevance-feedback", "probabilistic-relevance-feedback",
             "when-does-relevance-feedback-work", "relevance-feedback-on-the-web",
             "evaluation-of-relevance-feedback-strategies", "pseudo-relevance-feedback",
             "indirect-relevance-feedback", "summary"]),
        ("Global methods for query reformulation", "global-methods-for-query-reformulation",
            ["vocabulary-tools-for-query-reformulation", "query-expansion", "automatic-thesaurus-generation"]),
        ("References and further reading", "references-and-further-reading-9", []),
    ]),
    (10, "XML retrieval", "xml-retrieval", [
        ("Basic XML concepts", "basic-xml-concepts", []),
        ("Challenges in XML retrieval", "challenges-in-xml-retrieval", []),
        ("A vector space model for XML retrieval", "a-vector-space-model-for-xml-retrieval", []),
        ("Evaluation of XML retrieval", "evaluation-of-xml-retrieval", []),
        ("Text-centric vs. data-centric XML retrieval", "text-centric-vs-data-centric-xml-retrieval", []),
        ("References and further reading", "references-and-further-reading-10", []),
        ("Exercises", "exercises-1", []),
    ]),
    (11, "Probabilistic information retrieval", "probabilistic-information-retrieval", [
        ("Review of basic probability theory", "review-of-basic-probability-theory", []),
        ("The Probability Ranking Principle", "the-probability-ranking-principle",
            ["the-10-loss-case", "the-prp-with-retrieval-costs"]),
        ("The Binary Independence Model", "the-binary-independence-model",
            ["deriving-a-ranking-function-for-query-terms", "probability-estimates-in-theory",
             "probability-estimates-in-practice", "probabilistic-approaches-to-relevance-feedback"]),
        ("An appraisal and some extensions", "an-appraisal-and-some-extensions",
            ["an-appraisal-of-probabilistic-models", "tree-structured-dependencies-between-terms",
             "okapi-bm25-a-non-binary-model", "bayesian-network-approaches-to-ir"]),
        ("References and further reading", "references-and-further-reading-11", []),
    ]),
    (12, "Language models for information retrieval", "language-models-for-information-retrieval", [
        ("Language models", "language-models",
            ["finite-automata-and-language-models", "types-of-language-models",
             "multinomial-distributions-over-words"]),
        ("The query likelihood model", "the-query-likelihood-model",
            ["using-query-likelihood-language-models-in-ir",
             "estimating-the-query-generation-probability", "ponte-and-crofts-experiments"]),
        ("Language modeling versus other approaches in IR", "language-modeling-versus-other-approaches-in-ir", []),
        ("Extended language modeling approaches", "extended-language-modeling-approaches", []),
        ("References and further reading", "references-and-further-reading-12", []),
    ]),
    (13, "Text classification and Naive Bayes", "text-classification-and-naive-bayes", [
        ("The text classification problem", "the-text-classification-problem", []),
        ("Naive Bayes text classification", "naive-bayes-text-classification",
            ["relation-to-multinomial-unigram-language-model"]),
        ("The Bernoulli model", "the-bernoulli-model", []),
        ("Properties of Naive Bayes", "properties-of-naive-bayes",
            ["a-variant-of-the-multinomial-model"]),
        ("Feature selection", "feature-selection",
            ["mutual-information", "feature-selectionchi2-feature-selection",
             "frequency-based-feature-selection", "feature-selection-for-multiple-classifiers",
             "comparison-of-feature-selection-methods"]),
        ("Evaluation of text classification", "evaluation-of-text-classification", []),
        ("References and further reading", "references-and-further-reading-13", []),
    ]),
    (14, "Vector space classification", "vector-space-classification", [
        ("Document representations and measures of relatedness in vector spaces",
            "document-representations-and-measures-of-relatedness-in-vector-spaces", []),
        ("Rocchio classification", "rocchio-classification", []),
        ("k nearest neighbor", "k-nearest-neighbor",
            ["time-complexity-and-optimality-of-knn"]),
        ("Linear versus nonlinear classifiers", "linear-versus-nonlinear-classifiers", []),
        ("Classification with more than two classes", "classification-with-more-than-two-classes", []),
        ("The bias-variance tradeoff", "the-bias-variance-tradeoff", []),
        ("References and further reading", "references-and-further-reading-14", []),
        ("Exercises", "exercises-2", []),
    ]),
    (15, "Support vector machines and machine learning on documents",
        "support-vector-machines-and-machine-learning-on-documents", [
        ("Support vector machines: The linearly separable case",
            "support-vector-machines-the-linearly-separable-case", []),
        ("Extensions to the SVM model", "extensions-to-the-svm-model",
            ["soft-margin-classification", "multiclass-svms", "nonlinear-svms", "experimental-results"]),
        ("Issues in the classification of text documents", "issues-in-the-classification-of-text-documents",
            ["choosing-what-kind-of-classifier-to-use", "improving-classifier-performance"]),
        ("Machine learning methods in ad hoc information retrieval",
            "machine-learning-methods-in-ad-hoc-information-retrieval",
            ["a-simple-example-of-machine-learned-scoring", "result-ranking-by-machine-learning"]),
        ("References and further reading", "references-and-further-reading-15", []),
    ]),
    (16, "Flat clustering", "flat-clustering", [
        ("Clustering in information retrieval", "clustering-in-information-retrieval", []),
        ("Problem statement", "problem-statement",
            ["cardinality---the-number-of-clusters"]),
        ("Evaluation of clustering", "evaluation-of-clustering", []),
        ("K-means", "k-means",
            ["cluster-cardinality-in-k-means"]),
        ("Model-based clustering", "model-based-clustering", []),
    ]),
]


def url_for(slug: str) -> str:
    # Slugs like "references-and-further-reading-2" or "exercises-1" are
    # already complete filenames; only bare slugs need "-1" appended.
    if re.search(r"-\d+$", slug):
        return f"{BASE}{slug}.html"
    return f"{BASE}{slug}-1.html"


def fetch(url: str, cache: dict) -> str:
    if url in cache:
        return cache[url]
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30, headers=HEADERS)
            r.raise_for_status()
            cache[url] = r.text
            time.sleep(0.2)  # be polite
            return r.text
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2)


def extract_body_text(html: str) -> str:
    """
    Extract main textual content from an IR-book HTML page, stripping the
    top/bottom navigation (next/up/prev/contents/index) and the final
    'Subsections' list and footer.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove obvious nav chrome: any <a> containing navigation icons
    for img in soup.find_all("img"):
        alt = (img.get("alt") or "").strip().lower()
        src = (img.get("src") or "").lower()
        if alt in {"next", "up", "previous", "contents", "index"} or \
           any(k in src for k in ["next.png", "up.png", "prev.png", "contents.png", "index.png"]):
            # Kill the enclosing link if present
            a = img.find_parent("a")
            (a or img).decompose()

    # Horizontal rules delimit nav blocks in LaTeX2HTML output. We keep all
    # text but will strip the "Next: / Up: / Previous:" navigation lines and
    # the trailing "Subsections" list and copyright footer by pattern.
    text = soup.get_text("\n")

    # Normalize whitespace per-line but preserve paragraph breaks
    lines = [ln.strip() for ln in text.splitlines()]
    # Collapse runs of blank lines to a single blank line
    cleaned = []
    prev_blank = False
    for ln in lines:
        if not ln:
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            cleaned.append(ln)
            prev_blank = False
    text = "\n".join(cleaned).strip()

    # Remove leading nav block: lines like "Next: ...", "Up: ...", "Previous: ..."
    # and standalone "Contents" / "Index" tokens until we hit real content.
    nav_patterns = [
        re.compile(r"^\s*Next:\s", re.I),
        re.compile(r"^\s*Up:\s", re.I),
        re.compile(r"^\s*Previous:\s", re.I),
        re.compile(r"^\s*Contents\s*$", re.I),
        re.compile(r"^\s*Index\s*$", re.I),
    ]

    def strip_nav_block(lines):
        # Drop any leading run of nav lines / blanks
        i = 0
        while i < len(lines):
            ln = lines[i]
            if ln == "" or any(p.match(ln) for p in nav_patterns):
                i += 1
                continue
            break
        return lines[i:]

    lines = text.split("\n")
    lines = strip_nav_block(lines)
    # Also strip a trailing nav block
    lines = list(reversed(strip_nav_block(list(reversed(lines)))))

    # Remove the "Subsections" list near the end (everything from a line that
    # is exactly "Subsections" up to the footer/copyright).
    try:
        sub_idx = next(i for i, ln in enumerate(lines) if ln.strip() == "Subsections")
        lines = lines[:sub_idx]
    except StopIteration:
        pass

    # Remove footer boilerplate
    footer_markers = [
        "© 2008 Cambridge University Press",
        "This is an automatically generated page",
        "2009-04-07",
    ]
    lines = [ln for ln in lines if not any(m in ln for m in footer_markers)]

    # Strip any leftover lone "PDF edition" note line
    lines = [ln for ln in lines if "PDF edition" not in ln]

    return "\n".join(ln for ln in lines).strip()


def main():
    cache = {}
    records = []

    for chapter_num, chapter_name, chapter_slug, sections in TOC:
        chapter_url = url_for(chapter_slug)
        print(f"[{chapter_num}] {chapter_name} -> {chapter_url}")
        chapter_html = fetch(chapter_url, cache)
        records.append({
            "section_number": str(chapter_num),
            "section_name": chapter_name,
            "section_text": extract_body_text(chapter_html),
            "source_url": chapter_url,
        })

        for section_idx, (section_name, section_slug, sub_slugs) in enumerate(sections, start=1):
            if section_name in ("References and further reading", "Exercises"):
                continue
            section_number = f"{chapter_num}.{section_idx}"
            url = url_for(section_slug)
            print(f"[{section_number}] {section_name} -> {url}")
            html = fetch(url, cache)
            parts = [extract_body_text(html)]

            for sub_slug in sub_slugs:
                sub_url = url_for(sub_slug)
                print(f"       + {sub_url}")
                sub_html = fetch(sub_url, cache)
                parts.append(extract_body_text(sub_html))

            full_text = "\n\n".join(p for p in parts if p).strip()

            records.append({
                "section_number": section_number,
                "section_name": section_name,
                "section_text": full_text,
                "source_url": url,
            })

    out_path = "iir_sections.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(records)} sections to {out_path}")


if __name__ == "__main__":
    main()
