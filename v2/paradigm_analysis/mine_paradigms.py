"""
mine_paradigms.py
-----------------
Data-driven paradigm discovery: downloads abstracts from PubMed for each
subfield and extracts task/paradigm mentions by pattern matching — without
any pre-specified list of paradigm names.

This gives a bottom-up picture of what paradigms are actually used,
including the long tail of less common tasks.

Usage:
    python mine_paradigms.py                        # all subfields, 2000 abstracts each
    python mine_paradigms.py --n 5000               # more abstracts per subfield
    python mine_paradigms.py --subfield "Working Memory"
    python mine_paradigms.py --out paradigm_counts_mined.json
    python mine_paradigms.py --plot                 # also generate frequency plots

Output:
    paradigm_counts_mined.json   — ranked paradigm counts per subfield
    (optional) figures in mined_figures/

How it works:
    1. ESearch  — fetch up to N PMIDs matching the subfield query
    2. EFetch   — download abstracts + titles in batches of 200
    3. Pattern matching — extract phrases ending in task-anchor words
       (task, paradigm, maze, game, test, assay, procedure, battery)
    4. Normalise — lowercase, deduplicate near-synonyms, filter stopwords
    5. Rank and report

Caveats:
    - Captures only paradigms named in title/abstract (same scope as [tiab])
    - Short n-gram window (1–4 words before anchor) may miss long names
    - No semantic disambiguation: "choice task" and "forced choice task"
      are counted separately until the normalisation step merges them
"""

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Subfield queries (same anchors as queries.py TOTAL, but filtered to
# empirical articles by requiring participant/subject/animal mentions and
# excluding review-type publication types)
# ---------------------------------------------------------------------------

SUBFIELD_QUERIES = {
    "Working Memory": (
        '"working memory"[tiab] '
        'AND ("participants"[tiab] OR "subjects"[tiab] OR "patients"[tiab] '
        '     OR "rats"[tiab] OR "mice"[tiab]) '
        'NOT (Review[pt] OR Meta-Analysis[pt] OR Systematic Review[pt] '
        '     OR Editorial[pt] OR Letter[pt] OR Comment[pt])'
    ),
    "Decision Making": (
        '"decision making"[tiab] '
        'AND ("participants"[tiab] OR "subjects"[tiab] OR "patients"[tiab] '
        '     OR "rats"[tiab] OR "mice"[tiab]) '
        'NOT (Review[pt] OR Meta-Analysis[pt] OR Systematic Review[pt] '
        '     OR Editorial[pt] OR Letter[pt] OR Comment[pt])'
    ),
    "Spatial Navigation": (
        '"spatial navigation"[tiab] '
        'AND ("participants"[tiab] OR "subjects"[tiab] OR "patients"[tiab] '
        '     OR "rats"[tiab] OR "mice"[tiab]) '
        'NOT (Review[pt] OR Meta-Analysis[pt] OR Systematic Review[pt] '
        '     OR Editorial[pt] OR Letter[pt] OR Comment[pt])'
    ),
    "Attention": (
        '"attention"[tiab] AND ("task"[tiab] OR "paradigm"[tiab]) '
        'AND ("participants"[tiab] OR "subjects"[tiab] OR "patients"[tiab] '
        '     OR "rats"[tiab] OR "mice"[tiab]) '
        'NOT (Review[pt] OR Meta-Analysis[pt] OR Systematic Review[pt] '
        '     OR Editorial[pt] OR Letter[pt] OR Comment[pt])'
    ),
}

# ---------------------------------------------------------------------------
# Anchor words: a phrase ending in one of these is a candidate paradigm name
# ---------------------------------------------------------------------------

ANCHORS = [
    "task", "tasks",
    "paradigm", "paradigms",
    "maze", "mazes",
    "game", "games",
    "test", "tests",
    "assay", "assays",
    "procedure", "procedures",
    "battery", "batteries",
    "protocol", "protocols",
]

# Words that should never be the start of a paradigm name
# (too generic or grammatical)
STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those",
    "our", "their", "its", "his", "her", "each", "every",
    "standard", "experimental", "current", "novel", "new",
    "simple", "basic", "modified", "adapted", "different",
    "various", "similar", "same", "single", "dual", "double",
    "well", "known", "classic", "traditional", "conventional",
    "human", "animal", "rat", "mouse", "primate",
    "cognitive", "behavioral", "behavioural", "neuropsychological",
    "clinical", "computerized", "computerised",
}

# ---------------------------------------------------------------------------
# Normalisation map: merge common surface variants → canonical name.
# Add entries here as needed.
# ---------------------------------------------------------------------------

NORMALISE = {
    "n back task":              "n-back task",
    "n-back tasks":             "n-back task",
    "nback task":               "n-back task",
    "go nogo task":             "go/no-go task",
    "go no go task":            "go/no-go task",
    "go/nogo task":             "go/no-go task",
    "gonogo task":              "go/no-go task",
    "iowa gambling tasks":      "iowa gambling task",
    "morris water maze":        "morris water maze",
    "morris water mazes":       "morris water maze",
    "water maze":               "morris water maze",
    "radial arm mazes":         "radial arm maze",
    "radial-arm maze":          "radial arm maze",
    "barnes mazes":             "barnes maze",
    "stroop tasks":             "stroop task",
    "stroop color word task":   "stroop task",
    "stroop colour word task":  "stroop task",
    "flanker tasks":            "flanker task",
    "eriksen flanker task":     "flanker task",
    "eriksen flanker tasks":    "flanker task",
    "oddball tasks":            "oddball task",
    "oddball paradigms":        "oddball paradigm",
    "continuous performance tasks": "continuous performance task",
    "delay discounting task":   "delay discounting task",
    "delay discounting tasks":  "delay discounting task",
    "intertemporal choice task":"delay discounting task",
    "stop signal tasks":        "stop signal task",
    "stop-signal task":         "stop signal task",
    "stop-signal tasks":        "stop signal task",
    "ultimatum games":          "ultimatum game",
    "trust games":              "trust game",
    "dictator games":           "dictator game",
    "balloon analogue risk task": "bart",
}


# ---------------------------------------------------------------------------
# PubMed helpers
# ---------------------------------------------------------------------------

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def post(url, params):
    body = urllib.parse.urlencode(params).encode()
    req  = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def get_pmids(query, n, email="", api_key=""):
    """Return up to n PMIDs matching query, sorted by most recent."""
    params = {
        "db":         "pubmed",
        "term":       query,
        "retmax":     n,
        "retmode":    "xml",
        "sort":       "most recent",
    }
    if email:    params["email"]   = email
    if api_key:  params["api_key"] = api_key
    root = ET.fromstring(post(ESEARCH_URL, params))
    total = int(root.findtext("Count") or 0)
    pmids = [e.text for e in root.findall(".//Id")]
    return pmids, total


def fetch_abstracts(pmids, email="", api_key="", batch=200, delay=0.5):
    """Yield (title, abstract) strings for each PMID, batched."""
    for i in range(0, len(pmids), batch):
        chunk = pmids[i:i + batch]
        params = {
            "db":       "pubmed",
            "id":       ",".join(chunk),
            "rettype":  "abstract",
            "retmode":  "xml",
        }
        if email:    params["email"]   = email
        if api_key:  params["api_key"] = api_key
        raw  = post(EFETCH_URL, params)
        root = ET.fromstring(raw)
        for article in root.findall(".//PubmedArticle"):
            title = article.findtext(".//ArticleTitle") or ""
            parts = article.findall(".//AbstractText")
            abstract = " ".join(p.text or "" for p in parts if p.text)
            yield title + " " + abstract
        print(f"    fetched {min(i + batch, len(pmids))}/{len(pmids)} abstracts …",
              end="\r", flush=True)
        time.sleep(delay)
    print()


# ---------------------------------------------------------------------------
# Pattern extraction
# ---------------------------------------------------------------------------

# Left-boundary triggers: a named paradigm phrase almost always follows one
# of these in a methods/results sentence.
# The optional secondary article handles "using a X task", "with the Y maze", etc.
_LEFT_TRIGGERS = r'(?:the|a|an|and|or|using|with|by|on)\s+(?:(?:a|an|the)\s+)?'

# The paradigm name itself: 1–3 content words (allowing hyphens, slashes,
# digits for things like "n-back", "2AFC", "go/no-go").
_CONTENT_WORD  = r'[\w][\w\-/]*'
_NAME_BODY     = rf'(?:{_CONTENT_WORD}\s+){{0,2}}{_CONTENT_WORD}'

# Anchor words joined as alternation
_ANCHOR_ALT    = '|'.join(re.escape(a) for a in ANCHORS)

# Full pattern:  left-trigger  name-body  anchor
_ANCHOR_RE = re.compile(
    rf'(?i){_LEFT_TRIGGERS}({_NAME_BODY})\s+({_ANCHOR_ALT})\b'
)


def extract_candidates(text):
    """Return a list of normalised paradigm-name strings found in text."""
    results = []
    for m in _ANCHOR_RE.finditer(text):
        prefix = m.group(1).strip().lower()
        anchor = m.group(2).strip().lower()

        # Skip if the prefix is a stopword or too short/generic
        words = prefix.split()
        if not words or words[0] in STOPWORDS or len(prefix) < 3:
            continue
        # Skip pure-number prefixes ("3 task")
        if re.match(r'^\d+$', prefix):
            continue

        phrase = f"{prefix} {anchor}"
        phrase = NORMALISE.get(phrase, phrase)
        results.append(phrase)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mine paradigm names from PubMed abstracts.")
    parser.add_argument("--n",         type=int, default=2000,
                        help="Max abstracts to fetch per subfield (default: 2000)")
    parser.add_argument("--subfield",  default=None,
                        help="Run for one subfield only (default: all)")
    parser.add_argument("--email",     default="")
    parser.add_argument("--api-key",   default="")
    parser.add_argument("--delay",     type=float, default=0.4,
                        help="Seconds between fetch batches")
    parser.add_argument("--top",       type=int, default=30,
                        help="Number of top paradigms to display/save (default: 30)")
    parser.add_argument("--out",       default="paradigm_counts_mined.json")
    parser.add_argument("--plot",      action="store_true",
                        help="Generate frequency bar charts")
    args = parser.parse_args()

    subfields = (
        {args.subfield: SUBFIELD_QUERIES[args.subfield]}
        if args.subfield else SUBFIELD_QUERIES
    )

    all_results = {}

    for sf_name, query in subfields.items():
        print(f"\n{'='*60}")
        print(f"  {sf_name}")
        print(f"{'='*60}")

        print(f"  Fetching up to {args.n} PMIDs …")
        pmids, total_in_pubmed = get_pmids(
            query, args.n, email=args.email, api_key=args.api_key
        )
        print(f"  Total matching PubMed (empirical filter): {total_in_pubmed:,}")
        print(f"  Downloading {len(pmids)} abstracts …")
        time.sleep(args.delay)

        counter = Counter()
        n_docs  = 0
        for text in fetch_abstracts(
            pmids, email=args.email, api_key=args.api_key,
            batch=200, delay=args.delay
        ):
            for phrase in extract_candidates(text):
                counter[phrase] += 1
            n_docs += 1

        top = counter.most_common(args.top)

        print(f"\n  Top {args.top} paradigm mentions (out of {n_docs} abstracts):\n")
        print(f"  {'Rank':<5} {'Paradigm':<45} {'Count':>6}  {'% docs':>7}")
        print(f"  {'-'*4}  {'-'*44}  {'-'*6}  {'-'*7}")
        for rank, (phrase, count) in enumerate(top, 1):
            pct = count / n_docs * 100
            print(f"  {rank:<5} {phrase:<45} {count:>6,}  {pct:>6.1f}%")

        all_results[sf_name] = {
            "total_empirical_pubmed": total_in_pubmed,
            "n_abstracts_sampled":    n_docs,
            "paradigm_counts":        dict(top),
        }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n✓ Results saved to {out_path.resolve()}")

    if args.plot:
        _plot_results(all_results, args.top)


def _plot_results(results, top_n):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from queries import SUBFIELD_COLORS
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return

    out_dir = Path("mined_figures")
    out_dir.mkdir(exist_ok=True)

    for sf_name, data in results.items():
        counts = data["paradigm_counts"]
        if not counts:
            continue
        items  = sorted(counts.items(), key=lambda x: -x[1])[:top_n]
        names  = [x[0] for x in items]
        vals   = [x[1] for x in items]
        total  = data["n_abstracts_sampled"]
        color  = SUBFIELD_COLORS.get(sf_name, "#607D8B")

        fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.35)))
        ax.barh(range(len(names)), vals,
                color=color, alpha=0.8, edgecolor="white")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Number of abstracts mentioning paradigm", fontsize=11)
        ax.set_title(
            f"{sf_name} — Data-driven paradigm frequencies\n"
            f"(from {total:,} sampled abstracts; "
            f"n={data['total_empirical_pubmed']:,} empirical papers on PubMed)",
            fontsize=12, fontweight="bold"
        )
        for i, (bar_val, name) in enumerate(zip(vals, names)):
            pct = bar_val / total * 100
            ax.text(bar_val + max(vals) * 0.01, i,
                    f"{bar_val:,}  ({pct:.1f}%)", va="center", fontsize=8)
        ax.set_xlim(0, max(vals) * 1.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fname = out_dir / f"mined_{sf_name.lower().replace(' ', '_')}.png"
        plt.savefig(fname, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}")


if __name__ == "__main__":
    main()
