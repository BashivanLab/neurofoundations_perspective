"""
1_download_abstracts.py
-----------------------
Downloads titles + abstracts from PubMed for each subfield using the
empirical-article filter (excludes reviews, meta-analyses, editorials;
requires mention of participants/subjects/animals).

Saves one JSONL file per subfield:
    abstracts/working_memory.jsonl
    abstracts/decision_making.jsonl
    abstracts/spatial_navigation.jsonl
    abstracts/attention.jsonl

Each line is:
    {"pmid": "12345678", "title": "...", "abstract": "...", "subfield": "..."}

Resumable: already-downloaded PMIDs are skipped if the output file exists.

Usage:
    python 1_download_abstracts.py
    python 1_download_abstracts.py --subfield "Working Memory"
    python 1_download_abstracts.py --max-per-subfield 10000
    python 1_download_abstracts.py --email you@uni.edu --api-key YOUR_KEY
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# Empirical-article filter queries
# Each query restricts to:
#   - subfield keyword in title/abstract
#   - presence of participants/subjects/animals  (proxy for empirical study)
#   - NOT review-type publication types
# ---------------------------------------------------------------------------

SUBFIELD_QUERIES = {
    "Working Memory": (
        '"working memory"[tiab] '
        'AND ("participants"[tiab] OR "subjects"[tiab] OR "patients"[tiab] '
        '     OR "rats"[tiab] OR "mice"[tiab]) '
        'NOT (Review[pt] OR Meta-Analysis[pt] OR "Systematic Review"[pt] '
        '     OR Editorial[pt] OR Letter[pt] OR Comment[pt])'
    ),
    "Decision Making": (
        '"decision making"[tiab] '
        'AND ("participants"[tiab] OR "subjects"[tiab] OR "patients"[tiab] '
        '     OR "rats"[tiab] OR "mice"[tiab]) '
        'NOT (Review[pt] OR Meta-Analysis[pt] OR "Systematic Review"[pt] '
        '     OR Editorial[pt] OR Letter[pt] OR Comment[pt])'
    ),
    "Spatial Navigation": (
        '"spatial navigation"[tiab] '
        'AND ("participants"[tiab] OR "subjects"[tiab] OR "patients"[tiab] '
        '     OR "rats"[tiab] OR "mice"[tiab]) '
        'NOT (Review[pt] OR Meta-Analysis[pt] OR "Systematic Review"[pt] '
        '     OR Editorial[pt] OR Letter[pt] OR Comment[pt])'
    ),
    "Attention": (
        '"attention"[tiab] AND ("task"[tiab] OR "paradigm"[tiab]) '
        'AND ("participants"[tiab] OR "subjects"[tiab] OR "patients"[tiab] '
        '     OR "rats"[tiab] OR "mice"[tiab]) '
        'NOT (Review[pt] OR Meta-Analysis[pt] OR "Systematic Review"[pt] '
        '     OR Editorial[pt] OR Letter[pt] OR Comment[pt])'
    ),
}

EUTILS  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def post(url, params):
    body = urllib.parse.urlencode(params).encode()
    req  = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def get_all_pmids(query, email="", api_key="", delay=0.4):
    """Fetch every PMID matching query using ESearch with retstart paging."""
    params = dict(db="pubmed", term=query, retmode="xml",
                  retmax=0, usehistory="y")
    if email:    params["email"]   = email
    if api_key:  params["api_key"] = api_key

    root      = ET.fromstring(post(f"{EUTILS}/esearch.fcgi", params))
    total     = int(root.findtext("Count") or 0)
    webenv    = root.findtext("WebEnv")
    query_key = root.findtext("QueryKey")
    print(f"    Total matching PubMed: {total:,}")

    pmids = []
    batch = 10_000
    for start in range(0, total, batch):
        p = dict(db="pubmed", query_key=query_key, WebEnv=webenv,
                 retstart=start, retmax=batch, retmode="xml")
        if email:    p["email"]   = email
        if api_key:  p["api_key"] = api_key
        r    = ET.fromstring(post(f"{EUTILS}/esearch.fcgi", p))
        pmids += [e.text for e in r.findall(".//Id")]
        print(f"    PMIDs retrieved: {len(pmids):,} / {total:,}", end="\r")
        time.sleep(delay)
    print()
    return pmids, total


def fetch_records(pmids, email="", api_key="", batch=200, delay=0.4):
    """Yield dicts {pmid, title, abstract} for each PMID."""
    for i in range(0, len(pmids), batch):
        chunk  = pmids[i : i + batch]
        params = dict(db="pubmed", id=",".join(chunk),
                      rettype="abstract", retmode="xml")
        if email:    params["email"]   = email
        if api_key:  params["api_key"] = api_key
        raw  = post(f"{EUTILS}/efetch.fcgi", params)
        root = ET.fromstring(raw)

        for art in root.findall(".//PubmedArticle"):
            pmid  = art.findtext(".//PMID") or ""
            title = art.findtext(".//ArticleTitle") or ""
            parts = art.findall(".//AbstractText")
            abstract = " ".join(p.text or "" for p in parts if p.text)
            yield {"pmid": pmid, "title": title, "abstract": abstract}

        done = min(i + batch, len(pmids))
        print(f"    Fetched {done:,} / {len(pmids):,} records …", end="\r")
        time.sleep(delay)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subfield", default=None,
                        help="Run one subfield only (default: all)")
    parser.add_argument("--max-per-subfield", type=int, default=0,
                        help="Cap on PMIDs to download (0 = no cap)")
    parser.add_argument("--out",  default="abstracts",
                        help="Output directory (default: abstracts/)")
    parser.add_argument("--email",   default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--delay", type=float, default=0.4)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    subfields = (
        {args.subfield: SUBFIELD_QUERIES[args.subfield]}
        if args.subfield else SUBFIELD_QUERIES
    )

    for sf_name, query in subfields.items():
        slug     = sf_name.lower().replace(" ", "_")
        out_file = out_dir / f"{slug}.jsonl"

        print(f"\n{'='*60}")
        print(f"  {sf_name}")
        print(f"{'='*60}")

        # Load already-downloaded PMIDs so we can resume
        seen = set()
        if out_file.exists():
            with out_file.open() as f:
                for line in f:
                    try:
                        seen.add(json.loads(line)["pmid"])
                    except Exception:
                        pass
            print(f"    Resuming — {len(seen):,} records already on disk")

        print("  Fetching PMID list …")
        all_pmids, total = get_all_pmids(
            query, email=args.email, api_key=args.api_key, delay=args.delay
        )

        pmids = [p for p in all_pmids if p not in seen]
        if args.max_per_subfield:
            pmids = pmids[:args.max_per_subfield]
        print(f"    New PMIDs to download: {len(pmids):,}")

        if not pmids:
            print("    Nothing new to download.")
            continue

        print("  Downloading abstracts …")
        with out_file.open("a") as f:
            for record in fetch_records(
                pmids,
                email=args.email,
                api_key=args.api_key,
                batch=200,
                delay=args.delay,
            ):
                record["subfield"] = sf_name
                f.write(json.dumps(record) + "\n")

        total_on_disk = sum(1 for _ in out_file.open())
        print(f"    Done. {total_on_disk:,} records in {out_file}")


if __name__ == "__main__":
    main()
