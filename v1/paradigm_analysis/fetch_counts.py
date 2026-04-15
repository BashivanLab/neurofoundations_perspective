"""
fetch_counts.py
---------------
Queries PubMed E-utilities for article counts for every paradigm
defined in queries.py and saves results to counts.json.

All queries use the [tiab] field tag, restricting matches to titles and
abstracts only (excludes MeSH terms, author keywords, full-text, etc.).
This is standard practice in systematic reviews and avoids inflating
counts with papers that merely cite a paradigm in passing.

Usage:
    python fetch_counts.py

Optional flags:
    --email your@email.com   Passed to NCBI as a courtesy identifier
    --out   counts.json      Output file (default: counts.json)
    --delay 0.34             Seconds between requests (default: 0.34,
                             respects NCBI's 3 req/s limit without an API key)
    --api-key <key>          NCBI API key (allows 10 req/s; get one free at
                             https://www.ncbi.nlm.nih.gov/account/)
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from queries import SUBFIELDS


# ---------------------------------------------------------------------------
# PubMed helpers
# ---------------------------------------------------------------------------

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


def fetch_count(query: str, email: str = "", api_key: str = "") -> int:
    """Return the number of PubMed records matching *query*.

    Uses HTTP POST so that long UNION queries (which can exceed URL length
    limits with GET) are sent in the request body instead.
    """
    params = {
        "db":      "pubmed",
        "term":    query,
        "rettype": "count",
        "retmode": "xml",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key

    body = urllib.parse.urlencode(params).encode()
    req  = urllib.request.Request(EUTILS_BASE, data=body, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    with urllib.request.urlopen(req, timeout=20) as resp:
        root = ET.fromstring(resp.read())
    count_elem = root.find("Count")
    if count_elem is None:
        raise ValueError(f"No <Count> in response for query: {query!r}")
    return int(count_elem.text.strip())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch PubMed paradigm counts.")
    parser.add_argument("--email",   default="", help="Your e-mail (NCBI courtesy)")
    parser.add_argument("--api-key", default="", help="NCBI API key (optional, raises rate limit)")
    parser.add_argument("--out",     default="counts.json", help="Output JSON file")
    parser.add_argument("--delay",   type=float, default=0.34,
                        help="Seconds between requests (default 0.34 → ≤3 req/s)")
    args = parser.parse_args()

    # Reduce delay when an API key is provided (10 req/s allowed)
    delay = 0.11 if args.api_key else args.delay

    results = {}

    total_queries = sum(len(v) for v in SUBFIELDS.values())
    done = 0

    for subfield, paradigms in SUBFIELDS.items():
        results[subfield] = {}
        print(f"\n{'='*60}")
        print(f"  {subfield}")
        print(f"{'='*60}")

        for label, query in paradigms.items():
            try:
                count = fetch_count(query, email=args.email, api_key=args.api_key)
                results[subfield][label] = count
                marker = "★" if label == "TOTAL" else ("∪" if label == "UNION" else " ")
                print(f"  {marker} {label:<35} {count:>8,}")
            except Exception as exc:
                print(f"  ✗ {label:<35} ERROR: {exc}")
                results[subfield][label] = None
            done += 1
            if done < total_queries:
                time.sleep(delay)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Counts saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
