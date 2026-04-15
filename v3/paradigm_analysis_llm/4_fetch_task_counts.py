"""
4_fetch_task_counts.py
----------------------
Fetches PubMed [tiab] counts for every task discovered by the LLM,
using the query strings built by 3_aggregate_tasks.py.

Input:   task_queries.json   (from 3_aggregate_tasks.py)
Output:  pubmed_counts.json
         {subfield: {task_name: count, "_TOTAL": count, "_UNION": count}}

The _UNION count is computed as a single OR query of all task names,
giving the number of empirical papers that mention ANY tracked task.

Usage:
    python 4_fetch_task_counts.py
    python 4_fetch_task_counts.py --queries task_queries.json
    python 4_fetch_task_counts.py --email you@uni.edu --api-key YOUR_KEY
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


def post_count(query: str, email="", api_key="") -> int:
    params = {"db": "pubmed", "term": query, "rettype": "count", "retmode": "xml"}
    if email:   params["email"]   = email
    if api_key: params["api_key"] = api_key
    body = urllib.parse.urlencode(params).encode()
    req  = urllib.request.Request(EUTILS, data=body, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req, timeout=20) as r:
        root = ET.fromstring(r.read())
    elem = root.find("Count")
    if elem is None:
        raise ValueError("No <Count> in response")
    return int(elem.text.strip())


def build_union_query(task_queries: dict, anchor_query: str) -> str:
    """OR all task name phrases together with the subfield anchor."""
    # Extract just the quoted phrase part from each query
    # e.g. '"n-back task"[tiab] AND "working memory"[tiab]'
    # → '"n-back task"[tiab]'
    phrases = []
    for label, q in task_queries.items():
        if label.startswith("_"):
            continue
        # The phrase is the first quoted [tiab] term
        import re
        m = re.search(r'"[^"]+"?\[tiab\]', q)
        if m:
            phrases.append(m.group())
    if not phrases:
        return anchor_query
    union_phrase = "(" + " OR ".join(phrases) + ")"
    return f"{union_phrase} AND {anchor_query}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries",  default="task_queries.json")
    parser.add_argument("--out",      default="pubmed_counts.json")
    parser.add_argument("--email",    default="")
    parser.add_argument("--api-key",  default="")
    parser.add_argument("--delay",    type=float, default=0.4)
    args = parser.parse_args()

    queries_path = Path(args.queries)
    all_queries  = json.loads(queries_path.read_text())

    results = {}

    for sf_name, task_queries in all_queries.items():
        results[sf_name] = {}
        total_queries = len(task_queries) + 1  # +1 for UNION
        done = 0

        print(f"\n{'='*60}")
        print(f"  {sf_name}  ({len(task_queries)} queries + UNION)")
        print(f"{'='*60}")

        # Fetch TOTAL first
        total_query = task_queries.get("_TOTAL", "")
        if total_query:
            try:
                count = post_count(total_query, args.email, args.api_key)
                results[sf_name]["TOTAL"] = count
                print(f"  ★ {'TOTAL':<50} {count:>8,}")
            except Exception as e:
                print(f"  ✗ TOTAL: {e}")
                results[sf_name]["TOTAL"] = None
            time.sleep(args.delay)

        # Fetch individual task counts
        for label, query in task_queries.items():
            if label.startswith("_"):
                continue
            try:
                count = post_count(query, args.email, args.api_key)
                results[sf_name][label] = count
                print(f"    {label:<50} {count:>8,}")
            except Exception as e:
                print(f"  ✗ {label:<50} ERROR: {e}")
                results[sf_name][label] = None
            done += 1
            time.sleep(args.delay)

        # Fetch UNION (POST handles long queries)
        anchor_query = task_queries.get("_TOTAL", f'"{sf_name.lower()}"[tiab]')
        union_query  = build_union_query(task_queries, anchor_query)
        try:
            union_count = post_count(union_query, args.email, args.api_key)
            results[sf_name]["UNION"] = union_count
            print(f"  ∪ {'UNION':<50} {union_count:>8,}")
        except Exception as e:
            print(f"  ✗ UNION: {e}")
            results[sf_name]["UNION"] = None
        time.sleep(args.delay)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Counts saved to {out_path.resolve()}")
    print(f"\nNext step: python 5_plot_results.py --data {out_path}")


if __name__ == "__main__":
    main()
