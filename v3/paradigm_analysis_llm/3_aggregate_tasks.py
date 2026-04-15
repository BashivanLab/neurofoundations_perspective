"""
3_aggregate_tasks.py
--------------------
Reads LLM extraction results and builds:
  1. A ranked frequency table of all discovered task names per subfield
  2. A normalisation map merging surface variants (edit-distance clustering)
  3. A queries.json file that can be fed directly into 4_fetch_task_counts.py

Input:   extracted/<subfield>.jsonl  (from 2_extract_tasks_llm.py)
Output:
    task_frequencies.json   — {subfield: {task_name: count, ...}}
    task_queries.json       — {subfield: {task_name: 'pubmed query', ...}}
    task_frequencies.csv    — human-readable spreadsheet

Usage:
    python 3_aggregate_tasks.py
    python 3_aggregate_tasks.py --min-count 3      # ignore tasks seen < 3 times
    python 3_aggregate_tasks.py --top 50           # keep top-50 per subfield
    python 3_aggregate_tasks.py --no-cluster       # skip fuzzy deduplication
"""

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    """Lowercase, strip punctuation variants, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower().strip()
    text = re.sub(r"[''`]", "", text)          # remove apostrophes
    text = re.sub(r"[\-–—]", "-", text)        # unify hyphens
    text = re.sub(r"\s+", " ", text)           # collapse spaces
    return text


# Hard-coded canonical forms for common surface variants.
# Add more here if needed after inspecting task_frequencies.json.
CANONICAL = {
    # working memory
    "n back task":                 "n-back task",
    "nback task":                  "n-back task",
    "n-back tasks":                "n-back task",
    "n-back paradigm":             "n-back task",
    "dual n-back task":            "n-back task",
    "2-back task":                 "n-back task",
    "2 back task":                 "n-back task",
    "3-back task":                 "n-back task",
    "digit span task":             "digit span",
    "digit span test":             "digit span",
    "forward digit span":          "digit span",
    "backward digit span":         "digit span",
    "sternberg memory task":       "sternberg task",
    "sternberg working memory task":"sternberg task",
    "delayed match to sample task":"delayed match-to-sample task",
    "delayed match-to-sample paradigm":"delayed match-to-sample task",
    "dmts task":                   "delayed match-to-sample task",
    "change detection paradigm":   "change detection task",
    "operation span task":         "complex span task",
    "reading span task":           "complex span task",
    "ospan task":                  "complex span task",
    "corsi block task":            "corsi block tapping task",
    "corsi block-tapping task":    "corsi block tapping task",
    "spatial span task":           "corsi block tapping task",
    # decision making
    "iowa gambling tasks":         "iowa gambling task",
    "igt":                         "iowa gambling task",
    "stop-signal task":            "stop signal task",
    "stop signal paradigm":        "stop signal task",
    "go nogo task":                "go/no-go task",
    "go no go task":               "go/no-go task",
    "go-nogo task":                "go/no-go task",
    "gonogo task":                 "go/no-go task",
    "temporal discounting task":   "delay discounting task",
    "intertemporal choice task":   "delay discounting task",
    "probabilistic reversal learning task":"reversal learning task",
    "probabilistic reversal task": "reversal learning task",
    "multi-armed bandit task":     "bandit task",
    "multi armed bandit task":     "bandit task",
    "k-armed bandit task":         "bandit task",
    "balloon analogue risk task":  "bart",
    "balloon analog risk task":    "bart",
    "random dot motion task":      "random dot kinematogram",
    "random dot kinematogram task":"random dot kinematogram",
    "moving dot task":             "random dot kinematogram",
    "two-alternative forced choice task":"2afc task",
    "two alternative forced choice task":"2afc task",
    "2-alternative forced choice task":  "2afc task",
    # navigation
    "morris water maze task":      "morris water maze",
    "water maze task":             "morris water maze",
    "morris maze":                 "morris water maze",
    "radial arm maze task":        "radial arm maze",
    "8-arm radial maze":           "radial arm maze",
    "barnes maze task":            "barnes maze",
    "t maze":                      "t-maze",
    "y maze":                      "y-maze",
    "virtual water maze":          "virtual morris water maze",
    # attention
    "stroop color word task":      "stroop task",
    "stroop colour word task":     "stroop task",
    "stroop color-word task":      "stroop task",
    "stroop interference task":    "stroop task",
    "stroop paradigm":             "stroop task",
    "eriksen flanker task":        "flanker task",
    "eriksen flanker paradigm":    "flanker task",
    "flanker paradigm":            "flanker task",
    "flanker tasks":               "flanker task",
    "auditory oddball task":       "oddball task",
    "visual oddball task":         "oddball task",
    "oddball paradigm":            "oddball task",
    "p300 oddball task":           "oddball task",
    "posner task":                 "posner cueing task",
    "posner paradigm":             "posner cueing task",
    "spatial cueing task":         "posner cueing task",
    "attentional blink task":      "attentional blink paradigm",
    "rapid serial visual presentation task":"rsvp task",
    "continuous performance task": "cpt",
    "conners continuous performance test": "cpt",
    "multiple object tracking task":"multiple object tracking",
    "mot task":                    "multiple object tracking",
}


def canonical(text: str) -> str:
    n = normalise(text)
    return CANONICAL.get(n, n)


# ---------------------------------------------------------------------------
# Optional fuzzy clustering (edit-distance based)
# ---------------------------------------------------------------------------

def edit_distance(a: str, b: str) -> int:
    """Simple Levenshtein distance."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                            prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def cluster_tasks(counter: Counter, threshold: int = 2) -> Counter:
    """Merge task names whose edit distance is <= threshold.
    The more-frequent variant wins; less-frequent one is absorbed.
    Only merges tasks whose lengths are similar (within 4 chars).
    """
    merged = Counter(counter)
    names  = sorted(merged.keys(), key=lambda k: -merged[k])

    for i, a in enumerate(names):
        if merged[a] == 0:
            continue
        for b in names[i + 1:]:
            if merged[b] == 0:
                continue
            if abs(len(a) - len(b)) > 4:
                continue
            if edit_distance(a, b) <= threshold:
                merged[a] += merged[b]
                merged[b]  = 0

    return Counter({k: v for k, v in merged.items() if v > 0})


# ---------------------------------------------------------------------------
# Build PubMed query string for a task name
# ---------------------------------------------------------------------------

def make_pubmed_query(task_name: str, subfield_anchor: str) -> str:
    """Create a [tiab] PubMed query for a task name + subfield anchor."""
    # Use the task name as a phrase search
    # Also try dropping trailing 'task'/'paradigm'/'maze' for broader recall
    quoted = f'"{task_name}"[tiab]'
    anchor = f'"{subfield_anchor}"[tiab]'
    return f"{quoted} AND {anchor}"


# Subfield anchor words for building queries
SUBFIELD_ANCHORS = {
    "Working Memory":    "working memory",
    "Decision Making":   "decision",
    "Spatial Navigation":"spatial",
    "Attention":         "attention",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir",     default="extracted")
    parser.add_argument("--out-dir",    default=".")
    parser.add_argument("--min-count",  type=int, default=2,
                        help="Ignore tasks mentioned fewer than N times (default: 2)")
    parser.add_argument("--top",        type=int, default=0,
                        help="Keep only top-N tasks per subfield (0 = keep all)")
    parser.add_argument("--no-cluster", action="store_true",
                        help="Skip fuzzy edit-distance clustering")
    args = parser.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    all_freqs   = {}
    all_queries = {}

    for jsonl_file in sorted(in_dir.glob("*.jsonl")):
        sf_slug = jsonl_file.stem
        sf_name = sf_slug.replace("_", " ").title()

        raw_counter = Counter()
        n_records   = 0
        n_with_task = 0

        with jsonl_file.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                n_records += 1
                tasks = rec.get("tasks", [])
                if tasks:
                    n_with_task += 1
                for t in tasks:
                    raw_counter[canonical(t)] += 1

        print(f"\n{sf_name}")
        print(f"  Records: {n_records:,}  |  with named task: {n_with_task:,}  "
              f"({n_with_task/max(n_records,1)*100:.1f}%)")
        print(f"  Unique task names (raw): {len(raw_counter):,}")

        # Fuzzy cluster
        if not args.no_cluster:
            counter = cluster_tasks(raw_counter, threshold=2)
            print(f"  Unique task names (after clustering): {len(counter):,}")
        else:
            counter = raw_counter

        # Apply min-count filter
        counter = Counter({k: v for k, v in counter.items()
                           if v >= args.min_count})
        print(f"  Unique task names (min_count>={args.min_count}): {len(counter):,}")

        # Apply top-N filter
        if args.top:
            counter = Counter(dict(counter.most_common(args.top)))

        # Show top-20 in console
        print(f"\n  {'Rank':<5} {'Task name':<50} {'Count':>6}")
        print(f"  {'-'*4}  {'-'*49}  {'-'*6}")
        for rank, (name, count) in enumerate(counter.most_common(20), 1):
            print(f"  {rank:<5} {name:<50} {count:>6,}")
        if len(counter) > 20:
            print(f"  … ({len(counter) - 20} more)")

        all_freqs[sf_name] = dict(counter.most_common())

        # Build PubMed queries for each task
        anchor = SUBFIELD_ANCHORS.get(sf_name, sf_name.lower())
        all_queries[sf_name] = {
            task: make_pubmed_query(task, anchor)
            for task in counter
        }
        # Add TOTAL and UNION placeholders (to be filled by 4_fetch_task_counts.py)
        all_queries[sf_name]["_TOTAL"] = f'"{anchor}"[tiab]'

    # Save outputs
    freqs_path   = out_dir / "task_frequencies.json"
    queries_path = out_dir / "task_queries.json"
    csv_path     = out_dir / "task_frequencies.csv"

    freqs_path.write_text(json.dumps(all_freqs, indent=2))
    queries_path.write_text(json.dumps(all_queries, indent=2))

    # CSV output
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subfield", "rank", "task_name", "llm_mention_count"])
        for sf, tasks in all_freqs.items():
            for rank, (task, count) in enumerate(tasks.items(), 1):
                writer.writerow([sf, rank, task, count])

    print(f"\n✓ task_frequencies.json  → {freqs_path}")
    print(f"✓ task_queries.json      → {queries_path}")
    print(f"✓ task_frequencies.csv   → {csv_path}")
    print(f"\nNext step: python 4_fetch_task_counts.py")


if __name__ == "__main__":
    main()
