# Paradigm Concentration Analysis

Quantifies how many studies in four neuroscience subfields
(working memory, decision making, spatial navigation, attention)
are concentrated in a small number of canonical tasks,
using live PubMed article counts.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.8+ required. No other dependencies.

## Step 1 — Fetch counts from PubMed

```bash
python fetch_counts.py
```

This queries the NCBI E-utilities API (~40 requests) and saves
results to `counts.json`.  With no API key, it runs at ≤3 req/s
and takes about 15 seconds.

**Options:**
```
--email    your@email.com   # polite identifier sent to NCBI
--api-key  <key>            # free key from https://www.ncbi.nlm.nih.gov/account/
                            # raises rate limit to 10 req/s
--out      counts.json      # output file (default: counts.json)
--delay    0.34             # seconds between requests
```

## Step 2 — Generate figures

```bash
python plot_results.py
```

Reads `counts.json` and writes 7 figures to `figures/`:

| File | Description |
|---|---|
| `01_summary_concentration.png` | 3-panel: top-1/top-3 bars, Gini, HHI |
| `02_lorenz_curves.png` | Lorenz curves across all subfields |
| `03_cumulative_concentration.png` | Cumulative % vs number of paradigms |
| `04_working_memory.png` | Bar + pie for working memory |
| `05_decision_making.png` | Bar + pie for decision making |
| `06_spatial_navigation.png` | Bar + pie for spatial navigation |
| `07_attention.png` | Bar + pie for attention |

**Options:**
```
--data  counts.json   # input (default: counts.json)
--out   figures/      # output directory (created if absent)
--fmt   pdf           # png | pdf | svg (default: png)
--dpi   300           # resolution for raster formats
```

## Editing queries

All search terms live in `queries.py`.
Add, remove, or rename paradigms there — no other file needs changing.

## Metrics computed

- **Top-K concentration**: % of paradigm-tagged studies using top 1 / 3 paradigms
- **Gini coefficient**: inequality of usage (0 = equal, 1 = monopoly)
- **Normalized HHI**: Herfindahl–Hirschman concentration index
- **Lorenz curves**: visual representation of usage inequality

## Notes on interpretation

**Field restriction (`[tiab]`):** All queries are restricted to titles
and abstracts only.  This is the standard for systematic reviews — it
avoids counting papers that merely cite a paradigm in passing (in a
reference list, MeSH term, or full-text mention) while still capturing
papers where the paradigm is actually used or discussed.  Counts are
therefore *conservative* lower bounds; the relative ordering and
skewness pattern are robust to this.

**Double-counting:** A paper using both the n-back and Sternberg tasks
will appear in both counts.  The paradigm totals therefore sum to more
than the subfield TOTAL, and individual percentages should not be
interpreted as exclusive shares.
