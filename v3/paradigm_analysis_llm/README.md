# Paradigm Concentration — LLM Strategy

Uses a local Qwen instruct model to extract task/paradigm names from
PubMed abstracts. Unlike the keyword-search approach in `paradigm_analysis/`,
this strategy discovers the task list from the data itself, giving a more
complete and unbiased picture of what paradigms are actually used.

## Pipeline overview

```
PubMed (empirical filter)
       │
       ▼
1_download_abstracts.py   →  abstracts/<subfield>.jsonl
       │
       ▼
2_extract_tasks_llm.py    →  extracted/<subfield>.jsonl
       │
       ▼
3_aggregate_tasks.py      →  task_frequencies.json
                              task_queries.json
                              task_frequencies.csv
       │
       ▼
4_fetch_task_counts.py    →  pubmed_counts.json
       │
       ▼
5_plot_results.py         →  figures/
```

## Setup

```bash
pip install -r requirements.txt
# Optional 4-bit quantization (saves ~50% GPU memory):
pip install bitsandbytes
```

## Step 1 — Download abstracts

Downloads all titles + abstracts matching the empirical-article filter
(original research only; excludes reviews, meta-analyses, editorials).

```bash
python 1_download_abstracts.py --email your@email.com
```

Output: `abstracts/working_memory.jsonl`, `abstracts/decision_making.jsonl`, …
Resumable: re-running skips already-downloaded records.

Options:
```
--max-per-subfield 5000   # cap downloads (useful for testing)
--api-key YOUR_KEY        # NCBI API key → 10x faster
```

## Step 2 — Extract task names with LLM

Two backends are supported: **vLLM** (recommended for multi-GPU servers) and
**transformers** (single GPU or CPU).

### Backend A: vLLM (multi-GPU server, recommended)

Install vLLM and start the server in one terminal:

```bash
pip install vllm

vllm serve Qwen/Qwen3.5-27B-FP8 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --max-model-len 262144 \
    --reasoning-parser qwen3
```

Wait until `Application startup complete` appears, then run the extraction
in a second terminal:

```bash
python 2_extract_tasks_llm.py \
    --model Qwen/Qwen3.5-27B-FP8 \
    --vllm-url http://localhost:8000 \
    --workers 64
```

`--workers` controls concurrent requests to the server (default: 64). Increase
it if GPU utilization is low; the server handles all batching internally.

**GPU requirements for Qwen3.5-27B-FP8:**
| GPUs | VRAM total | Notes |
|---|---|---|
| 2× A5000 | 48 GB | fits comfortably |
| 4× A5000 | 96 GB | headroom for larger context |
| 8× A5000 | 192 GB | maximum throughput |

### Backend B: transformers (single GPU / CPU)

```bash
# Default model (Qwen2.5-7B-Instruct, ~15GB VRAM)
python 2_extract_tasks_llm.py

# Qwen3-8B
python 2_extract_tasks_llm.py --model Qwen/Qwen3-8B

# With 4-bit quantization (~6GB VRAM, requires bitsandbytes)
python 2_extract_tasks_llm.py --quantize 4bit

# Apple Silicon (MPS)
python 2_extract_tasks_llm.py --model Qwen/Qwen2.5-7B-Instruct --device mps --batch-size 4

# CPU-only (slow but works)
python 2_extract_tasks_llm.py --device cpu --batch-size 1
```

**Model choice:** Any Qwen instruct model works. Recommended:
| Model | VRAM (fp16) | VRAM (4-bit) | Speed (A100) |
|---|---|---|---|
| Qwen/Qwen2.5-7B-Instruct | ~15 GB | ~5 GB | ~80 rec/s |
| Qwen/Qwen3-8B | ~16 GB | ~5 GB | ~75 rec/s |
| Qwen/Qwen2.5-14B-Instruct | ~28 GB | ~9 GB | ~45 rec/s |

> **Note:** The FP8 variant (`Qwen3.5-27B-FP8`) requires vLLM for correct
> inference. Loading it with plain transformers will silently ignore the FP8
> scale factors and produce incorrect outputs.

Output: `extracted/<subfield>.jsonl`
Each line: `{"pmid": "...", "subfield": "...", "tasks": ["n-back task", ...]}`
Resumable: re-running skips already-processed PMIDs.

## Step 3 — Aggregate task names

Counts task frequencies, merges surface variants, builds PubMed queries.

```bash
python 3_aggregate_tasks.py --min-count 3
```

Options:
```
--min-count 3     # ignore tasks seen fewer than 3 times (reduces noise)
--top 100         # keep top-100 tasks per subfield
--no-cluster      # skip fuzzy edit-distance deduplication
```

Output:
- `task_frequencies.json` — LLM-based mention counts per task
- `task_queries.json` — PubMed query strings for each task
- `task_frequencies.csv` — spreadsheet for inspection

## Step 4 — Fetch PubMed counts

Queries PubMed for each discovered task to get publication counts.

```bash
python 4_fetch_task_counts.py --email your@email.com
```

Output: `pubmed_counts.json`

## Step 5 — Plot results

```bash
python 5_plot_results.py --dpi 300 --fmt pdf
```

Output figures:
| File | Description |
|---|---|
| `00_coverage.png` | TOTAL vs UNION — how many papers name any task |
| `01_summary_concentration.png` | Top-1/3, Gini, HHI across subfields |
| `02_lorenz_curves.png` | Lorenz curves |
| `03_cumulative_concentration.png` | Cumulative concentration curves |
| `04–07_*.png` | Per-subfield bar + pie charts |
| `08_llm_vs_pubmed.png` | Scatter: LLM rank vs PubMed rank (validation) |

## What makes this defensible to reviewers

1. **No pre-specified paradigm list** — task names are extracted from the
   data itself by the LLM; analyst judgment is removed from paradigm selection.

2. **Empirical-article filter** — TOTAL denominator excludes reviews,
   meta-analyses, editorials, and papers without participant/animal mentions,
   so it represents empirical original research only.

3. **Two-level framing**:
   - Coverage rate (UNION/TOTAL): what fraction of empirical papers name
     a specific paradigm — itself a finding about field transparency.
   - Concentration (within UNION): how skewed the distribution is among
     papers that do name a paradigm.

4. **Validation figure** (08_llm_vs_pubmed.png): shows that LLM mention
   rank correlates with PubMed search count rank, validating that both
   approaches converge on the same paradigm hierarchy.
