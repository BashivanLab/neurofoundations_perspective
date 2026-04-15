"""
2_extract_tasks_llm.py
----------------------
Runs a Qwen instruct model over downloaded abstracts to extract
experimental task/paradigm names.

Supports two backends:
  1. vLLM  (recommended for multi-GPU servers) — point at a running vLLM server
  2. transformers (fallback for single-GPU or CPU)

Input:   abstracts/<subfield>.jsonl  (from 1_download_abstracts.py)
Output:  extracted/<subfield>.jsonl
         Each line: {"pmid": "...", "subfield": "...", "tasks": ["task A", "task B"]}

Resumable: already-processed PMIDs are skipped.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
vLLM backend (recommended — start server first):

    vllm serve Qwen/Qwen3.5-27B-FP8 \\
        --port 8000 \\
        --tensor-parallel-size 8 \\
        --max-model-len 262144 \\
        --reasoning-parser qwen3

    python 2_extract_tasks_llm.py \\
        --model Qwen/Qwen3.5-27B-FP8 \\
        --vllm-url http://localhost:8000 \\
        --workers 64

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
transformers backend (single GPU / CPU):

    python 2_extract_tasks_llm.py --model Qwen/Qwen2.5-14B-Instruct --device auto
    python 2_extract_tasks_llm.py --model Qwen/Qwen2.5-7B-Instruct --device mps
    python 2_extract_tasks_llm.py --quantize 4bit   # requires bitsandbytes

Requirements:
    pip install transformers accelerate          # transformers backend
    pip install vllm                             # vLLM backend (server only)
    pip install bitsandbytes                     # optional quantization
"""

import argparse
import json
import re
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a neuroscience expert. Your task is to extract the names of experimental tasks or paradigms from research paper abstracts.

Rules:
- Extract ONLY named experimental tasks/paradigms actually used in the study (e.g., "n-back task", "Morris water maze", "Iowa Gambling Task").
- Do NOT extract generic descriptions like "cognitive task", "behavioral task", or "memory test" unless a specific name is given.
- Do NOT extract outcome measures, brain regions, or statistical methods.
- If multiple tasks are mentioned, list all of them.
- If no specific named task is mentioned, return an empty list.
- Return ONLY a valid JSON array of strings. No explanation, no markdown."""

USER_TEMPLATE = """Title: {title}

Abstract: {abstract}

Return a JSON array of task/paradigm names used in this study."""


def build_messages(title: str, abstract: str) -> List[Dict[str, str]]:
    """Build the chat messages list (used by both backends)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_TEMPLATE.format(
            title=title or "(no title)",
            abstract=abstract or "(no abstract)",
        )},
    ]


def build_prompt_str(title: str, abstract: str, tokenizer,
                     enable_thinking: bool = False) -> str:
    """Format messages as a single string for the transformers backend."""
    messages = build_messages(title, abstract)
    kwargs: Dict[str, Any] = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


# ---------------------------------------------------------------------------
# JSON extraction from model output
# ---------------------------------------------------------------------------

def parse_tasks(raw: str) -> List[str]:
    """Extract a list of task names from model output.

    Handles common failure modes: markdown fences, extra text,
    <think> blocks (Qwen3 reasoning), malformed JSON.
    """
    raw = raw.strip()

    # Strip <think>...</think> blocks (Qwen3 thinking mode)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Try direct parse first
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return [str(t).strip().lower() for t in result if t]
    except json.JSONDecodeError:
        pass

    # Find a JSON array anywhere in the output
    m = re.search(r'\[.*?\]', raw, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group())
            if isinstance(result, list):
                return [str(t).strip().lower() for t in result if t]
        except json.JSONDecodeError:
            pass

    # Last resort: extract quoted strings
    items = re.findall(r'"([^"]{3,80})"', raw)
    if items:
        return [t.strip().lower() for t in items]

    return []


# ---------------------------------------------------------------------------
# vLLM backend
# ---------------------------------------------------------------------------

def call_vllm_one(messages: List[Dict[str, str]],
                  base_url: str,
                  model_name: str,
                  max_tokens: int = 256,
                  retries: int = 3) -> str:
    """POST one chat-completion request to the vLLM server.

    Returns the assistant's content string.
    Thinking is disabled via chat_template_kwargs so we get clean JSON.
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = json.dumps({
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        # Disable Qwen3 thinking for deterministic JSON extraction
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=payload, method="POST")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=60) as r:
                resp = json.loads(r.read())
            return resp["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)   # brief back-off
    return ""


def run_vllm(records: List[Dict],
             base_url: str,
             model_name: str,
             max_tokens: int = 256,
             workers: int = 64) -> List[List[str]]:
    """Process a list of records concurrently against the vLLM server.

    Returns a list of task-name lists, one per record, in the same order.
    """
    results: List[Optional[List[str]]] = [None] * len(records)

    def process_one(idx: int, rec: Dict) -> tuple:
        messages = build_messages(rec["title"], rec["abstract"])
        raw = call_vllm_one(messages, base_url, model_name, max_tokens)
        return idx, parse_tasks(raw)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_one, i, rec): i
                   for i, rec in enumerate(records)}
        for future in as_completed(futures):
            try:
                idx, tasks = future.result()
                results[idx] = tasks
            except Exception as e:
                idx = futures[future]
                print(f"\n  ✗ record {idx}: {e}")
                results[idx] = []

    return [r if r is not None else [] for r in results]


def check_vllm(base_url: str) -> bool:
    """Return True if the vLLM server is reachable."""
    try:
        url = f"{base_url.rstrip('/')}/health"
        with urllib.request.urlopen(url, timeout=5):
            return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    # Shared
    parser.add_argument("--model",      default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--subfield",   default=None,
                        help="Process only this subfield (default: all)")
    parser.add_argument("--in-dir",     default="abstracts")
    parser.add_argument("--out-dir",    default="extracted")
    parser.add_argument("--max-tokens", type=int, default=256)

    # vLLM backend
    parser.add_argument("--vllm-url",   default=None,
                        help="vLLM server base URL (e.g. http://localhost:8000). "
                             "If set, use vLLM instead of transformers.")
    parser.add_argument("--workers",    type=int, default=64,
                        help="Concurrent requests to vLLM server (default: 64)")

    # transformers backend
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device",     default="auto",
                        help="'auto', 'cuda', 'cpu', 'mps'")
    parser.add_argument("--quantize",   default=None, choices=["4bit", "8bit"],
                        help="Optional quantization (transformers backend only)")
    parser.add_argument("--thinking",   action="store_true",
                        help="Enable Qwen3 thinking mode (not recommended for extraction)")
    args = parser.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # Collect input files
    if args.subfield:
        slug   = args.subfield.lower().replace(" ", "_")
        inputs = [in_dir / f"{slug}.jsonl"]
    else:
        inputs = sorted(in_dir.glob("*.jsonl"))

    if not inputs:
        print(f"No .jsonl files found in {in_dir}/")
        return

    # ------------------------------------------------------------------ #
    # Select backend                                                       #
    # ------------------------------------------------------------------ #
    use_vllm = args.vllm_url is not None

    if use_vllm:
        print(f"\nBackend : vLLM  ({args.vllm_url})")
        print(f"Model   : {args.model}")
        print(f"Workers : {args.workers} concurrent requests")
        if not check_vllm(args.vllm_url):
            print(f"\n⚠  Cannot reach vLLM server at {args.vllm_url}")
            print("   Make sure it is running, e.g.:")
            print(f"   vllm serve {args.model} --port 8000 --tensor-parallel-size 8\n")
            return
        print("  Server reachable ✓\n")
    else:
        # transformers backend — lazy import
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"\nBackend : transformers")
        print(f"Model   : {args.model}")
        print(f"  quantize={args.quantize}  device={args.device}  batch_size={args.batch_size}")

        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Detect MPS (Apple Silicon)
        use_mps = (args.device in ("mps", "auto")
                   and torch.backends.mps.is_available()
                   and args.device != "cpu")
        if args.device == "auto" and use_mps:
            print("  Detected Apple MPS backend.")

        quant_config = None
        if args.quantize in ("4bit", "8bit"):
            if use_mps:
                print("  ⚠ bitsandbytes not supported on MPS — ignoring --quantize.")
            else:
                if args.quantize == "4bit":
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                else:
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)

        if use_mps:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.float16,
            ).to("mps")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.float16 if args.device != "cpu" else torch.float32,
                device_map=args.device if args.device != "mps" else None,
                quantization_config=quant_config,
            )
        model.eval()
        print("  Model loaded.\n")
        enable_thinking = args.thinking

    # ------------------------------------------------------------------ #
    # Process each subfield file                                           #
    # ------------------------------------------------------------------ #
    for in_file in inputs:
        sf_name  = in_file.stem.replace("_", " ").title()
        out_file = out_dir / in_file.name

        # Load already-processed PMIDs
        seen: set = set()
        if out_file.exists():
            with out_file.open() as f:
                for line in f:
                    try:
                        seen.add(json.loads(line)["pmid"])
                    except Exception:
                        pass

        # Load records to process
        records = []
        with in_file.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r["pmid"] not in seen:
                        records.append(r)
                except Exception:
                    pass

        print(f"{'='*60}")
        print(f"  {sf_name}  ({len(records):,} to process, {len(seen):,} already done)")
        print(f"{'='*60}")

        if not records:
            print("  Nothing to do.\n")
            continue

        t0 = time.time()
        n_with_tasks = 0

        with out_file.open("a") as out_f:

            if use_vllm:
                # ---- vLLM path: process in chunks, write as we go --------
                CHUNK = args.workers * 4   # keep ~4 rounds of work queued
                for chunk_start in range(0, len(records), CHUNK):
                    chunk = records[chunk_start : chunk_start + CHUNK]
                    task_lists = run_vllm(
                        chunk, args.vllm_url, args.model,
                        args.max_tokens, args.workers,
                    )
                    for rec, tasks in zip(chunk, task_lists):
                        n_with_tasks += int(bool(tasks))
                        out_f.write(json.dumps({
                            "pmid":     rec["pmid"],
                            "subfield": rec.get("subfield", sf_name),
                            "tasks":    tasks,
                        }) + "\n")
                    out_f.flush()

                    done    = min(chunk_start + CHUNK, len(records))
                    elapsed = time.time() - t0
                    rate    = done / elapsed
                    eta     = (len(records) - done) / rate if rate > 0 else 0
                    print(
                        f"  {done:>7,}/{len(records):,}  "
                        f"({done/len(records)*100:.1f}%)  "
                        f"{rate:.1f} rec/s  "
                        f"ETA {eta/60:.1f} min  "
                        f"with-tasks: {n_with_tasks}",
                        end="\r",
                    )

            else:
                # ---- transformers path: batched local inference ----------
                for start in range(0, len(records), args.batch_size):
                    batch = records[start : start + args.batch_size]

                    prompts = [
                        build_prompt_str(r["title"], r["abstract"],
                                         tokenizer, enable_thinking)
                        for r in batch
                    ]

                    device = next(model.parameters()).device
                    inputs_enc = tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=1024,
                    ).to(device)

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs_enc,
                            max_new_tokens=args.max_tokens,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )

                    input_len = inputs_enc["input_ids"].shape[1]
                    for rec, out_ids in zip(batch, outputs):
                        raw_text = tokenizer.decode(
                            out_ids[input_len:], skip_special_tokens=True
                        )
                        tasks = parse_tasks(raw_text)
                        n_with_tasks += int(bool(tasks))
                        out_f.write(json.dumps({
                            "pmid":     rec["pmid"],
                            "subfield": rec.get("subfield", sf_name),
                            "tasks":    tasks,
                        }) + "\n")

                    done    = min(start + args.batch_size, len(records))
                    elapsed = time.time() - t0
                    rate    = done / elapsed
                    eta     = (len(records) - done) / rate if rate > 0 else 0
                    print(
                        f"  {done:>7,}/{len(records):,}  "
                        f"({done/len(records)*100:.1f}%)  "
                        f"{rate:.1f} rec/s  "
                        f"ETA {eta/60:.1f} min  "
                        f"with-tasks: {n_with_tasks}",
                        end="\r",
                    )

        print(f"\n  Done. {n_with_tasks}/{len(records)} records had named tasks.")
        print(f"  Output: {out_file}\n")


if __name__ == "__main__":
    main()
