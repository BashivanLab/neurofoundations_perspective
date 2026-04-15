"""
2_extract_tasks_llm.py
----------------------
Runs a local Qwen instruct model over downloaded abstracts to extract
experimental task/paradigm names.

Input:   abstracts/<subfield>.jsonl  (from 1_download_abstracts.py)
Output:  extracted/<subfield>.jsonl
         Each line: {"pmid": "...", "subfield": "...", "tasks": ["task A", "task B"]}

Resumable: already-processed PMIDs are skipped.

Usage:
    python 2_extract_tasks_llm.py
    python 2_extract_tasks_llm.py --model Qwen/Qwen2.5-7B-Instruct
    python 2_extract_tasks_llm.py --model Qwen/Qwen3-8B
    python 2_extract_tasks_llm.py --subfield "Working Memory"
    python 2_extract_tasks_llm.py --batch-size 16 --device cuda
    python 2_extract_tasks_llm.py --quantize 4bit   # requires bitsandbytes

Requirements:
    pip install torch transformers accelerate
    pip install bitsandbytes   # optional, for 4-bit quantization
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import List

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


def build_prompt(title, abstract, tokenizer, enable_thinking=False):
    """Build the chat-formatted prompt for the model."""
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": USER_TEMPLATE.format(
            title=title or "(no title)",
            abstract=abstract or "(no abstract)"
        )},
    ]
    # Qwen3 supports an enable_thinking flag; for structured extraction
    # we disable it to get deterministic JSON output faster.
    kwargs = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
    except TypeError:
        # Older tokenizers don't support enable_thinking
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


# ---------------------------------------------------------------------------
# JSON extraction from model output
# ---------------------------------------------------------------------------

def parse_tasks(raw: str) -> List[str]:
    """Extract a list of task names from model output.

    The model should return a bare JSON array; this function handles common
    failure modes: markdown fences, extra text, malformed JSON.
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="Qwen/Qwen3.5-9B",
                        help="HuggingFace model ID (default: Qwen/Qwen3.5-9B)")
    parser.add_argument("--subfield",   default=None)
    parser.add_argument("--in-dir",     default="abstracts")
    parser.add_argument("--out-dir",    default="extracted")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max new tokens for generation (default: 256)")
    parser.add_argument("--device",     default="auto",
                        help="'auto', 'cuda', 'cpu', 'mps' (default: auto)")
    parser.add_argument("--quantize",   default=None,
                        choices=["4bit", "8bit"],
                        help="Optional quantization to reduce GPU memory")
    parser.add_argument("--thinking",   action="store_true",
                        help="Enable Qwen3 thinking mode (slower, not recommended for extraction)")
    args = parser.parse_args()

    # --- imports here so the script is importable without torch installed ---
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

    # --- Load model once ---
    print(f"\nLoading model: {args.model}")
    print(f"  quantize={args.quantize}  device={args.device}  batch_size={args.batch_size}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if args.quantize == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif args.quantize == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device != "cpu" else torch.float32,
        device_map=args.device,
        quantization_config=quant_config,
    )
    model.eval()
    print("  Model loaded.\n")

    enable_thinking = args.thinking if hasattr(args, "thinking") else False

    # --- Process each subfield file ---
    for in_file in inputs:
        sf_name  = in_file.stem.replace("_", " ").title()
        out_file = out_dir / in_file.name

        # Load already-processed PMIDs
        seen = set()
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
        print(f"  {sf_name}  ({len(records):,} records to process, {len(seen):,} already done)")
        print(f"{'='*60}")

        if not records:
            print("  Nothing to do.\n")
            continue

        t0 = time.time()
        n_with_tasks = 0

        with out_file.open("a") as out_f:
            for start in range(0, len(records), args.batch_size):
                batch = records[start : start + args.batch_size]

                prompts = [
                    build_prompt(r["title"], r["abstract"], tokenizer,
                                 enable_thinking=enable_thinking)
                    for r in batch
                ]

                inputs_enc = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                ).to(model.device)

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

                # Decode only the newly generated tokens
                input_len = inputs_enc["input_ids"].shape[1]
                for rec, out_ids in zip(batch, outputs):
                    new_ids   = out_ids[input_len:]
                    raw_text  = tokenizer.decode(new_ids, skip_special_tokens=True)
                    tasks     = parse_tasks(raw_text)
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
