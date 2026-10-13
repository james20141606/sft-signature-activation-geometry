#!/usr/bin/env python3
"""Build BFCL preference pairs from receipts, not judgement (PREREGISTRATION_dpo_seed_null.md).

chosen   = the gold assistant turn (what SFT trained on)
rejected = a base-model greedy generation that is VERIFIABLY WRONG (string-unequal
           to gold after normalisation)

Rows where the base model emits the gold call are dropped -- no pair exists -- and
the dropped count is reported. Greedy decoding, so the set is deterministic and all
seeds train on the identical preference data: the null must isolate seed variance,
not data variance.

Run ONCE, before any seed is trained:
  python build_dpo_pairs.py --train-parquet data/bfcl_sft_train.parquet \
      --base-model Qwen/Qwen3-1.7B --out data/bfcl_dpo_pairs.parquet
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def norm(s: str) -> str:
    """Whitespace/quote-insensitive comparison, so a formatting difference is not
    scored as a wrong call."""
    s = str(s).strip().replace("'", '"')
    return re.sub(r"\s+", "", s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", type=Path, required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows (smoke: 32)")
    a = ap.parse_args()

    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    df = pd.read_parquet(a.train_parquet)
    if a.limit:
        df = df.iloc[: a.limit]
    print(f"[data] {len(df)} rows from {a.train_parquet}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        a.base_model, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    prompts, golds = [], []
    for _, r in df.iterrows():
        msgs = list(r["messages"])
        gold = next(m["content"] for m in reversed(msgs) if m["role"] == "assistant")
        ctx = [m for m in msgs if m["role"] != "assistant"]
        # AMENDMENT 1: enable_thinking=False. Without it Qwen3 emits a <think>
        # monologue for every prompt, so "rejected" was a reasoning block rather
        # than a wrong function call -- the preference signal would have been
        # FORMAT (concise call vs monologue), not correctness. Verified: 900/900
        # first-pass generations began with <think> and 0/900 parsed as a call.
        try:
            prompts.append(tok.apply_chat_template(
                ctx, tokenize=False, add_generation_prompt=True,
                enable_thinking=False))
        except TypeError:      # tokenizer without the kwarg
            prompts.append(tok.apply_chat_template(
                ctx, tokenize=False, add_generation_prompt=True))
        golds.append(gold)

    gens = []
    for i in range(0, len(prompts), a.batch_size):
        batch = prompts[i:i + a.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        for j in range(len(batch)):
            gens.append(tok.decode(out[j, enc.input_ids.shape[1]:],
                                   skip_special_tokens=True).strip())
        print(f"[gen] {min(i + a.batch_size, len(prompts))}/{len(prompts)}", flush=True)

    rows, dropped, nonstruct = [], 0, 0
    call_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\(")
    for p, g, gen in zip(prompts, golds, gens):
        gen = gen.split("</think>")[-1].strip()   # belt and braces
        if not gen or norm(gen) == norm(g):
            dropped += 1          # base already correct (or empty) -> no pair
            continue
        if not call_re.match(gen):
            nonstruct += 1        # not a function call at all: a format contrast,
            continue              # not a correctness contrast (AMENDMENT 1)
        rows.append({"prompt": p, "chosen": g, "rejected": gen})

    out = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out)
    print(f"[pairs] usable={len(out)}  dropped(base correct/empty)={dropped}  "
          f"dropped(not a call -> format contrast)={nonstruct}  of {len(df)}",
          flush=True)
    print(f"[SAVED] {a.out}", flush=True)
    if len(out) < 200:
        print("[FLOOR] fewer than 200 usable pairs -- per the registration this arm "
              "is INCONCLUSIVE and no seed should be trained on it.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
