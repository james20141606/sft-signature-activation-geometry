#!/usr/bin/env python3
"""LoRA-DPO on Qwen3-1.7B, mirroring sft_warmup.py so the two seed nulls are
comparable (PREREGISTRATION_dpo_seed_null.md).

Implemented directly rather than via trl.DPOTrainer: trl 0.11.4's DPOTrainer is
incompatible with transformers 4.57.3 in this env
(`get_batch_samples() takes 3 positional arguments but 4 were given`, JID
15099744). Writing the loss out has two advantages beyond dodging the version
break: the optimizer settings match sft_warmup.py exactly, and the reference
model is the *same weights with the adapter disabled*, so no second model is
loaded and the reference is provably the base model.

DPO loss (Rafailov et al. 2023), beta and lr fixed by the registration:
    L = -log sigmoid( beta * [ (pol_c - ref_c) - (pol_r - ref_r) ] )
where each term is the summed log-probability of the completion tokens only.

Invoked through sft_seed.py so set_seed() runs BEFORE the model is built and
seed -> checkpoint is deterministic.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--train-parquet", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    # Dose ladder (PREREGISTRATION_dpo_dose_matched.md). --epochs cannot express a
    # dose beyond one pass: the loop is bounded by len(order), so any --epochs > 1
    # silently stops at the end of the data. --max-steps states the dose directly and
    # repeats the fixed index order to reach it. Defaults keep every committed
    # 12-step run's behaviour byte-identical.
    ap.add_argument("--max-steps", type=int, default=None,
                    help="train exactly N optimizer steps, cycling the fixed order; "
                         "overrides --epochs")
    ap.add_argument("--save-at", type=str, default=None,
                    help="comma-separated optimizer-step counts at which to save an "
                         "adapter to {out_dir}-step{n}; when given, the end-of-run "
                         "save to --out-dir is skipped")
    # Matched-construction null (PREREGISTRATION_dpo_matched_construction.md). The
    # fixed order was registered as a known property of the seed null, and the paper
    # discloses the resulting band asymmetry: the SFT null contains data-order
    # variance, the DPO null does not. This flag adds that component -- one shuffle
    # of the index order per pass through the data, RNG seeded with the run seed so
    # there is nothing to search over. Default off: every committed run's stream is
    # byte-identical.
    ap.add_argument("--reshuffle-per-epoch", action="store_true",
                    help="reshuffle the index order once per data pass, seeded by "
                         "--seed; only meaningful with --max-steps")
    return ap.parse_args()


def seq_logprob(model, input_ids, attn, prompt_lens):
    """Summed log p(completion tokens | prompt), per sequence."""
    import torch
    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits[:, :-1, :].float()
    tgt = input_ids[:, 1:]
    logp = torch.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    # mask: completion tokens only (positions >= prompt_len), and real tokens only
    idx = torch.arange(tgt.shape[1], device=tgt.device).unsqueeze(0)
    mask = (idx >= (prompt_lens.unsqueeze(1) - 1)) & (attn[:, 1:] == 1)
    return (logp * mask).sum(-1)


def main() -> int:
    a = parse_args()
    import pandas as pd

    df = pd.read_parquet(a.train_parquet)
    print(f"[data] {len(df)} preference pairs from {a.train_parquet}", flush=True)
    if len(df) < 200:
        raise SystemExit(
            f"[FLOOR] {len(df)} < 200 usable pairs; the registration declares this "
            "arm inconclusive rather than training on a thin set.")
    if a.dry_run:
        print("[dry-run] data OK; not training", flush=True)
        return 0

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.base_model, torch_dtype=torch.bfloat16, device_map="cuda")
    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()
    model.train()

    def encode(prompt, completion):
        p = tok(prompt, add_special_tokens=False).input_ids
        c = tok(completion, add_special_tokens=False).input_ids + [tok.eos_token_id]
        ids = (p + c)[: a.max_len]
        return ids, min(len(p), len(ids))

    rows = []
    for _, r in df.iterrows():
        ci, cl = encode(r["prompt"], r["chosen"])
        ri, rl = encode(r["prompt"], r["rejected"])
        rows.append((ci, cl, ri, rl))

    def collate(batch):
        seqs = [b[0] for b in batch] + [b[2] for b in batch]
        plens = [b[1] for b in batch] + [b[3] for b in batch]
        m = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), m), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(seqs), m), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
            att[i, : len(s)] = 1
        return (ids.cuda(), att.cuda(),
                torch.tensor(plens, device="cuda"), len(batch))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    per_step = a.batch_size * a.grad_accum
    order = list(range(len(rows)))
    save_at = sorted({int(s) for s in a.save_at.split(",")}) if a.save_at else []

    if a.max_steps is not None:
        n_steps = a.max_steps
        need = n_steps * per_step
        if a.reshuffle_per_epoch:
            # Matched construction: one seeded shuffle per pass. random.Random(seed)
            # rather than the global RNG so the stream depends on nothing but the
            # run seed and this flag.
            import random as _random
            rng = _random.Random(a.seed)
            stream = []
            while len(stream) < need:
                ep = order[:]
                rng.shuffle(ep)
                stream.extend(ep)
            stream = stream[:need]
        else:
            # Repeat the fixed order to reach the requested dose. No per-epoch
            # reshuffle: registered as a known property, and it is what makes the
            # ladder a single deterministic trajectory whose first rung is the
            # published 12-step run.
            reps = -(-need // len(order))
            stream = (order * reps)[:need]
    else:
        n_steps = int(len(rows) * a.epochs) // per_step
        stream = order[: int(len(order) * a.epochs)]
    print(f"[train] {len(rows)} pairs, bs {a.batch_size} x accum {a.grad_accum} "
          f"-> {n_steps} optimizer steps, lr {a.lr}, beta {a.beta}"
          + (f", save_at={save_at}" if save_at else "")
          + (", reshuffle_per_epoch" if a.reshuffle_per_epoch else ""), flush=True)

    def save(dest):
        dest.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(dest))
        tok.save_pretrained(str(dest))
        print(f"[SAVED] {dest}", flush=True)

    step, applied, acc_loss = 0, 0, 0.0
    for i in range(0, len(stream), a.batch_size):
        batch = [rows[j] for j in stream[i: i + a.batch_size]]
        if not batch:
            break
        ids, att, plens, nb = collate(batch)
        pol = seq_logprob(model, ids, att, plens)
        with torch.no_grad():
            with model.disable_adapter():        # reference == frozen base weights
                ref = seq_logprob(model, ids, att, plens)
        pol_c, pol_r = pol[:nb], pol[nb:]
        ref_c, ref_r = ref[:nb], ref[nb:]
        loss = -F.logsigmoid(a.beta * ((pol_c - ref_c) - (pol_r - ref_r))).mean()
        (loss / a.grad_accum).backward()
        acc_loss += float(loss)
        step += 1
        if step % a.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            applied += 1
            print(f"[step {applied}] loss={acc_loss / a.grad_accum:.4f}", flush=True)
            acc_loss = 0.0
            if applied in save_at:
                save(Path(str(a.out_dir) + f"-step{applied}"))
            if a.max_steps is not None and applied >= n_steps:
                break

    if applied == 0:
        raise SystemExit("[FATAL] 0 optimizer steps applied; refusing to save an "
                         "adapter identical to its initialisation.")
    if save_at:
        missed = [s for s in save_at if s > applied]
        if missed:
            raise SystemExit(f"[FATAL] only {applied} optimizer steps ran, so the "
                             f"registered checkpoint(s) {missed} were never reached; "
                             "refusing to report a dose that was not trained.")
    else:
        save(a.out_dir)
    print(f"[DONE] {applied} optimizer steps", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
