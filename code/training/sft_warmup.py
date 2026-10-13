"""SFT warm-up of Qwen3-1.7B on BFCL v4 SFT chat-completion parquet.

Uses `trl.SFTTrainer` with a small LoRA (rank 16) on the attention modules.
Output: a checkpoint usable as the SFT-only baseline that GRPO compares
against (the M2 'SFT-only' arm).

Usage (typically inside a sbatch — see scripts/slurm/run_sft_warmup.sbatch):
    python scripts/sft_warmup.py \
        --base-model Qwen/Qwen3-1.7B \
        --train-parquet $WORK/bfcl_sft_train.parquet \
        --out-dir $WORK/checkpoints/sft-qwen3-1p7b \
        --epochs 1 \
        --lr 5e-5 \
        --batch-size 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Heavy imports gated to main() so the script's argv parsing is quick.


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--train-parquet", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    # AMENDMENT 2 of PREREGISTRATION_llama_second_backbone.md: the recipe's formatting
    # rule is "the backbone family's canonical chat template". Qwen bundles it in the
    # base tokenizer; Llama does not, so it must be named. Default None keeps every
    # committed Qwen run byte-identical.
    ap.add_argument("--chat-template-from", default=None,
                    help="HF id whose tokenizer chat_template is copied onto the "
                         "training tokenizer when the base tokenizer ships none")
    # sftband_20260926 (notes/PREREG_sftband_20260926.md): full-parameter arm of the
    # SAME recipe. All four flags default to the committed LoRA behaviour, so every
    # existing LoRA run (and the new LoRA seeds 4..19) is byte-identical in recipe.
    ap.add_argument("--full-param", action="store_true",
                    help="update all weights instead of a LoRA adapter (fp32 master "
                         "weights + bf16 autocast, matching the fp32 LoRA adapters)")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="gradient checkpointing (memory only; same math)")
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-strategy", default="epoch",
                    help="'no' skips the ~21 GB optimizer checkpoint of a full-param run")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build trainer + run 0 steps; verify the pipeline only.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    print(f"[INFO] base_model={args.base_model}")
    print(f"[INFO] train={args.train_parquet}  out={args.out_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.chat_template_from:
        donor = AutoTokenizer.from_pretrained(args.chat_template_from)
        if not donor.chat_template:
            raise SystemExit(f"[FATAL] {args.chat_template_from} has no chat_template")
        if tokenizer.chat_template:
            raise SystemExit("[FATAL] base tokenizer already ships a chat template; "
                             "refusing to overwrite it (--chat-template-from is only "
                             "for families that bundle none)")
        tokenizer.chat_template = donor.chat_template
        print(f"[INFO] chat_template copied from {args.chat_template_from}")

    theta0 = None
    if args.full_param:
        # fp32 master weights: a bf16-resident full-param update at lr 5e-5 loses most
        # Adam steps to bf16 rounding; the LoRA arm's trainable weights are fp32 too
        # (peft autocasts adapters), so fp32 trainables + bf16 autocast is the match.
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.float32, attn_implementation="eager"
        )
        model.config.use_cache = False
        theta0 = {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters()}
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] FULL-PARAM: trainable params {n_train}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
        lora = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

    ds = load_dataset("parquet", data_files=str(args.train_parquet), split="train")
    print(f"[INFO] loaded {len(ds)} SFT rows")

    # trl 0.11.x calls it `max_seq_length`; newer trl uses `max_length`.
    import inspect
    sft_kwargs: dict = dict(
        output_dir=str(args.out_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        report_to=[],
        seed=args.seed,
        max_steps=1 if args.dry_run else -1,
    )
    if args.grad_ckpt:
        sft_kwargs["gradient_checkpointing"] = True
    sig = inspect.signature(SFTConfig.__init__).parameters
    if "max_length" in sig:
        sft_kwargs["max_length"] = args.max_len
    elif "max_seq_length" in sig:
        sft_kwargs["max_seq_length"] = args.max_len
    cfg = SFTConfig(**sft_kwargs)
    # trl 0.11.x calls it `tokenizer`; trl 0.20+ renamed to `processing_class`.
    trainer_sig = inspect.signature(SFTTrainer.__init__).parameters
    tokenizer_kw = {"processing_class": tokenizer} if "processing_class" in trainer_sig else {"tokenizer": tokenizer}
    trainer = SFTTrainer(
        model=model, args=cfg,
        train_dataset=ds, **tokenizer_kw,
    )
    trainer.train()

    if not args.dry_run and args.full_param:
        # Receipt written by the trainer process: loss log + weight displacement.
        import json, math
        num = den = 0.0
        with torch.no_grad():
            for n, p in trainer.model.named_parameters():
                p0 = theta0[n]
                d = p.detach().float().cpu() - p0
                num += float((d * d).sum()); den += float((p0 * p0).sum())
        rec = {"full_param": True, "rel_weight_delta_l2": math.sqrt(num / den),
               "log_history": trainer.state.log_history,
               "global_step": trainer.state.global_step,
               "recipe": {k: getattr(args, k) for k in ("epochs", "lr", "batch_size",
                          "grad_accum", "max_len", "seed", "grad_ckpt", "logging_steps")}}
        # Save in bf16: the graph builder loads every checkpoint in bf16 anyway, and a
        # fp32 copy is 6.9 GB per seed on a near-full /scratch.
        trainer.model.to(torch.bfloat16).save_pretrained(str(args.out_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(args.out_dir))
        (args.out_dir / "train_receipt.json").write_text(json.dumps(rec, indent=2, default=str))
        print(f"[DONE] full-param saved (bf16) to {args.out_dir}; "
              f"rel_weight_delta_l2={rec['rel_weight_delta_l2']:.6g}")
    elif not args.dry_run:
        trainer.save_model(str(args.out_dir))
        tokenizer.save_pretrained(str(args.out_dir))
        print(f"[DONE] saved to {args.out_dir}")
    else:
        print("[DONE] dry-run completed (1 train step)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
