#!/usr/bin/env python3
"""Run grpo_project's ``sft_warmup.py`` with the global RNG seeded *before* the model
is built, so an SFT seed null is a null over reproducible runs.

Why this wrapper exists: ``sft_warmup.py`` forwards ``--seed`` into ``SFTConfig``,
and transformers' ``Trainer.__init__`` calls ``set_seed(args.seed)`` -- but
``get_peft_model()`` has already run by then, so the LoRA A-matrix init draws from
the ambient (process-random) torch RNG. ``--seed`` alone therefore fixes only the
data order, not the initialization, and a rerun at the same seed does not
reproduce. Seeding here, before the delegate imports torch's model builders, makes
seed -> checkpoint deterministic, which is what a per-algorithm seed-noise null
needs (each null pair must be a difference between two *specified* runs).

Usage:
    python sft_seed.py --sft-script /path/to/sft_warmup.py --seed 1 [sft_warmup args...]
"""
from __future__ import annotations

import runpy
import sys

from transformers import set_seed


def main() -> int:
    argv = sys.argv[1:]
    if "--sft-script" not in argv:
        raise SystemExit("sft_seed.py: --sft-script /path/to/sft_warmup.py is required")
    i = argv.index("--sft-script")
    script = argv[i + 1]
    del argv[i:i + 2]

    if "--seed" not in argv:
        raise SystemExit("sft_seed.py: --seed N is required")
    seed = int(argv[argv.index("--seed") + 1])

    print(f"[sft_seed] set_seed({seed}) before delegating to {script}", flush=True)
    set_seed(seed)

    sys.argv = [script] + argv
    runpy.run_path(script, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
