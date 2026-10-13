#!/usr/bin/env python3
"""Stamp `checkpoint` / `hf_model_path` provenance into committed attribution-graph JSONs.

Graphs built before `m2_attribution_graph.py` learned to record its `--hf-model-path`
carry no statement of WHICH weights produced them, so two graphs over different
checkpoints are indistinguishable from their contents alone. This backfills that field
for the graphs feeding the paper's headline analyses.

The mapping below is not guessed: every entry is the literal `--hf-model-path` argument
appearing next to that `--out` filename in a committed job script under
`code/scripts/slurm/`. Job scripts whose paths are shell variables ($MERGED, $out) are
deliberately NOT stamped -- their target cannot be resolved from the committed text, and
inventing provenance is worse than leaving it absent.

    python3 code/scripts/stamp_graph_provenance.py [--dry-run]

Verifies after writing and exits non-zero if any intended stamp did not land. (An earlier
ad-hoc version inserted after an anchor key that some older graphs lack, silently no-opped
on three files, and still printed success -- hence the verification pass.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NOTES = os.path.join(ROOT, "notes")

BASE = "(base backbone, no adapter)"

# filename -> (checkpoint label, hf_model_path as passed to m2_attribution_graph.py, source job script)
MAPPING = {
    # --- Qwen 1.7B headline chain
    "attribution_graph_v1_qwen3-1p7b_qwenscope_l14to22.json":
        ("Qwen/Qwen3-1.7B " + BASE, None, "run_sig_attribution_q3_1p7b_l14to22.sbatch"),
    "basediff_sft_l14to22.json":
        ("sft-qwen3-1p7b (audited SFT, LoRA adapter merged at analysis)",
         "grpo_project/checkpoints/sft-qwen3-1p7b", "run_sig_basediff_sft_full.sbatch"),
    "basediff_grpo_l14to22.json":
        ("grpo-baseline global_step_100",
         "grpo_project/checkpoints/grpo-baseline-sif-qwen3-1p7b-step2000/global_step_100/actor/huggingface",
         "run_sig_basediff_grpo_full.sbatch"),
    # --- Tulu-3 staircase (public checkpoints, literal in the job scripts)
    "tulu_base_l16to25.json":
        ("meta-llama/Llama-3.1-8B " + BASE, None, "run_sig_tulu_stair_A.sbatch"),
    "tulu_sft_l16to25.json":
        ("allenai/Llama-3.1-Tulu-3-8B-SFT", "allenai/Llama-3.1-Tulu-3-8B-SFT",
         "run_sig_tulu_stair_A.sbatch"),
    "tulu_dpo_l16to25.json":
        ("allenai/Llama-3.1-Tulu-3-8B-DPO", "allenai/Llama-3.1-Tulu-3-8B-DPO",
         "run_sig_tulu_stair_B.sbatch"),
    "tulu_rlvr_ppo_l16to25.json":
        ("allenai/Llama-3.1-Tulu-3-8B (RLVR, PPO)", "allenai/Llama-3.1-Tulu-3-8B",
         "run_sig_tulu_stair_B.sbatch"),
    "tulu_rlvr_grpo_l16to25.json":
        ("allenai/Llama-3.1-Tulu-3.1-8B (RLVR, GRPO)", "allenai/Llama-3.1-Tulu-3.1-8B",
         "run_sig_tulu_stair_C.sbatch"),
    # --- Gemma cross-family pair
    "basediff_gemma_base_l13to20.json":
        ("google/gemma-2-2b " + BASE, None, "run_sig_basediff_gemma_full.sbatch"),
    "basediff_gemma_it_l13to20.json":
        ("gemma-2-2b-it (production instruct)", "models/gemma-2-2b-it",
         "run_sig_basediff_gemma_full.sbatch"),
    # --- 8B scale arm
    "attribution_graph_v1_qwen3-8b_l14to22.json":
        ("Qwen/Qwen3-8B " + BASE, None, "run_sig_14to22 sweep"),
}

# The random-dictionary control's eight graphs additionally need their ENCODER recorded:
# without it a randsae graph is indistinguishable from a trained-SAE graph of the same
# checkpoint, and the control's whole point is that the encoder differs. Checkpoints are
# the literal graph_arm arguments in run_sig_randsae_control.sbatch.
RANDSAE_SRC = "run_sig_randsae_control.sbatch"
RANDSAE = {
    "randsae_base_l14to22.json":  ("Qwen/Qwen3-1.7B " + BASE, None),
    "randsae_sft_l14to22.json":   ("sft-qwen3-1p7b (LoRA adapter)", "grpo_project/checkpoints/sft-qwen3-1p7b"),
    "randsae_grpo_l14to22.json":  ("grpo-baseline global_step_100",
                                   "grpo_project/checkpoints/grpo-baseline-sif-qwen3-1p7b-step2000/global_step_100/actor/huggingface"),
    "randsae_s1_l14to22.json":    ("GRPO seed s1 @ step 200", "merged/grpo_s1_step200_hf"),
    "randsae_s3_l14to22.json":    ("GRPO seed s3 @ step 100", "merged/grpo_s3_step100_hf"),
    "randsae_s4_l14to22.json":    ("GRPO seed s4 @ step 300", "merged/grpo_s4_step300_hf"),
    "randsae_s5_l14to22.json":    ("GRPO seed s5 @ step 300", "merged/grpo_s5_step300_hf"),
    "randsae_shaped_l14to22.json": ("shaped-lambda0.1 GRPO @ step 150", "merged/grpo_shaped_l0p1_step150_hf"),
}
RANDSAE_ENCODER = {
    "random_sae": "random-dir",
    "random_sae_seed": 0,
    "encoder": ("random unit encoder directions, Gaussian row-normalised; d_sae=32768 "
                "matched to qwen-scope; sparsity matched by top-k calibrated to the "
                "trained SAE's mean L0 (~50)"),
    "note": ("NOT the trained qwen-scope dictionary. Without this field a random-dictionary "
             "graph is indistinguishable from a trained-SAE graph of the same checkpoint."),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    intended, wrote, missing = 0, 0, []
    full = dict(MAPPING)
    for name, (ckpt, hfpath) in RANDSAE.items():
        full[name] = (ckpt, hfpath, RANDSAE_SRC)
    for name, (ckpt, hfpath, src) in full.items():
        path = os.path.join(NOTES, name)
        if not os.path.exists(path):
            missing.append(f"{name}: file not present")
            continue
        d = json.load(open(path))
        if "edges_src" not in d:
            missing.append(f"{name}: not an attribution graph")
            continue
        intended += 1
        if d.get("checkpoint") == ckpt:
            continue  # already stamped
        d["checkpoint"] = ckpt
        d["hf_model_path"] = hfpath
        d["provenance_source"] = f"--hf-model-path in code/scripts/slurm/{src}"
        if name in RANDSAE:
            d["random_dictionary_control"] = RANDSAE_ENCODER
        if not args.dry_run:
            json.dump(d, open(path, "w"), indent=2)
        wrote += 1
        print(f"  stamped {name}  <- {ckpt}")

    # VERIFY: re-read from disk and confirm every intended stamp actually landed.
    bad = []
    if not args.dry_run:
        for name, (ckpt, _h, _s) in full.items():
            path = os.path.join(NOTES, name)
            if not os.path.exists(path):
                continue
            d = json.load(open(path))
            if "edges_src" not in d:
                continue
            if d.get("checkpoint") != ckpt:
                bad.append(f"{name}: checkpoint field did not land")

    for m in missing:
        print("  SKIP " + m)
    for b in bad:
        print("  FAIL " + b)
    print(f"stamp_graph_provenance: {wrote} written, {intended} verified target(s), "
          f"{len(bad)} failure(s) -> {'PASS' if not bad else 'FAIL'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
