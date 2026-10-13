#!/usr/bin/env python3
"""Verify the Signature release and its core statistics."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "main.tex"

# Receipt-driven replacement for a hardcoded [0, 2, 4] literal (final review 2026-10-04,
# F4 / mirrors the same fix in check_rewiring.py): the EXPECTED dose curve is parsed out
# of main.tex's own claim sentence rather than hand-maintained as a Python constant, so a
# future revision is cross-checked against the receipts instead of silently going stale.
# Revision R1 (2026-10-07): the paper states the curve on ONE GPU architecture and, separately,
# the archived (cross-architecture) curve; both are recomputed here from the released graph
# JSONs by code/scripts/samearch.py (same one-sided rule as the paper) and compared.
DOSE_SENTENCE_RE = re.compile(
    r"Under one architecture \$K_\{\\mathrm\{out\}\}\(\\tau\)=(\d+),(\d+),(\d+)\$ of \$4\$ "
    r"although the contrasts \$c_s\(\\tau\)\$ are flat")
ARCHIVED_DOSE_RE = re.compile(
    r"The archived graphs, whose base was built on a different GPU architecture from most run "
    r"graphs, gave \$(\d+),(\d+),(\d+)\$")
REQUIRED = [
    "README.md",
    "main.tex",
    "code/scripts/samearch.py",
    "code/scripts/gba_read.py",
    "results/sftband_20260926/sftband_readouts.json",
    "results/sftband_20260926/sftband_verdict.json",
    "results/sftband_20260926/run/graphs/base_X.json",
    "results/graph_build_audit_20261005/run/graphs/qwen_base_full.json",
    "results/graph_build_audit_20261005/run/meta/env_dpo_full_19223776.json",
    "results/graph_build_audit_20261005/run/meta/env_llama_full_19223775.json",
    "results/graph_build_audit_20261005/run/dpo_readout_full.json",
    "results/graph_build_audit_20261005/run/llama_readout_full.json",
    "results/graph_build_audit_20261005/run/battery_same_env_descriptive.json",
    "preregistrations/PREREG_graph_build_audit_20261004.md",
    "MANIFEST.sha256",
    "notes/circuit_rewiring_sft_seednull.json",
    "notes/circuit_rewiring_dpo_seednull.json",
    "notes/circuit_rewiring_dpo_dose29.json",
    "notes/circuit_rewiring_dpo_dose60.json",
    "notes/circuit_rewiring_dpo_mc29.json",
    "notes/circuit_rewiring_llama_sft_seednull.json",
    "notes/functional_transfer_k5000.json",
    "notes/functional_transfer_full_heldout_count.json",
    "notes/functional_transfer_full_A_k2000.json",
    "notes/functional_transfer_full_A_k5000.json",
    "notes/functional_transfer_full_A_k10000.json",
    "notes/functional_transfer_full_B_k2000.json",
    "notes/functional_transfer_full_B_k5000.json",
    "notes/functional_transfer_full_B_k10000.json",
    "notes/functional_transfer_full_C_k5000.json",
    "code/scripts/functional_transfer_read.py",
    "code/scripts/functional_transfer_full.py",
    "code/scripts/functional_transfer_sweep_read.py",
    "code/scripts/functional_transfer_tex.py",
    "code/scripts/check_functional_transfer.py",
    "preregistrations/PREREG_functional_transfer_full_20260924.md",
    "code/scripts/check_rewiring.py",
    "code/scripts/circuit_rewiring.py",
    "code/scripts/m2_attribution_graph.py",
    "code/scripts/audit_conventions.py",
    "code/scripts/sig4_h3.py",
    "code/scripts/stamp_graph_provenance.py",
    "code/attribution_fn.py",
    "code/attribution_graph.py",
    "code/random_sae_baseline.py",
    "code/analysis/fig_main_seednull_v2.py",
    "code/tests/test_dpo_loss.py",
    "code/training/sft_warmup.py",
    "code/training/dpo_warmup.py",
    "data/bfcl_sft_train.parquet",
    "data/bfcl_dpo_pairs.parquet",
    "LICENSES/BFCL-Apache-2.0.txt",
    "results/sftband_20260926/r0_stored/r0_stored_graphs.json",
    "results/sftband_20260926/r0_stored/manifest.json",
    "code/scripts/sftband_readouts.py",
    "preregistrations/PREREG_sftband_20260926.md",
    "code/training/prepare_sft_from_bfcl.py",
    "code/training/build_dpo_pairs.py",
    "preregistrations/PREREGISTRATION_dpo_dose_matched.md",
    "preregistrations/PREREGISTRATION_functional_transfer.md",
    "preregistrations/PREREGISTRATION_llama_second_backbone_v2.md",
    "preregistrations/PREREGISTRATION_dpo_matched_construction.md",
    "preregistrations/PREREG_matched_step300_equivalence.md",
]


def main() -> int:
    missing = [p for p in REQUIRED if not (ROOT / p).is_file()]
    if missing:
        print("MISSING: " + ", ".join(missing))
        return 1

    bad = []
    for line in (ROOT / "MANIFEST.sha256").read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        rel = rel.lstrip("*").removeprefix("./")
        path = ROOT / rel
        if not path.is_file():
            bad.append(f"manifest path missing: {rel}")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            bad.append(f"hash mismatch: {rel}")
    if bad:
        print("\n".join("FAIL: " + x for x in bad))
        return 1

    proc = subprocess.run(
        [sys.executable, str(ROOT / "code/scripts/check_rewiring.py")],
        cwd=ROOT, text=True, capture_output=True,
    )
    print(proc.stdout, end="")
    if proc.returncode:
        print(proc.stderr, end="", file=sys.stderr)
        return proc.returncode

    sys.path.insert(0, str(ROOT / "code" / "scripts"))
    import samearch  # noqa: E402
    problems = samearch.consistency_problems()
    if problems:
        print("\n".join("FAIL: samearch " + x for x in problems))
        return 1
    tex = TEX.read_text()
    for label, fams, rx in (("same-architecture", samearch.same_arch(), DOSE_SENTENCE_RE),
                            ("archived", samearch.archived(), ARCHIVED_DOSE_RE)):
        counts = [fams[k]["k_out"] for k in ("dpo12", "dpo29", "dpo60")]
        m = rx.search(tex)
        assert m is not None, "main.tex no longer states the %s DPO dose curve" % label
        stated = [int(m.group(i)) for i in (1, 2, 3)]
        assert counts == stated, (label, counts, stated)
        print("DPO %s counts at 12/29/60 steps: %s (stated %s)" % (label, counts, stated))
    print("release: required files, manifest, core stats, and DPO curves (same-architecture + archived) PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
