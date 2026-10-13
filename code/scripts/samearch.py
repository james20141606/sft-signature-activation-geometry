#!/usr/bin/env python3
"""Band readings for the discussion-period revision (R1), recomputed from graph JSONs.

Two readings of every Qwen/Llama band in Table 1, both at full float precision:

  SAME  -- base and run graphs built on one GPU architecture (Hopper / H200). Sources:
           results/sftband_20260926/run/graphs (JIDs 18577379-82, H200) and
           results/graph_build_audit_20261005/run/graphs (JIDs 19223775-76, H200).
           PREREG_graph_build_audit_20261004.md branches LL1 (Llama) and V2 (DPO 12/60).
  ARCH  -- the archived graphs in notes/ that the submitted Table 1 read. Several pair a
           base built on L40S/A100 with run graphs built on Hopper (cpu_records.json), so
           these readings cross GPU architectures. Kept for the appendix table only.

Nothing here reads a consolidated readout (sftband_readouts.json, samebuild_recompute.json,
gba_read.py output): every number is recomputed from the edge lists, so this module is an
independent check of those readers. `--selftest` asserts (i) the five same-architecture
base rebuilds are bit-identical edge sets, (ii) the ARCH recompute rounded to four decimals
reproduces every committed notes/circuit_rewiring_*.json value it replaces, (iii) every
SAME run graph's build record names an H200, and (iv) planted edits to a family are seen.

    python3 code/scripts/samearch.py              # print both readings
    python3 code/scripts/samearch.py --selftest
"""
import json
import os
import statistics as st
import sys
from itertools import combinations

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SB = "results/sftband_20260926/run/graphs"
GBA = "results/graph_build_audit_20261005/run/graphs"
NB = "notes/attribution_graph_v1_qwen3-1p7b_qwenscope_l14to22.json"

_CACHE = {}


def edges(rel):
    if rel not in _CACHE:
        g = json.load(open(os.path.join(ROOT, rel)))
        _CACHE[rel] = frozenset(zip(g["edges_src"], g["edges_dst"]))
    return _CACHE[rel]


def dist(a, b):
    """Edge-Jaccard distance D = 1 - J over the top-N edge sets, full precision."""
    ea, eb = edges(a), edges(b)
    return 1.0 - len(ea & eb) / len(ea | eb)


SAME = {
    "sft": (f"{SB}/base_X.json", [f"{SB}/lora_s{i}.json" for i in range(4)]),
    "sft_audited": (f"{SB}/base_X.json", [f"{SB}/sft_audited.json"]),
    "llama": (f"{GBA}/llama_base_b1_full.json", [f"{GBA}/llama_lsft_s{i}_full.json" for i in range(4)]),
    "dpo12": (f"{GBA}/qwen_base_full.json", [f"{GBA}/dpo12_s{i}_full.json" for i in range(4)]),
    "dpo29": (f"{SB}/base_X.json", [f"{SB}/dpo29_s{i}.json" for i in range(4)]),
    "dpo60": (f"{GBA}/qwen_base_full.json", [f"{GBA}/dpo60_s{i}_full.json" for i in range(4)]),
    "dpomc29": (f"{GBA}/qwen_base_full.json", [f"{GBA}/dpomc29_s{i}_full.json" for i in range(4)]),
    "grpo": (f"{SB}/base_X.json", [f"{SB}/grpo_{n}.json" for n in
                                   ("shaped150", "s1_200", "s3_100", "s4_300", "s5_300")]),
    "full": (f"{SB}/base_F.json", [f"{SB}/full_s{i}.json" for i in range(4)]),
}
# k = 20 LoRA-SFT band (PREREG_sftband_20260926 analysis L): each run against the base built
# in its own job (base_X: seeds 0-3, base_L1: 4-11, base_L2: 12-19); the bases are identical.
LORA20_BASE = {i: ("base_X" if i < 4 else "base_L1" if i < 12 else "base_L2") for i in range(20)}
SAME_BASES = [f"{SB}/base_X.json", f"{SB}/base_L1.json", f"{SB}/base_L2.json",
              f"{SB}/base_F.json", f"{GBA}/qwen_base_full.json"]

ARCH = {
    "sft": (NB, [f"notes/basediff_sft_seed{i}_l14to22.json" for i in range(4)]),
    "sft_audited": (NB, ["notes/basediff_sft_l14to22.json"]),
    "llama": ("notes/tulu_v2_base_l16to25.json",
              [f"notes/basediff_llama_sft_seed{i}_l16to25.json" for i in range(4)]),
    "dpo12": (NB, [f"notes/basediff_dpo_seed{i}_l14to22.json" for i in range(4)]),
    "dpo29": (NB, [f"notes/basediff_dpo_seed{i}_step29_l14to22.json" for i in range(4)]),
    "dpo60": (NB, [f"notes/basediff_dpo_seed{i}_step60_l14to22.json" for i in range(4)]),
    "dpomc29": (NB, [f"notes/basediff_dpo_mc_seed{i}_step29_l14to22.json" for i in range(4)]),
    "grpo": (NB, ["notes/basediff_grpo_l14to22.json"] +
             [f"notes/basediff_grpo_{s}_l14to22.json" for s in ("s1", "s3", "s4", "s5", "shaped_l0p1")]),
}
# GPU that built each archived graph set (results/graph_build_audit_20261005/cpu_records.json).
ARCH_GPU = {"base (Qwen)": "L40S", "sft": "H200", "sft_audited": "A100", "dpo12": "A100",
            "dpo29": "H100", "dpo60": "H100", "dpomc29": "H200",
            "grpo": "A100 (baseline), L40S (s4, s5, shaped), H200 (s1, s3)",
            "base (Llama)": "A100", "llama": "H200"}


def family(base, runs, pairs_from=None):
    """Band and contrasts in D = 1 - J. `pairs_from` lets the k=20 family keep per-run bases."""
    pairs = [dist(a, b) for a, b in combinations(runs, 2)]
    cons = [dist(base if pairs_from is None else pairs_from[i], r) for i, r in enumerate(runs)]
    out = {"k": len(runs), "pairs": pairs, "contrasts": cons}
    if len(pairs) >= 2:
        mu, sd = st.mean(pairs), st.stdev(pairs)
        out.update(band_min=min(pairs), band_max=max(pairs), mean=mu, sd=sd,
                   k_out=sum(c > max(pairs) for c in cons),
                   # z in the paper's sign convention, z = (J - mu_J)/sd_J = -(D - mu_D)/sd_D
                   z=[-(c - mu) / sd for c in cons],
                   r_min_over_max=min(cons) / max(pairs))
    return out


def same_arch():
    fams = {k: family(b, r) for k, (b, r) in SAME.items()}
    runs = [f"{SB}/lora_s{i}.json" for i in range(20)]
    bases = [f"{SB}/{LORA20_BASE[i]}.json" for i in range(20)]
    fams["lora20"] = family(None, runs, pairs_from=bases)
    # the audited checkpoint is scored against the seeded SFT band (it is not one of the k)
    s = fams["sft"]
    c = fams["sft_audited"]["contrasts"][0]
    fams["sft_audited"].update(z=[-(c - s["mean"]) / s["sd"]], k_out=int(c > s["band_max"]))
    return fams


def archived():
    fams = {k: family(b, r) for k, (b, r) in ARCH.items()}
    s = fams["sft"]
    c = fams["sft_audited"]["contrasts"][0]
    fams["sft_audited"].update(z=[-(c - s["mean"]) / s["sd"]], k_out=int(c > s["band_max"]))
    return fams


def gap_line(f):
    """Matched-dose (29-step) SFT vs DPO comparison, no band: gap and spreads in D."""
    sft, dpo = f["sft"]["contrasts"], f["dpo29"]["contrasts"]
    gap = min(sft) - max(dpo)
    ss, sd_ = max(sft) - min(sft), max(dpo) - min(dpo)
    return {"sft": (min(sft), max(sft)), "dpo": (min(dpo), max(dpo)), "gap": gap,
            "spread_sft": ss, "spread_dpo": sd_, "ratio_sft": gap / ss, "ratio_dpo": gap / sd_,
            "mean_ratio": st.mean(sft) / st.mean(dpo)}


def fz(z, fmt):
    return ("+" if z >= 0 else "-") + (fmt % abs(z))


def f4(x):
    return "$%.4f$" % x


def table_cells(f, k=None, outside_fmt="$%d/%d$", zfmt="%.2f", z=True, extra_out=0, extra_k=0):
    """Cells 2..7 of a Table 1 row: k, seed-pair D, sigma, contrast D, outside, z range."""
    zs = sorted(f["z"])
    return ["%d" % (k if k is not None else f["k"]),
            "%s--%s" % (f4(f["band_min"]), f4(f["band_max"])),
            "$%.4f$" % f["sd"],
            "%s--%s" % (f4(min(f["contrasts"])), f4(max(f["contrasts"]))),
            outside_fmt % (f["k_out"] + extra_out, f["k"] + extra_k),
            ("$%s$ to $%s$" % (fz(zs[0], zfmt), fz(zs[-1], zfmt))) if z else "---$^{b}$"]


def run_gpus():
    """GPU name recorded for every SAME run graph (sftband manifest; gba env records)."""
    man = json.load(open(os.path.join(ROOT, "results/sftband_20260926/sftband_readouts.json")))
    gpus = {"sftband:" + k: v["gpu"] for k, v in man["manifest"]["runs"].items()}
    meta = os.path.join(ROOT, "results/graph_build_audit_20261005/run/meta")
    for jid, name in (("19223775", "env_llama_full_19223775.json"),
                      ("19223776", "env_dpo_full_19223776.json")):
        gpus["gba:" + jid] = json.load(open(os.path.join(meta, name)))["gpu_name"]
    return gpus


def consistency_problems():
    """Bases identical, H200 build records, ARCH recompute == committed receipts (4 dp).
    Returned as a list so the paper gates (check_rewiring.py) can include it."""
    bad = []
    for a, b in combinations(SAME_BASES, 2):
        if dist(a, b) != 0.0:
            bad.append("same-arch bases differ: %s vs %s (D=%r)" % (a, b, dist(a, b)))
    for k, g in run_gpus().items():
        if "H200" not in g:
            bad.append("SAME build record %s names %r, not an H200" % (k, g))
    # the ARCH recompute must reproduce the committed four-decimal receipts it replaces
    A = archived()
    receipts = [("sft", "circuit_rewiring_sft_seednull.json", "base->SFT_s", "SFT"),
                ("dpo12", "circuit_rewiring_dpo_seednull.json", "base->DPO_s", "DPO"),
                ("dpo29", "circuit_rewiring_dpo_dose29.json", "base->DPO_s", "DPO"),
                ("dpo60", "circuit_rewiring_dpo_dose60.json", "base->DPO_s", "DPO"),
                ("dpomc29", "circuit_rewiring_dpo_mc29.json", "base->DPOMC_s", "DPOMC"),
                ("llama", "circuit_rewiring_llama_sft_seednull.json", "base->LSFT_s", "LSFT")]
    for fam, rec, pre, tag in receipts:
        d = json.load(open(os.path.join(ROOT, "notes", rec)))
        want_c = sorted(round(x["edge_jaccard"], 4) for x in d["post_contrasts"] if x["pair"].startswith(pre))
        want_p = sorted(round(x["edge_jaccard"], 4) for x in d["seed_null"] if x["pair"].startswith(tag))
        got_c = sorted(round(1 - x, 4) for x in A[fam]["contrasts"])
        got_p = sorted(round(1 - x, 4) for x in A[fam]["pairs"])
        if got_c != want_c or got_p != want_p:
            bad.append("ARCH %s does not reproduce %s (contrasts %s vs %s; pairs %s vs %s)"
                       % (fam, rec, got_c, want_c, got_p, want_p))
    d6 = json.load(open(os.path.join(ROOT, "notes/circuit_rewiring_qwen_6seed.json")))
    want = sorted(round(x["edge_jaccard"], 4) for x in d6["seed_null"])
    if sorted(round(1 - x, 4) for x in A["grpo"]["pairs"]) != want:
        bad.append("ARCH grpo pairs do not reproduce circuit_rewiring_qwen_6seed.json")
    aud = [x for x in d6["post_contrasts"] if x["pair"] == "base->SFT"][0]["edge_jaccard"]
    if round(1 - A["sft_audited"]["contrasts"][0], 4) != aud:
        bad.append("ARCH audited SFT contrast does not reproduce the 6seed receipt")
    return bad


def selftest():
    bad = consistency_problems()
    # planted edit: dropping one edge from a run graph must move that family's reading
    S = same_arch()
    key = SAME["dpo12"][1][0]
    orig = _CACHE[key]
    _CACHE[key] = frozenset(list(orig)[:-50])
    moved = family(*SAME["dpo12"])["contrasts"][0] != S["dpo12"]["contrasts"][0]
    _CACHE[key] = orig
    if not moved:
        bad.append("planted 50-edge deletion in dpo12 s0 not seen by family()")
    for b in bad:
        print("FAIL", b)
    print("samearch selftest: %s" % ("FAIL" if bad else "PASS (bases identical, GPUs H200, "
                                                         "ARCH reproduces 7 receipts, plant seen)"))
    return 1 if bad else 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    for name, fams in (("SAME (H200)", same_arch()), ("ARCH (archived)", archived())):
        print("==", name)
        for k, f in fams.items():
            if "sd" not in f:
                print("  %-12s contrast %.4f z %s" % (k, f["contrasts"][0], ["%.2f" % z for z in f.get("z", [])]))
                continue
            print("  %-12s k=%-2d band %.4f-%.4f sd %.5f mean %.5f contr %.4f-%.4f out %d/%d z %+.2f..%+.2f r %.3f"
                  % (k, f["k"], f["band_min"], f["band_max"], f["sd"], f["mean"], min(f["contrasts"]),
                     max(f["contrasts"]), f["k_out"], f["k"], min(f["z"]), max(f["z"]), f["r_min_over_max"]))
        g = gap_line(fams)
        print("  gap@29: SFT %.4f-%.4f DPO %.4f-%.4f gap %.4f spreads %.4f/%.4f ratios %.2f/%.2f mean ratio %.3f"
              % (g["sft"] + g["dpo"] + (g["gap"], g["spread_sft"], g["spread_dpo"], g["ratio_sft"],
                                         g["ratio_dpo"], g["mean_ratio"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
