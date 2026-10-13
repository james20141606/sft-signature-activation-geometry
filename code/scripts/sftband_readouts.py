#!/usr/bin/env python3
"""sftband_20260926 reader: rerun bands for several diffing readouts.

Pre-registration: notes/PREREG_sftband_20260926.md (committed before any result).
This reader computes, for every run family F in the manifest and every readout S,

    contrasts  D_S(base_s, R_s)   for each run R_s in F (base_s = the base built in
                                  the SAME job as R_s, so GPU/driver differences
                                  never enter a contrast; see PREREG section 3)
    band       D_S(R_s, R_t)      for all C(k,2) pairs within F

and reports k, m, band min/max/mean/SD, each contrast, n_outside (contrast more
displaced than EVERY seed pair), the Proposition-1(b) event E_b (= all k outside)
and, when E_b holds, the ceiling 1/(k+1). z is descriptive only.

Readouts (all are distances, larger = more different):
  edge   1 - edge-set Jaccard of the top-N co-activation graphs (the paper's statistic)
  wcos   1 - cosine of |w| over the union of edges, computed at full float precision
  feat   ||mu_a - mu_b||_2 / (0.5(||mu_a||+||mu_b||)), mu = mean dst-SAE code over X
  dense  mean_x ||h_a(x)-h_b(x)|| / (0.5(||h_a(x)||+||h_b(x)||)) at the dst layer
  cka    1 - linear CKA(H_a, H_b) at the dst layer (dictionary-free)
edge/wcos read the graph JSONs; feat/dense/cka read the --dump-states .npz files.

Usage:
  python sftband_readouts.py --manifest M.json --out results.json [--selftest]
Manifest: {"runs": {NAME: {"family": F, "graph": path, "states": path|null,
                            "base": BASENAME}}, "bases": {BASENAME: {"graph":..., "states":...}}}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np

GRAPH_STATS = ("edge", "wcos")
STATE_STATS = ("feat", "dense", "cka")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_edges(path):
    """Identical collapse rule to circuit_rewiring.load_edges (dup pairs summed, |w|)."""
    g = json.loads(Path(path).read_text())
    ew = {}
    for s, d, x in zip(g["edges_src"], g["edges_dst"], g["weights"]):
        ew[(int(s), int(d))] = ew.get((int(s), int(d)), 0.0) + abs(float(x))
    return ew


def d_edge(e1, e2):
    a, b = set(e1), set(e2)
    return 1.0 - len(a & b) / len(a | b)


def d_wcos(e1, e2):
    keys = list(set(e1) | set(e2))
    v1 = np.array([e1.get(k, 0.0) for k in keys], dtype=np.float64)
    v2 = np.array([e2.get(k, 0.0) for k in keys], dtype=np.float64)
    return 1.0 - float(v1 @ v2) / float(np.linalg.norm(v1) * np.linalg.norm(v2))


def load_states(path):
    z = np.load(path, allow_pickle=False)
    return {"mu": z["zdst_mean"].astype(np.float64),
            "H": z["h_dst"].astype(np.float64),
            "key": (z["prompt_idx"].tolist(), z["pos"].tolist())}


def d_feat(s1, s2):
    m1, m2 = s1["mu"], s2["mu"]
    return float(np.linalg.norm(m1 - m2) / (0.5 * (np.linalg.norm(m1) + np.linalg.norm(m2))))


def d_dense(s1, s2):
    a, b = s1["H"], s2["H"]
    num = np.linalg.norm(a - b, axis=1)
    den = 0.5 * (np.linalg.norm(a, axis=1) + np.linalg.norm(b, axis=1))
    return float(np.mean(num / den))


def d_cka(s1, s2):
    a = s1["H"] - s1["H"].mean(0, keepdims=True)
    b = s2["H"] - s2["H"].mean(0, keepdims=True)
    # linear CKA via the n x n Gram form: ||A^T B||_F^2 / (||A^T A||_F ||B^T B||_F)
    ka, kb = a @ a.T, b @ b.T
    hsic = float(np.sum(ka * kb))
    return 1.0 - hsic / float(np.sqrt(np.sum(ka * ka)) * np.sqrt(np.sum(kb * kb)))


SCORE_AGAINST = {"audited_sft": "lora_sft"}
DIST = {"edge": d_edge, "wcos": d_wcos, "feat": d_feat, "dense": d_dense, "cka": d_cka}


def band_block(contrasts: dict, pairs: dict):
    k = len(contrasts)
    pv = np.array(list(pairs.values()), dtype=np.float64)
    out = {"k": k, "m": len(pv)}
    if len(pv) < 2 or k < 1:
        out["note"] = "fewer than 2 pairs or no contrast; no band"
        out["contrasts"] = contrasts
        return out
    mean, sd = float(pv.mean()), float(pv.std(ddof=1))
    bmax = float(pv.max())
    n_out = int(sum(v > bmax for v in contrasts.values()))
    out.update({
        "band_min": float(pv.min()), "band_max": bmax, "band_mean": mean, "band_sd": sd,
        "band_n_distinct": int(len(np.unique(np.round(pv, 12)))),
        "contrasts": contrasts,
        "z_descriptive": {n: (v - mean) / sd for n, v in contrasts.items()} if sd > 0 else None,
        "n_outside": n_out,
        "E_b_all_outside": bool(n_out == k),
        "prop1b_ceiling_if_E_b": (1.0 / (k + 1)) if n_out == k else None,
        "min_contrast_over_band_max": (min(contrasts.values()) / bmax) if bmax > 0 else None,
    })
    return out


def analyse(man: dict, stats=None):
    runs, bases = man["runs"], man["bases"]
    stats = stats or list(GRAPH_STATS + STATE_STATS)
    E, S = {}, {}
    for name, spec in list(runs.items()) + [(b, v) for b, v in bases.items()]:
        if spec.get("graph") and Path(spec["graph"]).exists():
            E[name] = load_edges(spec["graph"])
        if spec.get("states") and Path(spec["states"]).exists():
            S[name] = load_states(spec["states"])
    # identical-X check: every paired readout needs the same probe states
    keys = {n: s["key"] for n, s in S.items()}
    if keys:
        ref = next(iter(keys.values()))
        bad = [n for n, k in keys.items() if k != ref]
        if bad:
            raise SystemExit(f"[FATAL] probe states differ for {bad}; paired readouts invalid")

    fams = sorted({v["family"] for v in runs.values()})
    res = {}
    for fam in fams:
        members = [n for n, v in runs.items() if v["family"] == fam]
        res[fam] = {"members": members}
        for st in stats:
            src = E if st in GRAPH_STATS else S
            ok = [n for n in members if n in src and runs[n]["base"] in src]
            if not ok:
                res[fam][st] = {"note": "inputs missing"}
                continue
            f = DIST[st]
            contrasts = {n: f(src[runs[n]["base"]], src[n]) for n in ok}
            pairs = {f"{a}<->{b}": f(src[a], src[b]) for a, b in combinations(ok, 2)}
            blk = band_block(contrasts, pairs)
            blk["pairs"] = pairs
            blk["missing"] = [n for n in members if n not in ok]
            res[fam][st] = blk
    # single checkpoints scored against another family's band (audited SFT vs LoRA band)
    for n, spec in runs.items():
        tgt = SCORE_AGAINST.get(spec["family"])
        if not tgt or tgt not in res:
            continue
        for st in stats:
            src = E if st in GRAPH_STATS else S
            blk = res[tgt].get(st, {})
            if n in src and spec["base"] in src and "band_max" in blk:
                v = DIST[st](src[spec["base"]], src[n])
                res[spec["family"]].setdefault("vs_" + tgt, {})[st] = {
                    "contrast": v, "outside": bool(v > blk["band_max"]),
                    "z_descriptive": (v - blk["band_mean"]) / blk["band_sd"]}
    # base continuity: distances among the per-job base rebuilds (+ stored base)
    bn = list(bases)
    res["_base_continuity"] = {
        st: {f"{a}<->{b}": DIST[st]((E if st in GRAPH_STATS else S)[a],
                                     (E if st in GRAPH_STATS else S)[b])
             for a, b in combinations(bn, 2)
             if a in (E if st in GRAPH_STATS else S) and b in (E if st in GRAPH_STATS else S)}
        for st in stats}
    # rebuilt-vs-stored graph continuity (PREREG section 3; reported, not a gate)
    cont = {}
    for c in man.get("continuity", []):
        if Path(c["a"]).exists() and Path(c["b"]).exists():
            ea, eb = load_edges(c["a"]), load_edges(c["b"])
            cont[c["label"]] = {"edge": d_edge(ea, eb), "wcos": d_wcos(ea, eb)}
        else:
            cont[c["label"]] = {"note": "missing"}
    res["_rebuilt_vs_stored"] = cont
    return res


def selftest():
    """Synthetic check that the reader orders things the right way round."""
    rng = np.random.default_rng(0)
    # edges: base = 0..999; runs keep 0..899 + 100 random extras (~10% move);
    # seed pairs therefore share 900 + overlap -> contrasts must sit outside the band
    # when runs share a common shift away from base.
    base = {(i, i): 1.0 for i in range(1000)}
    shared = {(5000 + i, 1): 1.0 for i in range(150)}
    runs = []
    for s in range(6):
        e = {(i, i): 1.0 for i in range(850)}
        e.update(shared)
        extra = rng.choice(np.arange(10000, 20000), 20, replace=False)
        # 20 private edges replace 20 shared-shift edges -> small seed noise
        for x in list(shared)[:20]:
            e.pop(x)
        e.update({(int(x), 2): 1.0 for x in extra})
        runs.append(e)
    c = {f"r{s}": d_edge(base, runs[s]) for s in range(6)}
    p = {f"{a}<->{b}": d_edge(runs[a], runs[b]) for a, b in combinations(range(6), 2)}
    blk = band_block(c, p)
    assert blk["E_b_all_outside"] and blk["prop1b_ceiling_if_E_b"] == 1 / 7, blk
    # null case: base is just another run -> contrasts inside band
    c0 = {f"r{s}": d_edge(runs[0], runs[s]) for s in range(1, 6)}
    p0 = {f"{a}<->{b}": d_edge(runs[a], runs[b]) for a, b in combinations(range(1, 6), 2)}
    blk0 = band_block(c0, p0)
    assert not blk0["E_b_all_outside"], blk0
    # dense readouts: identical states -> 0; CKA of a rotation -> 0
    H = rng.normal(size=(50, 8))
    Q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    s1 = {"mu": np.abs(H).mean(0), "H": H}
    s2 = {"mu": np.abs(H).mean(0), "H": H @ Q}
    assert abs(d_dense(s1, s1)) < 1e-12 and abs(d_feat(s1, s2)) < 1e-12
    assert abs(d_cka(s1, s2)) < 1e-9, d_cka(s1, s2)
    print("[selftest] OK")


def build_manifest(meta_dir, remap=None):
    """Collect the per-unit meta JSONs the sbatches write into a reader manifest."""
    def rp(x):
        if x and remap:
            old, new = remap.split("=", 1)
            return x.replace(old, new, 1)
        return x
    runs, bases, cont = {}, {}, []
    for f in sorted(Path(meta_dir).glob("*.json")):
        if any(t in f.name for t in ("_train_receipt", "_trainer_state", ".missing", "_retrained")):
            continue
        m = json.loads(f.read_text())
        spec = {"graph": rp(m["graph"]), "states": rp(m["states"]), "jid": m.get("jid"),
                "gpu": m.get("gpu"), "hf_model_path": m.get("hf_model_path")}
        if m["family"] == "base":
            bases[m["name"]] = spec
        else:
            spec.update({"family": m["family"], "base": m["base"]})
            runs[m["name"]] = spec
        if m.get("stored_graph"):
            cont.append({"label": f"{m['name']}:rebuilt_vs_stored",
                         "a": rp(m["graph"]), "b": rp(m["stored_graph"])})
    return {"runs": runs, "bases": bases, "continuity": cont}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-manifest", metavar="META_DIR")
    ap.add_argument("--remap", default=None, help="OLDPREFIX=NEWPREFIX for torch->local paths")
    ap.add_argument("--manifest")
    ap.add_argument("--out")
    ap.add_argument("--stats", nargs="*", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.build_manifest:
        man = build_manifest(a.build_manifest, a.remap)
        Path(a.out).write_text(json.dumps(man, indent=2))
        print(f"manifest: {len(man['runs'])} runs, {len(man['bases'])} bases -> {a.out}")
        return
    man = json.loads(Path(a.manifest).read_text())
    res = analyse(man, a.stats)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        commit = None
    files = {}
    for n, spec in list(man["runs"].items()) + list(man["bases"].items()):
        for kind in ("graph", "states"):
            p = spec.get(kind)
            if p and Path(p).exists():
                files[f"{n}:{kind}"] = {"path": p, "sha256": sha256(p)}
    out = {"prereg": "notes/PREREG_sftband_20260926.md", "reader": "code/scripts/sftband_readouts.py",
           "reader_commit": commit, "manifest": man, "input_sha256": files, "results": res}
    Path(a.out).write_text(json.dumps(out, indent=2))
    for fam, r in res.items():
        if fam.startswith("_"):
            continue
        for st in (a.stats or GRAPH_STATS + STATE_STATS):
            b = r.get(st, {})
            if "band_max" in b:
                print(f"{fam:<10} {st:<6} k={b['k']:>2} m={b['m']:>3} band=[{b['band_min']:.5f},"
                      f"{b['band_max']:.5f}] contrasts=[{min(b['contrasts'].values()):.5f},"
                      f"{max(b['contrasts'].values()):.5f}] outside={b['n_outside']}/{b['k']} "
                      f"E_b={b['E_b_all_outside']}")
            else:
                print(f"{fam:<10} {st:<6} {b.get('note')}")
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
