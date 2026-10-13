#!/usr/bin/env python3
"""Reader for graph_build_audit_20261004 (notes/PREREG_graph_build_audit_20261004.md).

Written 2026-10-05 before any job of this audit ran. Stdlib only. It reads graph JSONs
(edges_src, edges_dst, weights) and featstats JSONs and applies the pre-registered rules:

  llama   §1  continuity C_det/C_base/C_s, same-env band and contrasts, K_out, r_same, branch LL*
  dpo     §3  per-graph rebuilt-vs-stored, per-dose counts, branch V*, dose wording rule
  battery §4  featstats recomputed vs stored at stored precision, branch B*
  arch    §2  rebuilt-vs-stored for an arch-matched rebuild (R1/R7 test): D_edge and weight equality

D_edge(G,H) = 1 - |E(G) & E(H)| / |E(G) | E(H)| over (src,dst) edge sets. "Outside" is the
paper's one-sided rule c_s > max(seed-pair D). "Bit-exact" = same edge list and same weights.
`--selftest` plants known inputs for every rule and must print OK.
"""
import argparse, itertools, json, math, os, sys


def load_graph(p):
    g = json.load(open(p))
    return list(zip(g["edges_src"], g["edges_dst"])), list(g["weights"])


def d_edge(a, b):
    ea, eb = set(a), set(b)
    return 1.0 - len(ea & eb) / len(ea | eb)


def bit_exact(ga, gb):
    return ga[0] == gb[0] and ga[1] == gb[1]


def band(graphs):
    return [d_edge(graphs[i][0], graphs[j][0]) for i, j in itertools.combinations(range(len(graphs)), 2)]


def score(base, runs):
    b = band(runs)
    c = [d_edge(base[0], r[0]) for r in runs]
    hi = max(b)
    return {"band": [min(b), hi], "band_pairs": b, "contrasts": c,
            "k_out": sum(x > hi for x in c), "r": min(c) / hi}


# ---- §1 Llama -------------------------------------------------------------------------------
def llama_branch(c_det, c_base, c_s, k_out, r_same, r_stored):
    h = 1 + (r_stored - 1) / 2
    if c_det > 0:
        return "LL0", h
    if c_base == 0 and all(x == 0 for x in c_s):
        return "LL-C", h
    if k_out == 4 and r_same >= h:
        return "LL1", h
    if k_out == 4:
        return "LL2", h
    return "LL3", h


def run_llama(a):
    sb, ss = load_graph(a.stored_base), [load_graph(p) for p in a.stored_seeds]
    b1, b2 = load_graph(a.base1), load_graph(a.base2)
    rs = [load_graph(p) for p in a.rebuilt_seeds]
    stored = score(sb, ss)
    same = score(b1, rs)
    c_det, c_base = d_edge(b1[0], b2[0]), d_edge(sb[0], b1[0])
    c_s = [d_edge(x[0], y[0]) for x, y in zip(ss, rs)]
    br, h = llama_branch(c_det, c_base, c_s, same["k_out"], same["r"], stored["r"])
    return {"section": "§1 llama", "r_stored_full_precision": stored["r"], "h": h,
            "stored": stored, "same_env": same, "C_det": c_det, "C_base": c_base, "C_s": c_s,
            "bit_exact": {"base": bit_exact(sb, b1), "seeds": [bit_exact(x, y) for x, y in zip(ss, rs)]},
            "branch": br}


# ---- §3 DPO ---------------------------------------------------------------------------------
def dose_wording(c12, c29, c60):
    if c12 <= c29 <= c60 and c60 - c12 >= 2:
        return "keep 'band membership is dose-dependent' with counts %d/4, %d/4, %d/4" % (c12, c29, c60)
    return ("replace: under a same-environment base the counts (%d/4, %d/4, %d/4) show no consistent "
            "dose ordering; the 0/4, 2/4, 4/4 curve was within the build difference" % (c12, c29, c60))


def run_dpo(a):
    out = {"section": "§3 dpo", "doses": {}}
    base = load_graph(a.base)
    any_diff = False
    for spec in a.dose:   # name:stored1,stored2,..:rebuilt1,rebuilt2,..
        name, st, rb = spec.split(":")
        st, rb = st.split(","), rb.split(",")
        S, R = [load_graph(p) for p in st], [load_graph(p) for p in rb]
        per = [{"stored": s, "rebuilt": r, "D_edge": d_edge(x[0], y[0]), "bit_exact": bit_exact(x, y)}
               for s, r, x, y in zip(st, rb, S, R)]
        any_diff |= not all(p["bit_exact"] for p in per)
        out["doses"][name] = {"per_graph": per, "same_env": score(base, R)}
    out["branch"] = "V2" if any_diff else "V1"
    k = {n: out["doses"][n]["same_env"]["k_out"] for n in out["doses"]}
    if all(n in k for n in ("dpo12", "dpo60")) and a.c29 is not None:
        out["dose_wording"] = dose_wording(k["dpo12"], a.c29, k["dpo60"])
    return out


# ---- §4 battery -----------------------------------------------------------------------------
def decimals(x):
    s = repr(float(x))
    return 0 if "e" in s or "." not in s else len(s.split(".")[1].rstrip("0"))


def equal_at_stored(stored, new):
    if isinstance(stored, int) and not isinstance(stored, bool):
        return stored == new
    d = decimals(stored)
    return round(float(new), d) == round(float(stored), d)


SKIP = {"backbone", "hf_model_path", "sae_release", "dst_hook", "elapsed_seconds", "dst_layer",
        "d_sae", "n_prompts", "seq_len", "fvu", "fve"}


def compare_featstats(stored_p, new_p):
    s, n = json.load(open(stored_p)), json.load(open(new_p))
    rows = {}
    for k, v in s.items():
        if k in SKIP or not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        rows[k] = {"stored": v, "recomputed": n.get(k), "equal": k in n and equal_at_stored(v, n[k])}
    return rows


def run_battery(a):
    res = {}
    for spec in a.pair:   # name:stored:recomputed
        name, st, rc = spec.split(":")
        res[name] = compare_featstats(st, rc)
    ok = {k: all(r["equal"] for r in v.values()) for k, v in res.items()}
    return {"section": "§4 battery", "per_file": res, "all_equal": ok, "branch": battery_branch(ok)}


def battery_branch(ok):
    """ok: {'base': bool, 's1': bool, ...}. Needs base + four s* files, else B-NR."""
    base_ok = ok.get("base")
    s_ok = [ok[k] for k in ok if k != "base"]
    if base_ok is None or len(s_ok) < 4:
        return "B-NR"
    if base_ok and all(s_ok):
        return "B1"
    if not all(s_ok):
        return "B3"
    return "B2"


# ---- §2 arch --------------------------------------------------------------------------------
def run_arch(a):
    out = {"section": "§2 arch (R1/R7)", "pairs": []}
    for spec in a.pair:   # name:stored:rebuilt
        name, st, rb = spec.split(":")
        S, R = load_graph(st), load_graph(rb)
        out["pairs"].append({"name": name, "D_edge": d_edge(S[0], R[0]), "bit_exact": bit_exact(S, R)})
    if a.repeat:
        A, B = load_graph(a.repeat[0]), load_graph(a.repeat[1])
        out["same_job_repeat"] = {"D_edge": d_edge(A[0], B[0]), "bit_exact": bit_exact(A, B)}
    base = [p for p in out["pairs"] if p["name"] == "base"]
    out["R1_confirmed_for_base"] = bool(base and base[0]["bit_exact"])
    return out


# ---- selftest -------------------------------------------------------------------------------
def selftest():
    import random, tempfile
    rnd = random.Random(0)
    tmp = tempfile.mkdtemp()

    def mk(name, edges, w=None):
        p = os.path.join(tmp, name + ".json")
        json.dump({"edges_src": [e[0] for e in edges], "edges_dst": [e[1] for e in edges],
                   "weights": w or [1.0] * len(edges)}, open(p, "w"))
        return p

    univ = [(i, j) for i in range(60) for j in range(60)]
    base = rnd.sample(univ, 400)
    # seeds: share 380 edges with each other, 340 with base -> contrasts > band
    core = base[:300]
    rest = [e for e in univ if e not in set(base)]
    shared = rnd.sample(rest, 80)
    seeds = [core + shared + rnd.sample([e for e in rest if e not in set(shared)], 20) for _ in range(4)]
    assert abs(d_edge(base, base)) < 1e-12 and abs(d_edge(base, list(reversed(base)))) < 1e-12
    assert abs(d_edge([(0, 0), (0, 1)], [(0, 1), (0, 2)]) - (1 - 1 / 3)) < 1e-12
    s = score((base, None), [(x, None) for x in seeds])
    assert s["k_out"] == 4 and s["r"] > 1, s
    # LL rules
    assert llama_branch(0.01, 0, [0] * 4, 4, 2.0, 1.69)[0] == "LL0"
    assert llama_branch(0, 0, [0] * 4, 4, 1.69, 1.69)[0] == "LL-C"
    assert llama_branch(0, 0.03, [0] * 4, 4, 1.40, 1.69)[0] == "LL1"
    assert llama_branch(0, 0.03, [0] * 4, 4, 1.30, 1.69)[0] == "LL2"
    assert llama_branch(0, 0.03, [0] * 4, 3, 1.40, 1.69)[0] == "LL3"
    assert abs(llama_branch(0, 0.03, [0] * 4, 4, 1.4, 1.69)[1] - 1.345) < 1e-12
    # dose wording
    assert dose_wording(1, 1, 3).startswith("keep") and dose_wording(0, 2, 4).startswith("keep")
    assert dose_wording(1, 2, 2).startswith("replace") and dose_wording(2, 1, 4).startswith("replace")
    # battery precision equality
    assert equal_at_stored(0.6833, 0.68334) and not equal_at_stored(0.6833, 0.6836)
    assert equal_at_stored(31906, 31906) and not equal_at_stored(31906, 31905)
    a = mk("fs_a", [(0, 0)]); json.dump({"firing_freq_gini": 0.6833, "n_tokens": 31906}, open(a, "w"))
    b = mk("fs_b", [(0, 0)]); json.dump({"firing_freq_gini": 0.68329, "n_tokens": 31906}, open(b, "w"))
    c = mk("fs_c", [(0, 0)]); json.dump({"firing_freq_gini": 0.6840, "n_tokens": 31906}, open(c, "w"))
    assert all(r["equal"] for r in compare_featstats(a, b).values())
    assert not all(r["equal"] for r in compare_featstats(a, c).values())
    T, F = True, False
    assert battery_branch({"base": T, "s1": T, "s3": T, "s4": T, "s5": T}) == "B1"
    assert battery_branch({"base": F, "s1": T, "s3": T, "s4": T, "s5": T}) == "B2"
    assert battery_branch({"base": T, "s1": T, "s3": F, "s4": T, "s5": T}) == "B3"
    assert battery_branch({"base": F, "s1": F, "s3": T, "s4": T, "s5": T}) == "B3"
    assert battery_branch({"base": T, "s1": T, "s3": T}) == "B-NR"
    assert battery_branch({"base": T, "s1": T, "s3": T, "s4": T}) == "B-NR"
    # bit-exact must see a weight change even when edges agree
    g1 = mk("g1", base, [1.0] * 400); g2 = mk("g2", base, [1.0] * 399 + [1.0001])
    assert d_edge(load_graph(g1)[0], load_graph(g2)[0]) == 0 and not bit_exact(load_graph(g1), load_graph(g2))
    print("gba_read selftest OK")


def main():
    if "--selftest" in sys.argv:
        selftest(); return
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("llama")
    l.add_argument("--stored-base", required=True); l.add_argument("--stored-seeds", nargs=4, required=True)
    l.add_argument("--base1", required=True); l.add_argument("--base2", required=True)
    l.add_argument("--rebuilt-seeds", nargs=4, required=True)
    d = sub.add_parser("dpo")
    d.add_argument("--base", required=True); d.add_argument("--dose", nargs="+", required=True)
    d.add_argument("--c29", type=int, default=1, help="verified same-env 29-step count (samebuild_recompute.json)")
    b = sub.add_parser("battery"); b.add_argument("--pair", nargs="+", required=True)
    r = sub.add_parser("arch"); r.add_argument("--pair", nargs="+", required=True)
    r.add_argument("--repeat", nargs=2)
    for p in (l, d, b, r):
        p.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"llama": run_llama, "dpo": run_dpo, "battery": run_battery, "arch": run_arch}[a.cmd](a)
    res["prereg"] = "notes/PREREG_graph_build_audit_20261004.md"
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k in ("section", "branch", "C_det", "C_base",
                                                            "R1_confirmed_for_base", "dose_wording")}))


if __name__ == "__main__":
    main()
