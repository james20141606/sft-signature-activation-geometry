#!/usr/bin/env python3
"""Assert the seed-noise-null band reproduces from the committed rewiring receipts,
and that each receipt still names its own inputs.

The matched-band paragraph draws base->post and seed-pair edge-Jaccards from
circuit_rewiring_4seed_step300.json (s1 @ step 300); the 15-pair standardised
statistic (mean 0.9626, SD 0.0016) comes from circuit_rewiring_qwen_6seed.json
(s1 @ step 200). The two receipts report a pair called base->s1 with DIFFERENT
values precisely because s1 exists at two checkpoints -- so the check also asserts
that each receipt records the input path that disambiguates them, which is the
provenance that was missing until 2026-07-17.

    python3 code/scripts/check_rewiring.py             # exit 1 on drift or missing provenance
    python3 code/scripts/check_rewiring.py --selftest  # prove the dose-curve tex-cross-check can FAIL
"""
import json
import os
import re
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N = os.path.join(ROOT, "notes")
TEX = os.path.join(ROOT, "main.tex")


def load(name):
    return json.load(open(os.path.join(N, name)))


# Receipt-driven replacement for a hardcoded [0, 2, 4] literal (final review
# 2026-10-04, F4): the EXPECTED dose curve is now parsed out of main.tex's own
# claim sentence instead of being a human-maintained Python constant, so a
# future revision that changes the claimed counts is cross-checked against the
# receipts automatically rather than silently going stale. The pattern is long
# and includes the clause on both sides of the numbers (sentence-anchored), so
# a one-digit edit, a reordering, or a sentence-neighbour swap all break the
# match rather than silently matching a nearby unrelated "0,2,4".
# Revision R1 (2026-10-07): the sentence now states the SAME-ARCHITECTURE curve, and a second
# sentence states the archived (cross-architecture) curve; both are cross-checked, each against
# its own recompute (code/scripts/samearch.py, one-sided rule c_s < min d as in the paper).
DOSE_SENTENCE_RE = re.compile(
    r"Under one architecture \$K_\{\\mathrm\{out\}\}\(\\tau\)=(\d+),(\d+),(\d+)\$ of \$4\$ "
    r"although the contrasts \$c_s\(\\tau\)\$ are flat")
ARCHIVED_DOSE_RE = re.compile(
    r"The archived graphs, whose base was built on a different GPU architecture from most run "
    r"graphs, gave \$(\d+),(\d+),(\d+)\$, a curve that lay within the build difference")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import samearch  # noqa: E402


def expected_outside_counts(tex, rx=None):
    """The paper's own stated K_out(tau) curve at (12, 29, 60) steps, or None if the
    pinned sentence is no longer present verbatim."""
    m = (rx or DOSE_SENTENCE_RE).search(tex)
    return [int(m.group(i)) for i in (1, 2, 3)] if m else None


def jac(d, key, pair):
    for x in d[key]:
        if x["pair"] == pair:
            return x["edge_jaccard"]
    return None


def check(tex):
    bad = []

    d4 = load("circuit_rewiring_4seed_step300.json")
    # the three matched base->post@300 contrasts and the 3 seed-pair band values
    want4 = {"base->GRPOs1": 0.961, "base->GRPOs4": 0.9639, "base->GRPOs5": 0.9631,
             "GRPOs1<->GRPOs4": 0.9616, "GRPOs1<->GRPOs5": 0.9621, "GRPOs4<->GRPOs5": 0.9644}
    allrows4 = {x["pair"]: x["edge_jaccard"] for x in d4["post_contrasts"] + d4["seed_null"]}
    for pair, v in want4.items():
        if abs(allrows4.get(pair, -9) - v) > 1e-9:
            bad.append("4seed %s: receipt %s != paper %s" % (pair, allrows4.get(pair), v))
    # provenance: s1 must resolve to the step-300 graph
    s1_4 = (d4.get("inputs") or {}).get("GRPOs1", "")
    if "step300" not in s1_4:
        bad.append("4seed receipt does not record s1 as the step-300 graph (inputs=%s)" % s1_4)

    d6 = load("circuit_rewiring_qwen_6seed.json")
    band = [x["edge_jaccard"] for x in d6["seed_null"]]
    mean, sd = st.mean(band), st.stdev(band)
    if len(band) != 15:
        bad.append("6seed band has %d pairs, expected 15" % len(band))
    if abs(round(mean, 4) - 0.9626) > 1e-4:
        bad.append("6seed band mean %.4f != 0.9626" % mean)
    if abs(round(sd, 4) - 0.0016) > 1e-4:
        bad.append("6seed band SD %.4f != 0.0016" % sd)
    s1_6 = (d6.get("inputs") or {}).get("GRPO_s1", "")
    if "step300" in s1_6 or not s1_6:
        bad.append("6seed receipt s1 should be the step-200 graph, got %r" % s1_6)

    # ---- the two PER-ALGORITHM nulls the paper's headline z's are scored against.
    # Both receipts report `base->SFT` with the SAME value; what differs is the null
    # they are scored against, so the verdict block -- not the contrast -- is the thing
    # to assert. The DPO arm additionally has to keep its NEGATIVE verdict and its
    # precision-adequacy flag: a paper that quotes z=-1.02 while the receipt says the
    # statistic is not precision-adequate would be quoting a rounding artifact, which
    # is the exact defect this repo withdrew a z=-30.7 over.
    for fname, key, want in [
        ("circuit_rewiring_sft_seednull.json", "base_to_SFT",
         {"z": -19.9, "seed_null_mean": 0.9644, "seed_null_sd": 0.001367,
          "seed_null_n_pairs": 6, "precision_adequate": True,
          "rl_signature_beyond_seed_noise": True}),
        ("circuit_rewiring_dpo_seednull.json", "base_to_DPO_s0",
         {"z": -1.02, "seed_null_mean": 0.96705, "seed_null_sd": 0.002202,
          "seed_null_n_pairs": 6, "precision_adequate": True,
          "rl_signature_beyond_seed_noise": False}),
    ]:
        try:
            v = load(fname)["verdict"]["edge_jaccard"]
        except (OSError, KeyError) as e:
            bad.append("%s: unreadable edge_jaccard verdict (%s)" % (fname, e))
            continue
        for k, exp in want.items():
            got = v.get(k)
            same = (got == exp) if isinstance(exp, (bool, int)) and not isinstance(exp, float) \
                else (got is not None and abs(got - exp) <= 1e-9)
            if not same:
                bad.append("%s %s: receipt %r != paper %r" % (fname, k, got, exp))
        if key not in v:
            bad.append("%s: verdict does not name its scored contrast %r" % (fname, key))

    # The DPO arm's D3 positive control: the four DPO runs must displace the graph by
    # more than the SFT seed band's width, or the registration declares the arm unable
    # to resolve and NO branch may be read. The paper quotes 6.1--6.6x; recompute it.
    try:
        dd = load("circuit_rewiring_dpo_seednull.json")
        ss = load("circuit_rewiring_sft_seednull.json")["verdict"]["edge_jaccard"]
        w95 = 3.92 * ss["seed_null_sd"]                       # 95% width of the SFT null
        disp = [1 - c["edge_jaccard"] for c in dd["post_contrasts"]
                if c["pair"].startswith("base->DPO")]
        if len(disp) != 4:
            bad.append("DPO receipt has %d base->DPO contrasts, expected 4" % len(disp))
        ratios = [d / w95 for d in disp]
        if not ratios or min(ratios) <= 1.0:
            bad.append("D3 positive control FAILS: min displacement %.4f <= SFT band "
                       "95%% width %.5f -- the arm cannot resolve and no branch is readable"
                       % (min(disp) if disp else -1, w95))
        elif round(min(ratios), 1) != 6.1 or round(max(ratios), 1) != 6.6:
            bad.append("D3 control ratios %.1f--%.1f != paper's 6.1--6.6"
                       % (min(ratios), max(ratios)))
        # and every base->DPO contrast must still sit INSIDE the DPO band (branch D2)
        lo, hi = dd["verdict"]["edge_jaccard"]["seed_null_range"]
        outside = [c["pair"] for c in dd["post_contrasts"]
                   if c["pair"].startswith("base->DPO") and not lo <= c["edge_jaccard"] <= hi]
        if outside:
            bad.append("branch D2 asserted but these base->DPO contrasts fall outside "
                       "the band %s: %s" % ([lo, hi], outside))
    except (OSError, KeyError, ZeroDivisionError) as e:
        bad.append("DPO D3 positive control not checkable (%s)" % e)

    # ---- the DOSE LADDER (PREREGISTRATION_dpo_dose_matched.md).
    # The body's dose claim rests on THREE facts that no other check covers: the
    # equal-dose displacement ranges must stay DISJOINT (that is the whole point of
    # the ladder, and it is the one comparison that invokes no seed band), and DPO's
    # displacement must stay FLAT across the ladder; and literal band membership
    # must be 0/4, 2/4, 4/4 outside at 12/29/60. The receipt boolean is a separate
    # large-effect rule (band minimum - 0.02) and must never be substituted for
    # literal band membership.
    def disp(receipt, prefix="base->DPO"):
        return [1 - c["edge_jaccard"] for c in load(receipt)["post_contrasts"]
                if c["pair"].startswith(prefix)]

    try:
        F, A = samearch.same_arch(), samearch.archived()
        bad.extend("samearch: " + x for x in samearch.consistency_problems())
        # (1) the stated curves: same-architecture (body) and archived (body + appendix)
        same_counts = [F[k]["k_out"] for k in ("dpo12", "dpo29", "dpo60")]
        arch_counts = [A[k]["k_out"] for k in ("dpo12", "dpo29", "dpo60")]
        exp_counts = expected_outside_counts(tex)
        if exp_counts is None:
            bad.append("main.tex no longer states the dose K_out curve in its pinned "
                       "sentence ('Under one architecture $K_{\\mathrm{out}}(\\tau)=...$ of $4$ "
                       "although the contrasts...are flat') -- cannot cross-check receipts against the paper")
        elif same_counts != exp_counts:
            bad.append("DPO literal band-membership curve %s (same architecture) != main.tex's "
                       "stated %s outside at 12/29/60 steps (dose paragraph)" % (same_counts, exp_counts))
        exp_arch = expected_outside_counts(tex, ARCHIVED_DOSE_RE)
        if exp_arch is None:
            bad.append("main.tex no longer states the ARCHIVED dose curve in its pinned sentence")
        elif arch_counts != exp_arch:
            bad.append("ARCHIVED DPO band-membership curve %s != main.tex's stated %s"
                       % (arch_counts, exp_arch))
        # the archived receipts themselves (two-sided rule, 4 dp) must still give the
        # archived curve: a receipt edit cannot slip past the graph recompute
        rec_counts = []
        for x in ("circuit_rewiring_dpo_seednull.json", "circuit_rewiring_dpo_dose29.json",
                  "circuit_rewiring_dpo_dose60.json"):
            rec = load(x)
            lo, hi = rec["verdict"]["edge_jaccard"]["seed_null_range"]
            rec_counts.append(sum(not lo <= c["edge_jaccard"] <= hi for c in rec["post_contrasts"]
                                  if c["pair"].startswith("base->DPO")))
        if rec_counts != arch_counts:
            bad.append("archived receipts give %s but the archived graph recompute gives %s"
                       % (rec_counts, arch_counts))
        # (2) equal-dose disjointness and mean ratio, on both builds; each ratio pinned
        # (2 dp) to the sentence that quotes it.
        for fams, label, ctx in ((F, "same-architecture",
                                  "so SFT's mean displacement is $%.2f\\times$ DPO's at equal dose"),
                                 (A, "archived", "ratio $%.2f$)")):
            sft_d, d29 = fams["sft"]["contrasts"], fams["dpo29"]["contrasts"]
            if min(sft_d) <= max(d29):
                bad.append("EQUAL-DOSE ranges OVERLAP (%s): SFT@29 min %.4f <= DPO@29 max %.4f"
                           % (label, min(sft_d), max(d29)))
            r = (sum(sft_d) / len(sft_d)) / (sum(d29) / len(d29))
            if (ctx % r) not in tex:
                bad.append("%s equal-dose mean-displacement ratio %.2f not quoted as %r"
                           % (label, r, ctx % r))
        # (3) flatness on one architecture: rung means may not differ by more than the
        # largest within-rung spread; and no rung may reach SFT's displacement.
        rungs = [F[k]["contrasts"] for k in ("dpo12", "dpo29", "dpo60")]
        means = [sum(x) / len(x) for x in rungs]
        spread = max(max(x) - min(x) for x in rungs)
        if max(means) - min(means) > spread:
            bad.append("DPO displacement is NOT flat across the ladder: rung means %s "
                       "vary by %.4f > within-rung spread %.4f"
                       % (["%.4f" % m for m in means], max(means) - min(means), spread))
        if max(max(x) for x in rungs) >= min(F["sft"]["contrasts"]):
            bad.append("a DPO rung reached SFT's displacement range -- the 'never "
                       "approaches SFT' claim is false")
        # (4) matched-construction null: same-architecture and archived ranges as printed,
        # inside the fixed-order 29-step range, disjoint from SFT.
        for fams, label, want in ((F, "same-architecture", "land at $%.4f$--$%.4f$ on one architecture"),
                                  (A, "archived", "(Archived build: $%.4f$--$%.4f$ inside")):
            mc, d29, sft_d = (fams["dpomc29"]["contrasts"], fams["dpo29"]["contrasts"],
                              fams["sft"]["contrasts"])
            if (want % (min(mc), max(mc))) not in tex:
                bad.append("%s MC range not quoted as %r" % (label, want % (min(mc), max(mc))))
            if not (min(d29) <= min(mc) and max(mc) <= max(d29)):
                bad.append("%s MC range is NOT inside the fixed-order 29-step range" % label)
            if min(sft_d) <= max(mc):
                bad.append("%s MC range overlaps SFT -- branch MC1 text is false" % label)
            if fams[ "dpomc29"]["k_out"] != 0:
                bad.append("%s MC null no longer 0/4 outside" % label)
        # (5) Llama second backbone (R1): archived 0.0434-0.0444 / 1.7x (App. E), and the
        # same-architecture rebuild (branch LL1) 0.0442-0.0452 / 1.72x; band unchanged.
        for fams, label, want in ((A, "archived", "displace $%.4f$--$%.4f$, every one beyond"),
                                  (F, "same-architecture", "the contrasts are $%.4f$--$%.4f$, all four stay outside")):
            L = fams["llama"]
            if L["k_out"] != 4:
                bad.append("%s llama: %d/4 outside -- replication text false" % (label, L["k_out"]))
            if (want % (min(L["contrasts"]), max(L["contrasts"]))) not in tex:
                bad.append("%s llama contrasts not quoted as %r"
                           % (label, want % (min(L["contrasts"]), max(L["contrasts"]))))
            if (round(L["band_min"], 4), round(L["band_max"], 4)) != (0.0222, 0.0257):
                bad.append("%s llama seed band [%.4f, %.4f] != paper's [0.0222, 0.0257]"
                           % (label, L["band_min"], L["band_max"]))
        r_same, r_arch = F["llama"]["r_min_over_max"], A["llama"]["r_min_over_max"]
        ctx = "the smallest is $%.2f\\times$ the band maximum ($%.2f\\times$ on the archived graphs" % (r_same, r_arch)
        if ctx not in tex:
            bad.append("llama LL1 margin sentence: expected %r" % ctx)
        if round(r_arch, 1) != 1.7 or "smallest contrast is $1.7\\times$ the band's maximum" not in tex:
            bad.append("archived llama margin %.2f / its 1.7x sentence" % r_arch)
    except (OSError, KeyError, ZeroDivisionError, ValueError) as e:
        bad.append("dose-ladder checks not checkable (%s)" % e)

    return bad, mean, sd


def selftest():
    """Plant known violations of the new tex<->receipt dose-curve cross-check and
    confirm the gate FAILS on each, plus a null-edit control that must NOT fire it."""
    tex = open(TEX).read()
    anchor = r"\tau)=3,1,3$ of $4$"
    full_sentence = (r"Under one architecture $K_{\mathrm{out}}(\tau)=3,1,3$ of $4$ although the "
                      r"contrasts $c_s(\tau)$ are flat: ")
    plants = [
        ("one-digit change: K_out(29) 1 -> 2", anchor, r"\tau)=3,2,3$ of $4$"),
        ("order swap: 3,1,3 -> 1,3,3", anchor, r"\tau)=1,3,3$ of $4$"),
        ("reverted to the archived 0,2,4 (plausible-looking old value)", anchor, r"\tau)=0,2,4$ of $4$"),
        ("sentence deleted outright (sentence-neighbour loss)", full_sentence, ""),
        ("archived curve edited 0,2,4 -> 0,1,4", "different GPU architecture from most run graphs, gave $0,2,4$",
         "different GPU architecture from most run graphs, gave $0,1,4$"),
    ]
    caught = 0
    for name, o, n in plants:
        if o not in tex:
            print("PLANT NOT APPLICABLE:", name)
            continue
        bad, _, _ = check(tex.replace(o, n, 1))
        hit = any("K_out curve" in b or "band-membership curve" in b or "dose curve" in b for b in bad)
        print(("  [CAUGHT] " if hit else "  [MISSED] ") + name)
        caught += hit
    total = len(plants)
    # null-edit control: rewording an UNRELATED sentence must not trip the new rule.
    null_o = "The intuition: the band is built from only $k$ runs"
    null_n = "Intuitively, the band is built from only $k$ runs"
    if null_o in tex:
        bad_n, _, _ = check(tex.replace(null_o, null_n, 1))
        false_positive = any("K_out curve" in b or "band-membership curve" in b for b in bad_n)
        ok = not false_positive
        print(("  [OK     ] " if ok else "  [FALSE-POS] ") +
              "null-edit control (unrelated sentence reworded, dose rule must stay quiet)")
    else:
        ok = True
        print("  [SKIPPED] null-edit control (anchor text not found)")
    total += 1
    passed = caught + (1 if ok else 0)
    print("selftest caught %d/%d (incl. 1 null-edit control)" % (passed, total))
    return 0 if passed == total else 1


def main():
    if "--selftest" in sys.argv:
        return selftest()
    tex = open(TEX).read()
    bad, mean, sd = check(tex)
    for b in bad:
        print("MISMATCH  " + b)
    print("4seed matched band + 6seed 15-pair stat (mean %.4f SD %.4f) "
          "+ SFT/DPO per-algorithm verdicts + DPO D3 control + dose ladder "
          "(equal-dose disjointness, flatness, tex<->receipt dose-curve cross-check): %s"
          % (mean, sd, "PASS" if not bad else "FAIL"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
