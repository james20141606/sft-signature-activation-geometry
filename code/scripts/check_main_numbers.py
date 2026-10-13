#!/usr/bin/env python3
"""Pin result-class numbers in main.tex that the other checkers do not cover.

check_table_main.py pins Table 1/2 and part of the abstract; check_rewiring.py
pins the dose ladder, the per-algorithm nulls and the Llama replication;
check_retractions.py pins retraction wording. Between them, the 2026-10-05
mutation sweep (project_notes/next_cycle_20260927/bootloops_audit_20261003/
MUTATION_SWEEP_20261005.md, signature preset) still missed 36/60, and final
review F4 separately flagged 34 unpinned numbers, including the BFCL
n=100 / 40-20-20-20 correction (it supersedes an older n=80 / 20-20-20-20
assumption, so a revert back to it would be silent). This file closes the
highest-confidence subset of that list: every pin below is EITHER
receipt-derived (recomputed here from notes/*.json, independently of any
other script's cached output) OR a closed-form combinatorial identity, and
is matched as a whitespace-normalized substring that carries the sentence
the number sits in -- so a single-digit change, a sign flip, or an
adjacent-sentence swap/deletion breaks the match, not just an exact-number
edit.

Round 2 (2026-10-05, same day): a real mutation-sweep re-run (n=60, seed 0)
against this file's first 10 functions still missed 14/43 result-class
mutations, 11 of them real gaps (3 were sentence-neighbour restatements of
an already-pinned claim that this file simply hadn't anchored at their
specific second/third location: the Tulu-section p<=1/3 restatement, the
Proposition 1(c) general/symbolic statement vs. its only-instantiated
1/35 form, and the matched-step-300/three-pair-band sentence that carries
no number of its own). check_design_constants() below closes the rest:
fixed design constants (probe-set/top-edge sizes, backbone parameter
count, a 95% positive-control width, an on-task threshold ratio) that
recur verbatim but are design choices, not computed statistics, so they
are pinned as literal context rather than receipt-derived values.

Deliberately out of scope here (lower confidence / likely already covered,
noted 2026-10-05): the App.~F causal-sweep table (tab:functional) and the
abstract's 12x/3.5x causal-sweep figures -- check_functional_transfer.py's
territory; fig_main_seednull_v2.py itself -- a figure generator, not a
checker, outside this task's file list.

Round 3 (2026-10-05, checker-only follow-up): a confirmation mutation sweep
against the 13-function/28-selftest file above (n=60, both seeds, run on
the then-HEAD 7cf52bc) still missed 14/60 (seed 0) and 20/60 (seed 1),
union 34 mutation rows across 20 lines. All of it was either an "equation"
line this file had not reached at all (Prop.1's own symbolic box, eq:graph,
eq:band), a design constant restated in a spot no prior pin anchored
(Fig.1's four-part enumeration, the Runs./Dose./SAE-config paragraphs, the
Table-1 caption and two footnotes, the edge-subset-robustness and
SFT/DPO-29-steps sentences, the Conclusion's p<=1/(k+1) restatement), or a
number this file already computes but had not yet pinned at its second
location (check_bfcl's n_items for the Sec.~method/Sec.~results-rewiring
restatements; check_detection_threshold's receipt-derived threshold/d6 for
the "15 pairs of six GRPO-family runs" and "edge-Jaccard shift of 0.0032"
sentences, both un-pinned despite the underlying numbers already being
loaded). The main.tex:172 "Subsets of 2,000 and 5,000 edges" sentence,
left out of Round 2 on purpose, is pinned here as a literal: notes/ has no
dedicated edge-subset-robustness receipt (only the causal-sweep's own
2000/5000/10000-edge-budget files, which are check_functional_transfer.py's
territory per the note above), and the claim recurred as a real miss on
both seeds, so a literal context pin beats leaving a reproducible gap open.

    python3 code/scripts/check_main_numbers.py            # exit 1 on any mismatch
    python3 code/scripts/check_main_numbers.py --selftest # plant every target + null-edit control, must FAIL/OK
"""
import json
import math
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N = os.path.join(ROOT, "notes")
TEX = os.path.join(ROOT, "main.tex")


def load(name):
    return json.load(open(os.path.join(N, name)))


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import samearch  # noqa: E402  revision R1: same-architecture recompute from graph JSONs
_SAME = None


def same():
    global _SAME
    if _SAME is None:
        _SAME = samearch.same_arch()
    return _SAME


def norm(s):
    """Collapse all whitespace runs to single spaces so a pinned sentence still
    matches after main.tex is re-wrapped, while any digit/word change, sign
    flip, reordering, or sentence-neighbour swap still breaks the match."""
    return " ".join(s.split())


def g4(x):
    return "%.4f" % x


def pair_val(d, key, pair, field="edge_jaccard"):
    for x in d[key]:
        if x["pair"] == pair:
            return x[field]
    return None


def zscore(contrast, seed_vals):
    return (contrast - st.mean(seed_vals)) / st.stdev(seed_vals)


# ---------------------------------------------------------------------------
# BFCL behavioral calibration: Fig.~dissociation caption, App.~extended's
# "Behavioral sanity check, in full" paragraph, and Limitation L9. All three
# must track notes/bfcl_behavior_ci.json's *correction* field, which states
# this harness supersedes an older n_items=80 / "4 categories x 20" split --
# a revert to that older split must be caught, not just a wrong CI.
BFCL_CATS = ["simple_python", "multiple", "parallel", "parallel_multiple"]


def check_bfcl(flat, bad):
    try:
        d = load("bfcl_behavior_ci.json")
    except (OSError, ValueError) as e:
        bad.append("notes/bfcl_behavior_ci.json unreadable (%s)" % e)
        return
    n_items = d.get("n_items")
    npc = d.get("n_per_category") or {}
    cat_counts = [npc.get(c) for c in BFCL_CATS]
    if n_items != 100 or cat_counts != [40, 20, 20, 20]:
        bad.append("notes/bfcl_behavior_ci.json: harness shape is n_items=%r, "
                    "per-category=%r -- expected the corrected 100 / 40-20-20-20 "
                    "split (the correction field says this supersedes an older "
                    "n=80 / 20-20-20-20 assumption); every BFCL pin below assumes "
                    "the corrected split so is skipped" % (n_items, cat_counts))
        return
    base, sft = d["base"], d["sft"]
    grpo_m, grpo_t = d["grpo_merge_eval"], d["grpo_intraining"]
    d_sft, d_grpo = d["delta_sft_vs_base"], d["delta_grpo_merge_vs_base"]

    cats_str = "/".join("$%d$" % c for c in cat_counts)

    # Fig.~dissociation caption
    want_caption = (
        "On the held-out BFCL validation split ($%d$ items, %s over four "
        "categories, from the same release files as the training rows; "
        "accuracy is the unweighted category mean), GRPO improves accuracy "
        "$+%.2f$ over base (95\\%% CI $[+%.2f,+%.2f]$); SFT's $+%.2f$ is not "
        "resolved from zero by this harness ($[%s%.2f,+%.2f]$)."
        % (n_items, cats_str, d_grpo["delta"], d_grpo["ci95"][0], d_grpo["ci95"][1],
           d_sft["delta"], "-" if d_sft["ci95"][0] < 0 else "+",
           abs(d_sft["ci95"][0]), d_sft["ci95"][1]))
    if norm(want_caption) not in flat:
        bad.append("fig:dissociation caption: expected %r" % want_caption)

    # App.~extended paragraph, point estimates + CIs
    ncat = len(BFCL_CATS)
    want_var = ("each 95\\%% CI is a stratified normal approximation, "
                "$\\mathrm{Var}=\\frac1{%d}\\sum_c p_c(1-p_c)/n_c$" % (ncat * ncat))
    if norm(want_var) not in flat:
        bad.append("App.~extended BFCL Var formula: expected denominator %d "
                    "(= %d categories squared); %r" % (ncat * ncat, ncat, want_var))

    lo_grpo_combined = min(grpo_m["ci95"][0], grpo_t["ci95"][0])
    hi_grpo_combined = min(1.0, max(grpo_m["ci95"][1], grpo_t["ci95"][1]))
    want_points = (
        "The base model scores $%.3f$ ($[%.2f,%.2f]$); SFT reaches $%.4f$ "
        "$[%.2f,%.2f]$; the audited GRPO checkpoint reaches $%.4f$--$%.3f$ "
        "$[%.2f,%.2f]$"
        % (base["acc"], base["ci95"][0], base["ci95"][1],
           sft["acc"], sft["ci95"][0], sft["ci95"][1],
           grpo_m["acc"], grpo_t["acc"], lo_grpo_combined, hi_grpo_combined))
    if norm(want_points) not in flat:
        bad.append("App.~extended BFCL point estimates: expected %r" % want_points)

    want_deltas = (
        "conservative unpaired $\\Delta=+%.3f$, SE $%.3f$, 95\\%% CI "
        "$[+%.2f,+%.2f]$) while SFT's is \\emph{not} resolved by this harness "
        "($\\Delta=+%.3f$, SE $%.3f$, 95\\%% CI $[%s%.2f,+%.2f]$, which includes zero)."
        % (d_grpo["delta"], d_grpo["se"], d_grpo["ci95"][0], d_grpo["ci95"][1],
           d_sft["delta"], d_sft["se"], "-" if d_sft["ci95"][0] < 0 else "+",
           abs(d_sft["ci95"][0]), d_sft["ci95"][1]))
    if norm(want_deltas) not in flat:
        bad.append("App.~extended BFCL delta/SE/CI sentence: expected %r" % want_deltas)

    # Limitation L9
    want_l9 = (
        "(L9) The behavioral calibration is $n{=}%d$ (%s over four categories, "
        "unweighted category mean) and comes from two separate evaluation jobs; "
        "the SFT behavioral gain is not resolved from zero on it"
        % (n_items, cats_str))
    if norm(want_l9) not in flat:
        bad.append("Limitation L9: expected %r" % want_l9)

    # Round 3: two more sentences carry this same n_items, at locations the
    # 2026-10-05 confirmation sweep found unpinned (main.tex:68, main.tex:180).
    want_fig_gain = ("although it did learn its task (accuracy gain on $%d$ "
                      "held-out items, Sec.~\\ref{sec:results-rewiring})"
                      % n_items)
    if norm(want_fig_gain) not in flat:
        bad.append("Sec.~method retraction paragraph 'accuracy gain on N "
                    "held-out items' sentence: expected %r" % want_fig_gain)

    want_grpo_learned = ("GRPO did learn its task: BFCL accuracy rises over "
                          "base, with a 95\\%% interval that excludes zero "
                          "($n{=}%d$; App.~\\ref{app:extended})" % n_items)
    if norm(want_grpo_learned) not in flat:
        bad.append("Sec.~results-rewiring 'GRPO did learn its task' sentence: "
                    "expected %r" % want_grpo_learned)

    # retired-value guard: the correction note says n_items=80 (4 x 20) is
    # superseded -- it must not reappear as a live description of this harness.
    if "n{=}80" in flat or "n=80" in flat:
        bad.append("a live 'n{=}80' / 'n=80' BFCL harness description reappeared "
                    "in main.tex -- notes/bfcl_behavior_ci.json's correction field "
                    "says n_items=80 (4 categories x 20) is SUPERSEDED by n=100 "
                    "(40/20/20/20); this looks like the old assumption came back")


# ---------------------------------------------------------------------------
# The full seven-metric seed-null battery (Sec. results-battery / App.
# extended). Independently recomputed from the five RAW per-checkpoint
# files + base (not from notes/featstats_seednull.json's own consolidated
# output), per final-review F4's remediation: a generator's own cached
# number must not be trusted as its own check.
SEED_FAMILY_FILES = ["featstats_grpo_l22.json", "featstats_s1_l22.json",
                      "featstats_s3_l22.json", "featstats_s4_l22.json",
                      "featstats_s5_l22.json"]
BATTERY_METRICS = ["frac_dead", "firing_freq_gini", "firing_freq_top1pct_active",
                    "meanact_gini", "usage_entropy_norm", "act_mass_per_token",
                    "mean_act_when_active"]


def recompute_battery():
    base = load("featstats_base_l22.json")
    fam = [load(f) for f in SEED_FAMILY_FILES]
    out = {}
    for m in BATTERY_METRICS:
        vals = [f[m] for f in fam]
        band_max = max(abs(vals[i] - vals[j])
                        for i in range(len(vals)) for j in range(i + 1, len(vals)))
        out[m] = {"abs_delta_grpo": abs(vals[0] - base[m]), "band_max": band_max}
    return out


def check_battery(flat, bad):
    try:
        bat = recompute_battery()
    except (OSError, ValueError, KeyError) as e:
        bad.append("seed-null battery recompute failed (%s)" % e)
        return
    n = len(BATTERY_METRICS)
    if "for all \\emph{seven} metrics that can vary" not in flat:
        bad.append("main.tex no longer states the retained seed-null battery "
                    "count as 'seven' in its pinned sentence")
    if n != 7:
        bad.append("battery recompute now has %d retained metrics, not seven "
                    "-- the paper's count needs re-deriving" % n)

    mass, act = bat["act_mass_per_token"], bat["mean_act_when_active"]
    want_mass = ("activation mass per token: $|\\Delta_{\\text{base}\\to\\text{GRPO}}|=%.2f$ "
                 "vs.\\ seed band $[0,%.2f]$" % (mass["abs_delta_grpo"], mass["band_max"]))
    if norm(want_mass) not in flat:
        bad.append("battery act_mass_per_token example: expected %r" % want_mass)

    want_act = ("mean act per firing: $%.3f$ vs.\\ $[0,%.3f]$"
                % (act["abs_delta_grpo"], act["band_max"]))
    if norm(want_act) not in flat:
        bad.append("battery mean_act_when_active example: expected %r" % want_act)

    top1, gini = bat["firing_freq_top1pct_active"], bat["meanact_gini"]
    tied = (abs(top1["abs_delta_grpo"] - top1["band_max"]) < 1e-9 and
            abs(gini["abs_delta_grpo"] - gini["band_max"]) < 1e-9 and
            abs(top1["band_max"] - gini["band_max"]) < 1e-9)
    if not tied:
        bad.append("firing_freq_top1pct_active / meanact_gini no longer tie "
                    "delta==seed-band-max -- the 'equal to the seed band "
                    "maximum' sentence needs re-deriving (top1=%r, gini=%r)"
                    % (top1, gini))
    else:
        want_tie = "equal to the seed band maximum at $%.4f$" % top1["band_max"]
        if norm(want_tie) not in flat:
            bad.append("battery tie claim: expected %r" % want_tie)


# ---------------------------------------------------------------------------
# Precision rule: sigma_seed must exceed delta=0.0001 "on every band in
# Table 1". The floor (currently 11) is the minimum of sd/delta across the
# six Table-1 bands -- recomputed live, not hand-maintained.
PRECISION_DELTA = 0.0001


def recompute_precision_floor():
    # Revision R1: Table 1's six bands are the same-architecture ones (samearch.py).
    F = same()
    return [F[k]["sd"] for k in ("grpo", "sft", "dpo12", "dpo29", "dpo60", "llama")]


def check_precision_floor(flat, bad):
    try:
        sds = recompute_precision_floor()
    except (OSError, ValueError, KeyError) as e:
        bad.append("precision-floor recompute failed (%s)" % e)
        return
    floor_n = math.floor(min(sds) / PRECISION_DELTA)
    want = ("\\sigma_{\\mathrm{seed}}>%d\\delta$ on every band in Table~\\ref{tab:main}"
            % floor_n)
    if norm(want) not in flat:
        bad.append("precision-rule floor: expected %r (min band SD/delta = %.2f "
                    "across the 6 Table-1 bands: %s)"
                    % (want, min(sds) / PRECISION_DELTA,
                       ["%.2f" % (s / PRECISION_DELTA) for s in sds]))


# ---------------------------------------------------------------------------
# The DPO dose curve's K_out(tau)=(.,.,.) of 4 restated three more times
# (abstract, Fig.1 caption; the Fig.2/money float left the body in the 2026-10-10
# Figure 1 redraw, which plots the same arms) beyond the one
# check_rewiring.py already cross-checks against its own pinned sentence.
def recompute_outside_counts(build="same"):
    fams = same() if build == "same" else samearch.archived()
    return [fams[k]["k_out"] for k in ("dpo12", "dpo29", "dpo60")]


def check_dose_restatements(flat, bad):
    """Revision R1: the curve is restated FIVE times on one architecture (abstract, Fig.1,
    Sec.~4.2, L8, App.~E; Fig.2 left the body 2026-10-10) and twice as the archived curve (Sec.~4.2, App.~E)."""
    try:
        c0, c1, c2 = recompute_outside_counts("same")
        a0, a1, a2 = recompute_outside_counts("archived")
    except (OSError, ValueError, KeyError) as e:
        bad.append("dose-curve recompute failed (%s)" % e)
        return
    wants = [
        ("abstract", "DPO band counts ($%d/4$, $%d/4$, $%d/4$ contrasts outside at $12$, $29$, "
                     "$60$ steps) show no consistent dose ordering while its displacement stays flat"
         % (c0, c1, c2)),
        ("Fig.1 caption", "DPO's band counts ($%d/4$, $%d/4$, $%d/4$ contrasts outside at "
                          "$12/29/60$ steps) follow no dose ordering. All graphs in a row were "
                          "built on one GPU architecture (H200)." % (c0, c1, c2)),
        ("Sec.~4.2 DPO paragraph", "whether a contrast lands just inside or just outside "
                                   "varies: $%d/4$, $%d/4$, $%d/4$ at $12$, $29$, $60$ steps"
         % (c0, c1, c2)),
        ("L8 full text", "on one architecture $%d/4$, $%d/4$ and $%d/4$ contrasts lie outside "
                         "at $12$, $29$ and $60$ steps" % (c0, c1, c2)),
        ("App.~E two readings", "on one architecture $%d/4$, $%d/4$ and $%d/4$ contrasts lie "
                                "outside at doses $12$, $29$, $60$" % (c0, c1, c2)),
        ("Sec.~4.2 archived curve", "The archived graphs gave $%d/4$, $%d/4$, $%d/4$, which we "
                                    "had read as dose-dependent" % (a0, a1, a2)),
        ("App.~E archived curve", "The archived graphs gave $%d/4$, $%d/4$, $%d/4$ "
                                  "(full-precision $z$ ranges" % (a0, a1, a2)),
    ]
    for label, w in wants:
        if norm(w) not in flat:
            bad.append("%s dose-curve restatement: expected %r" % (label, w))


# ---------------------------------------------------------------------------
# Random-dictionary control (Table~randsae) and its in-text restatements.
# Both the trained-SAE z's (-15.9 SFT, -0.5 GRPO) and the random-dictionary
# z's (-16.4 / -16.9 edge Jaccard & weighted cosine, -0.0 GRPO) are
# recomputed here independently of check_rewiring.py's own 6seed-band pin.
def check_randsae(flat, bad):
    try:
        d6 = load("circuit_rewiring_qwen_6seed.json")
        dr = load("circuit_rewiring_randsae.json")
    except (OSError, ValueError) as e:
        bad.append("randsae recompute failed (%s)" % e)
        return

    trained_seed = [x["edge_jaccard"] for x in d6["seed_null"]]
    trained_c = pair_val(d6, "post_contrasts", "base->SFT")
    trained_wc_c = pair_val(d6, "post_contrasts", "base->SFT", "weighted_edge_cosine")
    trained_grpo_c = pair_val(d6, "post_contrasts", "base->GRPO")
    z_trained = zscore(trained_c, trained_seed)
    z_grpo_trained = zscore(trained_grpo_c, trained_seed)

    rand_seed = [x["edge_jaccard"] for x in dr["seed_null"]]
    rand_wc_seed = [x["weighted_edge_cosine"] for x in dr["seed_null"]]
    rand_c = pair_val(dr, "post_contrasts", "base->SFT")
    rand_wc_c = pair_val(dr, "post_contrasts", "base->SFT", "weighted_edge_cosine")
    rand_grpo_c = pair_val(dr, "post_contrasts", "base->GRPO")
    z_rand = zscore(rand_c, rand_seed)
    z_rand_wc = zscore(rand_wc_c, rand_wc_seed)
    z_grpo_rand = zscore(rand_grpo_c, rand_seed)

    want_row1 = ("base$\\to$SFT edge Jaccard ($z$) & $%.3f$ ($%.1f$) & $%.3f$ "
                 "($\\mathbf{%.1f}$)" % (trained_c, z_trained, rand_c, z_rand))
    if norm(want_row1) not in flat:
        bad.append("tab:randsae edge-Jaccard row: expected %r" % want_row1)

    want_row2 = ("base$\\to$SFT weighted edge cosine & $%.4f$ (n/a$^{\\ddagger}$) & "
                 "$%.4f$ ($\\mathbf{%.1f}$)" % (trained_wc_c, rand_wc_c, z_rand_wc))
    if norm(want_row2) not in flat:
        bad.append("tab:randsae weighted-cosine row: expected %r" % want_row2)

    want_survives = (
        "the SFT detection \\emph{fully survives} ($z=%.1f$ random vs.\\ %.1f$ "
        "trained on edge Jaccard; $z=%.1f$ on the random dictionary's weighted "
        "cosine, whose trained-SAE counterpart is too quantized to score) and "
        "the RL null holds under both ($z=%.1f$ random, %.1f$ trained)."
        % (z_rand, z_trained, z_rand_wc, z_grpo_rand, z_grpo_trained))
    # the literal source has "$z=-16.4$ random vs.\ $-15.9$ trained" -- i.e. the
    # SECOND number is not re-prefixed with "z=", so build both halves exactly:
    want_survives = (
        "the SFT detection \\emph{fully survives} ($z=%.1f$ random vs.\\ $%.1f$ "
        "trained on edge Jaccard; $z=%.1f$ on the random dictionary's weighted "
        "cosine, whose trained-SAE counterpart is too quantized to score) and "
        "the RL null holds under both ($z=%.1f$ random, $%.1f$ trained)."
        % (z_rand, z_trained, z_rand_wc, z_grpo_rand, z_grpo_trained))
    if norm(want_survives) not in flat:
        bad.append("random-dictionary-control-in-full sentence: expected %r" % want_survives)

    want_361 = "the \\emph{SFT} step displaces the graph at $z=%.1f$" % z_trained
    if norm(want_361) not in flat:
        bad.append("'SFT step displaces the graph at z=...' sentence: expected %r" % want_361)

    want_caption592 = "SFT clears the null at $z=%.1f$." % z_trained
    if norm(want_caption592) not in flat:
        bad.append("Fig. seed-null-battery caption 'SFT clears the null' sentence: "
                    "expected %r" % want_caption592)

    want_366 = ("so $z=%.1f$ (SFT) and $z=%.1f$ (GRPO) are computed against a "
                "genuinely resolved distribution" % (z_trained, z_grpo_trained))
    if norm(want_366) not in flat:
        bad.append("precision-adequacy paragraph 'computed against a genuinely "
                    "resolved distribution' sentence: expected %r" % want_366)


# ---------------------------------------------------------------------------
# Proposition 1's three instantiated numeric consequences: the SFT
# four-run / Tulu two-run bounds (p<=1/5, p<=1/3), the eight-run 70-way
# split (1/35, stated three times), and the discussion's k>=19 threshold.
def check_prop1_math(flat, bad):
    try:
        sft6 = load("circuit_rewiring_sft_seednull.json")
    except (OSError, ValueError) as e:
        bad.append("Prop.1 recompute failed (%s)" % e)
        return
    n_sft_pairs = len(sft6["seed_null"])
    k_sft = None
    for k in range(2, 12):
        if k * (k - 1) // 2 == n_sft_pairs:
            k_sft = k
            break
    if k_sft is None:
        bad.append("SFT seed-null has %d pairs, which is not C(k,2) for any "
                    "small k -- cannot derive Prop.1's 'k=4' bound" % n_sft_pairs)
    else:
        want_a = ("With $k=%d$, SFT's four seeded runs, all outside, give "
                  "$p\\le1/%d$ under this null and no smaller" % (k_sft, k_sft + 1))
        if norm(want_a) not in flat:
            bad.append("Prop.1(b) SFT instantiation: expected %r" % want_a)

    want_b = ("A single public run pair ($k=2$) gives at best $p\\le1/3$, which "
              "is why the Tulu-3 staircase is a consistency check")
    if norm(want_b) not in flat:
        bad.append("Prop.1(b) Tulu-3 instantiation: expected %r" % want_b)

    n70 = math.comb(8, 4)
    g = math.gcd(2, n70)
    num, den = 2 // g, n70 // g
    formula_val = 2 * math.factorial(4) * math.factorial(4) / math.factorial(8)
    if abs(formula_val - num / den) > 1e-9:
        bad.append("Prop.1(c) instantiation inconsistent: 2 n1! n2! / (n1+n2)! "
                    "= %.6f != %d/%d" % (formula_val, num, den))

    want_70 = ("the most extreme of the $%d$ equal splits of eight runs, so "
               "under exchangeable runs the two-sided permutation probability "
               "is $%d/%d$" % (n70, num, den))
    if norm(want_70) not in flat:
        bad.append("Sec.~results 70-split/1-35 sentence: expected %r" % want_70)

    k_min = math.ceil(1 / 0.05) - 1
    # Revision R1: the k>=19 sentence now cites the registered 20-run band (branch L1);
    # its k, count and 1/(k+1) are recomputed from the 20 same-job graphs.
    L = same()["lora20"]
    want_disc = ("one recipe's band certifies $p\\le0.05$ only with $k\\ge%d$ "
                 "reruns, all outside (part~b); a $%d$-run LoRA-SFT band, run after "
                 "submission under a registration, puts all $%d$ contrasts outside, which "
                 "reaches $p\\le1/%d$ (App.~\\ref{app:build}). Two recipes with four "
                 "runs each reach $p\\le%d/%d$ when their displacement ranges "
                 "are disjoint (part~c)" % (k_min, L["k"], L["k_out"], L["k"] + 1, num, den))
    if L["k_out"] != L["k"]:
        bad.append("20-run band no longer has every contrast outside (%d/%d)" % (L["k_out"], L["k"]))
    if norm(want_disc) not in flat:
        bad.append("Discussion k>=19 / 1-35 sentence: expected %r" % want_disc)
    want_l1 = ("Raising SFT's band from $4$ to $%d$ seeded reruns ($%d$ seed pairs), all $%d$ "
               "base-to-SFT contrasts ($%.4f$--$%.4f$) still lie outside it ($%.4f$--$%.4f$)"
               % (L["k"], len(L["pairs"]), L["k_out"], min(L["contrasts"]), max(L["contrasts"]),
                  L["band_min"], L["band_max"]))
    if norm(want_l1) not in flat:
        bad.append("App.~build L1 sentence: expected %r" % want_l1)
    Fu = same()["full"]
    want_f1 = ("($%d/%d$ contrasts outside; band $%.3f$--$%.3f$, contrasts $%.3f$--$%.3f$)"
               % (Fu["k_out"], Fu["k"], Fu["band_min"], Fu["band_max"], min(Fu["contrasts"]),
                  max(Fu["contrasts"])))
    if norm(want_f1) not in flat:
        bad.append("App.~build F1 sentence: expected %r" % want_f1)

    want_app = "For $n_1=n_2=4$ this is $2/%d=%d/%d$." % (n70, num, den)
    if norm(want_app) not in flat:
        bad.append("App.~proof Part(c) instantiation: expected %r" % want_app)

    # Part (c)'s general (symbolic) statement in the Proposition box itself --
    # the "2" leading coefficient is a closed-form fact (2*n1!*n2!/(n1+n2)!),
    # not receipt-derived, but still a fixed, context-anchored string: a
    # last_digit/sign mutation on it (mutation sweep 2026-10-05, main.tex:117)
    # was missed because only the *instantiated* 1/35 restatements were pinned.
    want_prop_c_general = ("occupy disjoint ranges with probability at most "
                            "$2\\,n_1!\\,n_2!/(n_1+n_2)!$.")
    if norm(want_prop_c_general) not in flat:
        bad.append("Proposition~1(c) general statement: expected %r"
                    % want_prop_c_general)

    # The Tulu-3 results section restates the k=2 -> p<=1/3 bound a second
    # time (distinct sentence/location from want_b above, which lives in
    # Sec.~method). Missed by the 2026-10-05 sweep at main.tex:188.
    want_tulu_restate = ("by Proposition~\\ref{prop:band} it could certify at "
                         "best $p\\le1/3$, so the staircase is a "
                         "production-scale consistency check, not a "
                         "calibrated null")
    if norm(want_tulu_restate) not in flat:
        bad.append("Sec.~results-tulu p<=1/3 restatement: expected %r"
                    % want_tulu_restate)


# ---------------------------------------------------------------------------
# The FVU/FVE row in "The SAE blind-spot bound, in full."
def check_fvu(flat, bad):
    try:
        base = load("featstats_base_l22.json")
        sft = load("featstats_sft_l22.json")
        grpo = load("featstats_grpo_l22.json")
    except (OSError, ValueError, KeyError) as e:
        bad.append("FVU recompute failed (%s)" % e)
        return
    fve_pct = 100 * (base["fve"] + sft["fve"] + grpo["fve"]) / 3
    want = ("FVU $%s$/$%s$/$%s$, FVE $\\approx%.1f\\%%$"
            % (g4(base["fvu"]), g4(sft["fvu"]), g4(grpo["fvu"]), fve_pct))
    if norm(want) not in flat:
        bad.append("SAE blind-spot FVU/FVE row: expected %r" % want)


# ---------------------------------------------------------------------------
# The audit's detection-threshold sentence: SFT's shift vs. the |z|>=2 flag
# threshold, and the ceiling on every RL contrast's shift against it.
def check_detection_threshold(flat, bad):
    try:
        d6 = load("circuit_rewiring_qwen_6seed.json")
    except (OSError, ValueError) as e:
        bad.append("detection-threshold recompute failed (%s)" % e)
        return
    seed = [x["edge_jaccard"] for x in d6["seed_null"]]
    mean, sd = st.mean(seed), st.stdev(seed)
    threshold = 2 * sd
    sft_c = pair_val(d6, "post_contrasts", "base->SFT")
    sft_ratio = abs(sft_c - mean) / threshold
    rl_contrasts = [c["edge_jaccard"] for c in d6["post_contrasts"]
                    if c["pair"] != "base->SFT"]
    rl_ratio_max = max(abs(c - mean) / threshold for c in rl_contrasts)

    want_sd = ("with null SD $%s$, any contrast shifting edge Jaccard by $\\ge%s$"
               % (g4(sd), g4(threshold)))
    if norm(want_sd) not in flat:
        bad.append("detection-threshold null-SD sentence: expected %r" % want_sd)

    want_ratio = ("SFT's shift is $%.1f\\times$ this threshold; no RL contrast "
                  "exceeds $%.1f\\times$ it." % (sft_ratio, rl_ratio_max))
    if norm(want_ratio) not in flat:
        bad.append("detection-threshold ratio sentence: expected %r "
                    "(SFT ratio %.3f, max RL ratio %.3f)"
                    % (want_ratio, sft_ratio, rl_ratio_max))

    # Round 3: the GRPO-family k/pair-count and the |z|>=2 threshold value
    # recur at main.tex:180, unpinned by the 2026-10-05 confirmation sweep
    # even though both numbers are already loaded/computed above.
    # Revision R1: the BODY sentence now reads the five-run same-architecture GRPO band;
    # the archived 15-pair numbers above stay pinned in App.~E (marked archived build).
    G = same()["grpo"]
    n_pairs_grpo = len(G["pairs"])
    k_grpo = None
    for k in range(2, 12):
        if k * (k - 1) // 2 == n_pairs_grpo:
            k_grpo = k
            break
    if k_grpo is None:
        bad.append("GRPO-family seed-null has %d pairs, which is not C(k,2) "
                    "for any small k -- cannot derive the 'k pairs of six "
                    "GRPO-family runs' sentence" % n_pairs_grpo)
    else:
        k_word = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
                  7: "seven", 8: "eight"}.get(k_grpo, str(k_grpo))
        want_pairs = ("Against the $%d$ pairs of the %s GRPO-family runs that can still "
                      "be rebuilt on one architecture" % (n_pairs_grpo, k_word))
        if norm(want_pairs) not in flat:
            bad.append("Sec.~results-rewiring GRPO-family pairs sentence: "
                        "expected %r (k=%d, pairs=%d from receipt)"
                        % (want_pairs, k_grpo, n_pairs_grpo))

    thr_s = 2 * G["sd"]
    aud = same()["sft_audited"]["contrasts"][0]
    ratio_s = (aud - G["mean"]) / thr_s
    frac_word = {8: "one-eighth", 9: "one-ninth", 10: "one-tenth", 11: "one-eleventh",
                 12: "one-twelfth"}.get(round(ratio_s), "1/%d" % round(ratio_s))
    want_band_shift = ("this band flags an edge-Jaccard shift of $%s$, about "
                        "%s of SFT's shift against it" % (g4(thr_s), frac_word))
    if norm(want_band_shift) not in flat:
        bad.append("Sec.~results-rewiring band-shift sentence: expected %r "
                    "(threshold=%.4f, ratio %.2f)" % (want_band_shift, thr_s, ratio_s))
    rl_max = max((c - G["mean"]) / thr_s for c in G["contrasts"])
    want_app = ("flags $|z|\\ge2$ shifts of $%s$; SFT's shift against it is $%.1f\\times$ "
                "that, and the most displaced RL contrast reaches $%.1f\\times$ it."
                % (g4(thr_s), ratio_s, rl_max))
    if norm(want_app) not in flat:
        bad.append("App.~build same-architecture threshold sentence: expected %r" % want_app)
    if G["k_out"] != 0 or "($%d/%d$; Table~\\ref{tab:main})" % (G["k_out"], G["k"]) not in flat:
        bad.append("Sec.~results-rewiring GRPO count: expected ($%d/%d$; Table~\\ref{tab:main})"
                   % (G["k_out"], G["k"]))


# ---------------------------------------------------------------------------
# Abstract's "over half as far apart" qualifier: the SFT seed-pair
# displacement must exceed half the base->SFT displacement even in the
# worst-case (smallest seed gap vs. largest base->SFT gap) pairing.
def check_half_as_far(flat, bad):
    try:
        d = load("circuit_rewiring_sft_seednull.json")
    except (OSError, ValueError) as e:
        bad.append("'over half as far apart' recompute failed (%s)" % e)
        return
    seed_disp = [1 - x["edge_jaccard"] for x in d["seed_null"]]
    cons_disp = [1 - c["edge_jaccard"] for c in d["post_contrasts"]
                 if c["pair"].startswith("base->SFT_s")]
    if not seed_disp or not cons_disp:
        bad.append("'over half as far apart': missing seed/contrast displacement rows")
        return
    ratio_worst = min(seed_disp) / max(cons_disp)
    want = "two LoRA-SFT reruns sit over half as far apart as either sits from the base"
    if ratio_worst <= 0.5:
        bad.append("'over half as far apart' no longer receipt-supported: "
                   "worst-case ratio %.3f <= 0.5 (min seed-pair displacement "
                   "%.4f, max base->SFT displacement %.4f)"
                   % (ratio_worst, min(seed_disp), max(cons_disp)))
    F = same()["sft"]
    if min(F["pairs"]) / max(F["contrasts"] + same()["sft_audited"]["contrasts"]) <= 0.5:
        bad.append("'over half as far apart' fails on the same-architecture graphs")
    elif norm(want) not in flat:
        bad.append("abstract 'over half as far apart' qualifier missing/altered: "
                   "expected %r (worst-case ratio %.3f supports it)" % (want, ratio_worst))


# ---------------------------------------------------------------------------
# The matched-step-300 "0.0006 below the three-pair band" gap, restated
# three times (results paragraph, precision-adequacy paragraph, Fig. caption).
def check_three_pair_gap(flat, bad):
    try:
        d4 = load("circuit_rewiring_4seed_step300.json")
    except (OSError, ValueError) as e:
        bad.append("three-pair-gap recompute failed (%s)" % e)
        return
    rows = {x["pair"]: x["edge_jaccard"] for x in d4["post_contrasts"] + d4["seed_null"]}
    try:
        band_vals = [rows["GRPOs1<->GRPOs4"], rows["GRPOs1<->GRPOs5"],
                     rows["GRPOs4<->GRPOs5"]]
        s1 = rows["base->GRPOs1"]
    except KeyError as e:
        bad.append("three-pair-gap recompute: receipt missing pair %s" % e)
        return
    gap = min(band_vals) - s1
    g = g4(gap)
    for label, ctx in (
        ("results paragraph", "s1 falls $%s$ below the three-pair matched band" % g),
        ("precision-adequacy paragraph",
         "still sits $%s$ below the three-pair same-step band" % g),
        ("Fig. caption", "and $%s$ below the three-pair same-step band." % g),
    ):
        if norm(ctx) not in flat:
            bad.append("three-pair gap restatement (%s) missing/altered: "
                       "expected %r (gap=%.4f)" % (label, ctx, gap))

    # Fourth restatement, in the main results paragraph ("Light GRPO is not
    # resolved from its run noise."): no numeric gap is quoted here (that is
    # in the three restatements above), just the step-300/band-membership
    # claim itself -- a literal context pin. A swap_sent mutation on the
    # "$300$" in this sentence (main.tex:180) was missed by the 2026-10-05
    # sweep because none of the three restatements above anchor this exact
    # sentence.
    # Revision R1: the body no longer reads s1 as outside (S2); it says the archived
    # exceedance lay within the build difference. Pin that disclosure in the body.
    want_s1 = ("The archived six-run band put one contrast (s1) just outside; that "
               "exceedance lay within the build difference")
    if norm(want_s1) not in flat:
        bad.append("results-paragraph archived-s1 disclosure missing/altered: expected %r"
                   % want_s1)


# ---------------------------------------------------------------------------
# Sec.~results-h1's top-1% concentration share and the base-control figure's
# max-|Delta Gini| shading, both receipt-derived from the five-Qwen-row
# base-diff summary (base.t1 -> 42.7%; max |dGini_vs_base| over the five
# non-base rows -> 0.005). Missed by the 2026-10-05 sweep at main.tex:159,164.
def check_concentration(flat, bad):
    try:
        d = load("basediff_qwen_summary.json")
    except (OSError, ValueError) as e:
        bad.append("concentration recompute failed (%s)" % e)
        return
    rows = d.get("rows", {})
    try:
        t1_pct = rows["base"]["t1"] * 100
    except KeyError as e:
        bad.append("concentration recompute: receipt missing %s" % e)
        return
    want_t1 = ("the top $1\\%%$ of active destination features receive "
              "$%.1f\\%%$ of inbound weight" % t1_pct)
    if norm(want_t1) not in flat:
        bad.append("Sec.~results-h1 top-1%% concentration sentence: "
                   "expected %r" % want_t1)

    non_base = [v for k, v in rows.items() if k != "base"]
    try:
        max_dgini = max(abs(v["dGini_vs_base"]) for v in non_base)
    except KeyError as e:
        bad.append("concentration recompute: missing dGini_vs_base field %s" % e)
        return
    want_fig = ("the shading is $\\pm$ the largest $|\\Delta\\mathrm{Gini}|$ "
               "among the five Qwen rows ($%.3f$), not a statistical null"
               % max_dgini)
    if norm(want_fig) not in flat:
        bad.append("base-control fig. caption max-|dGini| sentence: "
                   "expected %r (receipt max=%.4f)" % (want_fig, max_dgini))


# ---------------------------------------------------------------------------
# Fixed design constants and short restatements that recur verbatim but are
# not computed from a receipt (probe-set sizes, the backbone's parameter
# count, a positive-control width, a base-dominance claim). Each is a
# context-anchored literal pin: any single-digit mutation inside it breaks
# the match even though the surrounding prose is unaffected. Added
# 2026-10-05 after the mutation sweep missed main.tex:53,85,96,168,178,209,
# 211,215 (all "result"/"structural" class, none covered by the receipt-
# derived checks above because these numbers are design choices, not
# computed statistics).
def check_design_constants(flat, bad):
    pins = [
        ("intro: five-SFT-run edge-Jaccard shift (same architecture; range "
         "recomputed below in check_intro_five_runs)",
         "across five SFT runs (the audited one and four reseeded reruns), "
         "its $10{,}000$ strongest edges shift by an edge-Jaccard distance "
         "of $0.057$--$0.061$"),
        ("related work: light-GRPO backbone size",
         "The seed-calibrated audit supplies that reference for one regime, "
         "light GRPO on a $1.7$B model."),
        ("method: probe-set / top-edge / per-state product-count sizes",
         "Here $E(M)$ keeps the $N=10{,}000$ largest weights, and $T_x$ "
         "holds the $2{,}000$ largest products at state $x$ (of at most "
         "$2{,}500$ non-zero ones)"),
        ("battery paragraph: GRPO base-dominance under-1% claim",
         "light GRPO is base-dominated (relative changes under $1\\%$, on "
         "Pile and on GSM8K~\\citep{cobbe2021gsm8k} probes)"),
        ("DPO dose paragraph: 12-step positive-control width",
         "At the registered $12$ steps DPO passes its positive control (its "
         "displacement is several times the SFT band's $95\\%$ width"),
        ("Discussion: GRPO non-detection backbone size (first restatement)",
         "Our nulls do not resolve a footprint of light GRPO at $1.7$B from "
         "its run-to-run variation"),
        ("Discussion: GRPO non-detection backbone size (second restatement)",
         "the GRPO non-detection covers light, accuracy-only GRPO at $1.7$B "
         "read on general text"),
        ("Limitations L5: on-task flag threshold vs. pile effect",
         "its $2\\sigma$ flag threshold is ${\\sim}3\\times$ SFT's pile "
         "effect (${\\sim}4\\times$ in relative terms)"),
        # Round 3 (2026-10-05 confirmation sweep, 14/60 + 20/60 missed):
        # these are all fixed design constants or formula bodies that recur
        # verbatim, none of them computed from a notes/*.json receipt.
        # 2026-10-10 Figure 1 redraw: the schematic (and its (1)-(4) caption) is replaced by
        # the receipt-drawn strip plot; pin the file, its generator and the receipts named.
        ("Fig.1: receipt-drawn strip plot included, generator and receipts named",
         "\\includegraphics[width=\\textwidth]{figures/fig1_r1_20261010/fig1.pdf}"),
        ("Fig.1 caption: generator and receipt directories",
         "Drawn by \\texttt{code/analysis/fig1\\_r1\\_strip.py} from "
         "\\texttt{results/\\{sftband\\_20260926,graph\\_build\\_audit\\_20261005\\}/}."),
        ("method: Runs paragraph recipe roster (SFT/DPO/GRPO-family "
         "counts and step counts)",
         "four LoRA-SFT runs (rank-$16$, $\\alpha{=}32$ on the attention "
         "projections, $0.37\\%$ of parameters, $29$ steps), four "
         "LoRA-DPO runs~\\citep{rafailov2023dpo} at $12$ steps plus four "
         "retrained runs read at $29$ and $60$ steps, and six "
         "full-parameter GRPO-family runs~\\citep{shao2024deepseekmath} "
         "(an accuracy-only baseline, four further seeds, one "
         "shaped-reward variant, $100$--$300$ steps, so that band mixes "
         "seed, step and reward variance; L7)."),
        ("method: co-activation graph SAE config / layer indices",
         "(\\texttt{qwen-scope}~\\citep{deng2026qwenscope}, TopK with "
         "$k{=}50$, $d{=}32{,}768$, layers $14\\to22$). With codes "
         "$z^{\\mathrm{src}}(x;M)=f^{\\mathrm{src}}(h_{14}(x;M))"
         "\\in\\mathbb{R}^{d}_{\\ge0}$"),
        ("eq:graph: weighted co-activation / top-N edge formula",
         "w_{ij}(M)=\\frac{1}{|X|}\\sum_{x\\in X} "
         "z^{\\mathrm{src}}_i(x;M)\\,z^{\\mathrm{dst}}_j(x;M)\\,"
         "\\mathbf{1}[(i,j)\\in T_x],\\quad E(M)=\\mathrm{top}_N\\,w(M)."),
        ("eq:band: mu_seed / sigma^2_seed / z_s formula block",
         "\\mu_{\\mathrm{seed}}=\\frac1m\\sum_{s<t}d_{st},\\qquad "
         "\\sigma^2_{\\mathrm{seed}}=\\frac{1}{m-1}\\sum_{s<t}\\big(d_{st}-"
         "\\mu_{\\mathrm{seed}}\\big)^2,\\qquad "
         "z_s=\\frac{c_s-\\mu_{\\mathrm{seed}}}{\\sigma_{\\mathrm{seed}}},"),
        ("method: Dose paragraph step-count set",
         "For DPO every quantity in (\\ref{eq:band}) is indexed by the "
         "step count $\\tau\\in\\{12,29,60\\}$."),
        ("Proposition 1 box: statement intro / exchangeability setup",
         "Let $R_0,\\dots,R_k$ $(k\\ge2)$ be random checkpoints whose "
         "joint law is invariant under permutations of the indices, let "
         "$S$ be any symmetric similarity, and let "
         "$\\mathcal P=\\{\\{s,t\\}:1\\le s<t\\le k\\}$. Whatever the "
         "dependence among the $m=k(k-1)/2$ pairs,"),
        ("Proposition 1 box: part (a) statement",
         "\\textup{(a)} $\\Pr\\big[S(R_0,R_1)<\\min_{\\{s,t\\}\\in"
         "\\mathcal P}S(R_s,R_t)\\big]\\le 1/k$; and"),
        ("Proposition 1 box: part (b) statement",
         "\\textup{(b)} $\\Pr\\big[\\max_{1\\le s\\le k}S(R_0,R_s)<"
         "\\min_{\\{s,t\\}\\in\\mathcal P}S(R_s,R_t)\\big]\\le "
         "1/(k+1)$."),
        ("Table 1 caption: k/k(k-1)/2 displacement-definition sentence",
         "Each arm has $k$ independently seeded runs and $k(k-1)/2$ "
         "seed pairs; entries are displacements $D=1-J$ "
         "(Eq.~\\ref{eq:band})."),
        ("Table 1 footnote: matched-dose 29-step gap sentence (numbers "
         "recomputed in check_table_main.py)",
         "Matched dose, no band: SFT $0.0571$--$0.0615$ vs.\\ DPO "
         "$0.0317$--$0.0356$ at $29$ steps; gap $0.0216=5.0\\times$ and "
         "$5.6\\times$ the arms' spreads ($0.0043$, $0.0039$)."),
        ("Table 1 footnote c: GRPO-family baseline/step-range "
         "restatement",
         "$^{c}$four seeds and one shaped reward, $100$--$300$ steps (L7); "
         "the baseline checkpoint was deleted and cannot be rebuilt."),
        ("results-rewiring: edge-subset robustness sentence (no "
         "dedicated receipt in notes/; literal pin, see module "
         "docstring Round 3)",
         "Subsets of $2{,}000$ and $5{,}000$ edges keep SFT detected."),
        ("results-rewiring: SFT/DPO ordering 29-step gap sentence",
         "At SFT's $29$ steps the gap between the four SFT and four "
         "DPO runs is several times either arm's spread "
         "(Table~\\ref{tab:main})"),
        ("results-rewiring: step-100-to-300 within-run displacement "
         "sentence",
         "within one run, displacement does not grow from step $100$ "
         "to $300$ (App.~\\ref{app:extended})"),
        ("Conclusion: run-exchangeability k-runs certify restatement",
         "under run exchangeability, $k$ runs certify no better than "
         "$p\\le1/(k+1)$."),
        ("Sec.~method intuition paragraph: p<=1/k and p<=1/(k+1) "
         "general statement",
         "the strongest guarantee that band exceedance gives under "
         "exchangeability alone is $p\\le1/k$ for one contrast and "
         "$p\\le1/(k+1)$ when all $k$ are outside, however large $|z|$ "
         "is"),
        ("Sec.~method intuition paragraph: k runs / k+1 odd-one-out "
         "framing",
         "The intuition: the band is built from only $k$ runs, and any "
         "of the $k+1$ runs could have been the odd one out."),
        ("Sec.~method intuition paragraph: rank-quantile 1/(m+1) "
         "aside",
         "And a rank quantile $1/(m+1)$, which treats the contrast as "
         "exchangeable with the $m$ seed pairs, does not follow from "
         "run exchangeability"),
    ]
    for label, ctx in pins:
        if norm(ctx) not in flat:
            bad.append("design-constant pin (%s) missing/altered: expected %r"
                       % (label, ctx))


FS_DIR = os.path.join(ROOT, "results/graph_build_audit_20261005/run/featstats")
BATTERY_WORDS = {"frac_dead": "dead fraction", "firing_freq_gini": "firing-frequency Gini",
                 "meanact_gini": "mean-activation Gini", "usage_entropy_norm": "normalized usage entropy",
                 "firing_freq_top1pct_active": "firing-frequency top-$1\\%$",
                 "act_mass_per_token": "activation mass per token",
                 "mean_act_when_active": "mean activation when active"}


def check_battery_same_arch(flat, bad):
    """Registered branch B3: the battery's same-architecture (H200, JID 19223777) counts,
    recomputed from the five raw featstats files -- not from battery_same_env_descriptive.json."""
    try:
        base = json.load(open(os.path.join(FS_DIR, "featstats_base_full.json")))
        runs = [json.load(open(os.path.join(FS_DIR, "featstats_s%d_full.json" % i))) for i in (1, 3, 4, 5)]
    except (OSError, ValueError) as e:
        bad.append("same-architecture battery files unreadable (%s)" % e)
        return
    counts = {}
    for m in BATTERY_METRICS:
        v = [r[m] for r in runs]
        band = max(abs(a - b) for i, a in enumerate(v) for b in v[i + 1:])
        counts[m] = sum(abs(x - base[m]) <= band + 1e-12 for x in v)
    groups = {}
    for m, c in counts.items():
        groups.setdefault(c, []).append(m)
    def words(ms):
        w = [BATTERY_WORDS[m] for m in ms]
        return w[0] if len(w) == 1 else ", ".join(w[:-1]) + " and " + w[-1]
    order = [m for m in ("frac_dead", "firing_freq_gini", "meanact_gini", "usage_entropy_norm",
                         "firing_freq_top1pct_active", "act_mass_per_token", "mean_act_when_active")]
    parts = []
    for c in sorted(groups, reverse=True):
        ms = [m for m in order if m in groups[c]]
        parts.append("$%d/4$ %s %s" % (c, "runs on" if not parts else "on", words(ms)))
    want = ("the base$\\to$s$_i$ delta lies within the four-run max-pairwise band for "
            + ", ".join(parts[:-1]) + ", and " + parts[-1])
    if norm(want) not in flat:
        bad.append("same-architecture battery counts (B3): expected %r" % want)


def check_fig_seednull(flat, bad):
    """Fig.~seednull panel (a) (revision R1): same-architecture GRPO band and audited SFT,
    recomputed from graph JSONs; the caption must carry these numbers and the PDF must be
    the R1 redraw, not the archived fig04_redrawn.pdf that printed 's1: marginally below'."""
    F = same()
    G = F["grpo"]
    null = [1 - x for x in G["pairs"]]
    sft_j = 1 - F["sft_audited"]["contrasts"][0]
    lo, hi = min(null), max(null)
    n_below = sum((1 - c) < lo for c in G["contrasts"])
    z = (sft_j - st.mean(null)) / st.stdev(null)
    want = ("on one architecture (H200), the $%d$ seed pairs of the five rebuildable GRPO-family "
            "runs, %d of the five base-to-RL contrasts below the band, and base-to-SFT at $z=%.1f$, "
            "$%.1f\\times$ the null range beyond it" % (len(null), n_below, z, (lo - sft_j) / (hi - lo)))
    if norm(want) not in flat:
        bad.append("Fig.~seednull caption (a): expected %r" % want)
    if "figures/integrated_20260923/fig04_redrawn.pdf" in flat or \
            "figures/r1_20261007/fig06_seednull_samearch.pdf" not in flat:
        bad.append("Fig.~seednull must include the R1 same-architecture redraw, not the archived fig04")


def check_intro_five_runs(flat, bad):
    F = same()
    five = F["sft"]["contrasts"] + F["sft_audited"]["contrasts"]
    want = ("strongest edges shift by an edge-Jaccard distance of $%.3f$--$%.3f$"
            % (min(five), max(five)))
    if norm(want) not in flat:
        bad.append("intro five-SFT-run range (same architecture): expected %r" % want)
    if not all(c > F["sft"]["band_max"] for c in five):
        bad.append("not all five SFT contrasts outside the SFT band on one architecture")


CHECKS = [check_fig_seednull, check_battery_same_arch, check_intro_five_runs, check_bfcl, check_battery, check_precision_floor, check_dose_restatements,
          check_randsae, check_prop1_math, check_fvu, check_detection_threshold,
          check_half_as_far, check_three_pair_gap, check_concentration,
          check_design_constants]


def check(tex):
    flat = norm(tex)
    bad = []
    for fn in CHECKS:
        fn(flat, bad)
    return bad


def main():
    tex = open(TEX).read()
    if "--selftest" in sys.argv:
        return selftest(tex)
    bad = check(tex)
    for b in bad:
        print("MISMATCH  " + b)
    print("BFCL correction / seed-null battery / precision floor / dose-curve "
          "restatements / tab:randsae / Prop.1 instantiations / FVU row / "
          "detection threshold / abstract qualifier / three-pair gap / "
          "design constants: %s"
          % ("PASS" if not bad else "FAIL"))
    return 1 if bad else 0


def selftest(tex):
    """For every pin group above, plant the mutation it was written to catch
    (a single-digit edit, a sign/order flip, or an outright deletion of the
    pinned sentence) and require check() to flag it; then run one unrelated
    rewording as a null-edit control that must NOT fire any of these rules."""
    plants = [
        ("BFCL Fig.4: GRPO delta digit 0.36 -> 0.46",
         "GRPO improves accuracy $+0.36$ over base",
         "GRPO improves accuracy $+0.46$ over base"),
        ("BFCL App.E: base accuracy 0.531 -> 0.631",
         "The base model scores $0.531$ ($[0.43,0.63]$)",
         "The base model scores $0.631$ ($[0.43,0.63]$)"),
        ("BFCL L9: n=100 reverted to the superseded n=80",
         "(L9) The behavioral calibration is $n{=}100$",
         "(L9) The behavioral calibration is $n{=}80$"),
        ("battery: 'seven' -> 'six'",
         "for all \\emph{seven} metrics that can vary",
         "for all \\emph{six} metrics that can vary"),
        ("battery: act-mass example digit 0.31 -> 0.41",
         "activation mass per token: $|\\Delta_{\\text{base}\\to\\text{GRPO}}|=0.31$",
         "activation mass per token: $|\\Delta_{\\text{base}\\to\\text{GRPO}}|=0.41$"),
        ("precision floor: 10 -> 11 (the archived floor)",
         "\\sigma_{\\mathrm{seed}}>10\\delta$ on every band",
         "\\sigma_{\\mathrm{seed}}>11\\delta$ on every band"),
        ("abstract dose curve: 1/4 -> 2/4",
         "DPO band counts ($3/4$, $1/4$,\n$3/4$ contrasts outside",
         "DPO band counts ($3/4$, $2/4$,\n$3/4$ contrasts outside"),
        ("Fig.1 caption dose curve: order flip 3,1,3 -> 1,3,3",
         "band counts ($3/4$, $1/4$, $3/4$ contrasts outside at $12/29/60$",
         "band counts ($1/4$, $3/4$, $3/4$ contrasts outside at $12/29/60$"),
        ("Fig.1 caption: one-architecture qualifier dropped",
         "All graphs in a row were built on one GPU architecture (H200).",
         "All graphs in a row were built on one GPU family."),
        ("Sec.~4.2 DPO counts: 3/4 -> 4/4 at 60",
         "varies: $3/4$, $1/4$, $3/4$ at $12$, $29$, $60$ steps",
         "varies: $3/4$, $1/4$, $4/4$ at $12$, $29$, $60$ steps"),
        ("L8 full: counts deleted",
         "on one architecture $3/4$, $1/4$ and $3/4$ contrasts lie outside at $12$, $29$ and $60$ steps",
         "on one architecture contrasts lie outside at $12$, $29$ and $60$ steps"),
        ("Sec.~4.2 archived curve: 2/4 -> 1/4",
         "The archived graphs gave $0/4$, $2/4$, $4/4$, which we had read",
         "The archived graphs gave $0/4$, $1/4$, $4/4$, which we had read"),
        ("App.~E archived curve: 4/4 -> 3/4",
         "The archived graphs gave $0/4$, $2/4$, $4/4$ (full-precision",
         "The archived graphs gave $0/4$, $2/4$, $3/4$ (full-precision"),
        ("intro five-run range reverted to archived 0.059--0.063",
         "of $0.057$--$0.061$", "of $0.059$--$0.063$"),
        ("Discussion 20-run band: 1/21 -> 1/20",
         "which reaches $p\\le1/21$ (App.", "which reaches $p\\le1/20$ (App."),
        ("App.~build L1: band max 0.0390 -> 0.0391",
         "still lie outside it ($0.0298$--$0.0390$)", "still lie outside it ($0.0298$--$0.0391$)"),
        ("App.~build F1: 4/4 -> 3/4",
         "($4/4$ contrasts outside; band $0.095$", "($3/4$ contrasts outside; band $0.095$"),
        ("App.~build threshold: 9.9x -> 7.9x (archived)",
         "SFT's shift against it is $9.9\\times$", "SFT's shift against it is $7.9\\times$"),
        ("Sec.~4.2 GRPO count 0/5 -> 1/5",
         "($0/5$; Table~\\ref{tab:main})", "($1/5$; Table~\\ref{tab:main})"),
        ("tab:randsae row: -16.4 -> -14.4",
         "& $0.840$ ($\\mathbf{-16.4}$)",
         "& $0.840$ ($\\mathbf{-14.4}$)"),
        ("'fully survives' sentence: -16.9 -> -15.9 (sentence-neighbour value)",
         "$z=-16.9$ on the random dictionary's weighted",
         "$z=-15.9$ on the random dictionary's weighted"),
        ("Prop.1 SFT bound: 1/5 -> 1/4",
         "all outside, give $p\\le1/5$ under this null",
         "all outside, give $p\\le1/4$ under this null"),
        ("Prop.1 Tulu bound: 1/3 -> 1/2",
         "($k=2$) gives at best $p\\le1/3$",
         "($k=2$) gives at best $p\\le1/2$"),
        ("70-split sentence: 1/35 -> 1/36 (sign/value swap elsewhere unaffected)",
         "two-sided permutation probability is $1/35$ (Proposition",
         "two-sided permutation probability is $1/36$ (Proposition"),
        ("discussion k>=19 -> k>=18",
         "band certifies $p\\le0.05$ only with $k\\ge19$ reruns",
         "band certifies $p\\le0.05$ only with $k\\ge18$ reruns"),
        ("App. proof Part(c): 2/70 -> 2/72",
         "For $n_1=n_2=4$ this is $2/70=1/35$.",
         "For $n_1=n_2=4$ this is $2/72=1/35$."),
        ("FVU row: SFT 0.0588 -> 0.0598",
         "FVU $0.0592$/$0.0588$/$0.0593$",
         "FVU $0.0592$/$0.0598$/$0.0593$"),
        ("detection threshold: 7.9x -> 8.9x",
         "SFT's shift is $7.9\\times$ this threshold",
         "SFT's shift is $8.9\\times$ this threshold"),
        ("abstract 'over half' qualifier deleted/weakened to 'a third'",
         "it does not: two\nLoRA-SFT reruns sit over half as far apart as either sits from the base.",
         "it does not: two\nLoRA-SFT reruns sit a third as far apart as either sits from the base."),
        ("three-pair gap: 0.0006 -> 0.0016 (results paragraph)",
         "s1 falls $0.0006$ below the three-pair matched band",
         "s1 falls $0.0016$ below the three-pair matched band"),
        ("Fig.~seednull caption: z -19.8 -> -15.9 (archived band)",
         "and base-to-SFT at $z=-19.8$, $5.0\\times$", "and base-to-SFT at $z=-15.9$, $5.0\\times$"),
        ("Fig.~seednull reverted to the archived fig04 image",
         "figures/r1_20261007/fig06_seednull_samearch.pdf", "figures/integrated_20260923/fig04_redrawn.pdf"),
        ("B3 battery: 2/4 -> 3/4 on activation mass",
         "and $2/4$ on activation mass per token", "and $3/4$ on activation mass per token"),
        ("archived-s1 disclosure deleted from the body (S2)",
         "The archived six-run band put one contrast (s1) just outside; that "
         "exceedance lay within the build difference",
         "One contrast (s1) lies just outside the band"),
        ("Prop.1(c) general statement: leading coefficient 2 -> 1",
         "occupy disjoint ranges with probability at most "
         "$2\\,n_1!\\,n_2!/(n_1+n_2)!$.",
         "occupy disjoint ranges with probability at most "
         "$1\\,n_1!\\,n_2!/(n_1+n_2)!$."),
        ("Sec.~results-tulu p<=1/3 restatement: 1/3 -> 1/4 (missed by the "
         "2026-10-05 sweep -- only the Sec.~method instance was pinned)",
         "it could certify at best $p\\le1/3$, so the staircase is a "
         "production-scale consistency check",
         "it could certify at best $p\\le1/4$, so the staircase is a "
         "production-scale consistency check"),
        ("design constant: N=10,000 top edges (intro restatement) digit swap",
         "its $10{,}000$ strongest edges shift by an edge-Jaccard distance "
         "of $0.057$--$0.061$",
         "its $10{,}001$ strongest edges shift by an edge-Jaccard distance "
         "of $0.057$--$0.061$"),
        ("design constant: Limitations L5 on-task-threshold ratio 3x -> 2x",
         "its $2\\sigma$ flag threshold is ${\\sim}3\\times$ SFT's pile "
         "effect",
         "its $2\\sigma$ flag threshold is ${\\sim}2\\times$ SFT's pile "
         "effect"),
        ("concentration: top-1% share 42.7 -> 43.7 (Sec.~results-h1)",
         "the top $1\\%$ of active destination features receive $42.7\\%$ "
         "of inbound weight",
         "the top $1\\%$ of active destination features receive $43.7\\%$ "
         "of inbound weight"),
        ("concentration: fig. caption max-|dGini| 0.005 -> 0.006",
         "the shading is $\\pm$ the largest $|\\Delta\\mathrm{Gini}|$ among "
         "the five Qwen rows ($0.005$), not a statistical null",
         "the shading is $\\pm$ the largest $|\\Delta\\mathrm{Gini}|$ among "
         "the five Qwen rows ($0.006$), not a statistical null"),
        # Round 3 (2026-10-05 confirmation sweep): 23 new plants, one per pin
        # added above (2 in check_bfcl, 2 in check_detection_threshold, 19 in
        # check_design_constants).
        ("BFCL retraction paragraph: 100 held-out items -> 010 (digit swap)",
         "accuracy gain on $100$ held-out items",
         "accuracy gain on $010$ held-out items"),
        ("results-rewiring: 95% interval -> 59% (digit swap)",
         "with a 95\\% interval that excludes zero",
         "with a 59\\% interval that excludes zero"),
        ("detection threshold: GRPO-family pairs 10 -> 15 (archived)",
         "Against the $10$ pairs of the five GRPO-family runs",
         "Against the $15$ pairs of the five GRPO-family runs"),
        ("detection threshold: band-flag shift 0.0023 -> 0.0032 (archived)",
         "edge-Jaccard shift of $0.0023$, about one-tenth",
         "edge-Jaccard shift of $0.0032$, about one-tenth"),
        ("Fig.1 image reverted to the rejected schematic",
         "{figures/fig1_r1_20261010/fig1.pdf}",
         "{figures/final_20260923/fig1.pdf}"),
        ("Fig.1 caption: receipt directory renamed",
         "graph\\_build\\_audit\\_20261005\\}/}.}",
         "graph\\_build\\_audit\\_20261004\\}/}.}"),
        ("Runs paragraph: LoRA alpha fraction 0.37% -> 0.38%",
         "$0.37\\%$ of parameters, $29$ steps",
         "$0.38\\%$ of parameters, $29$ steps"),
        ("Co-activation graph paragraph: layer 14 -> 15",
         "layers $14\\to22$). With codes",
         "layers $15\\to22$). With codes"),
        ("eq:graph formula: indicator mathbf{1} -> mathbf{0}",
         "z^{\\mathrm{dst}}_j(x;M)\\,\\mathbf{1}[(i,j)\\in T_x]",
         "z^{\\mathrm{dst}}_j(x;M)\\,\\mathbf{0}[(i,j)\\in T_x]"),
        ("eq:band sigma^2 denominator: m-1 -> m-2",
         "\\sigma^2_{\\mathrm{seed}}=\\frac{1}{m-1}\\sum_{s<t}",
         "\\sigma^2_{\\mathrm{seed}}=\\frac{1}{m-2}\\sum_{s<t}"),
        ("Dose paragraph: tau set 60 -> 61",
         "step count $\\tau\\in\\{12,29,60\\}$.",
         "step count $\\tau\\in\\{12,29,61\\}$."),
        ("Prop.1 box intro: k>=2 -> k>=3",
         "Let $R_0,\\dots,R_k$ $(k\\ge2)$ be random checkpoints",
         "Let $R_0,\\dots,R_k$ $(k\\ge3)$ be random checkpoints"),
        ("Prop.1 box part(a): R_1 -> R_2",
         "\\textup{(a)} $\\Pr\\big[S(R_0,R_1)<\\min",
         "\\textup{(a)} $\\Pr\\big[S(R_0,R_2)<\\min"),
        ("Prop.1 box part(b): bound 1/(k+1) -> 2/(k+1)",
         "S(R_s,R_t)\\big]\\le 1/(k+1)$.",
         "S(R_s,R_t)\\big]\\le 2/(k+1)$."),
        ("Table 1 caption: k(k-1)/2 -> k(k-1)/3",
         "independently seeded runs and $k(k-1)/2$ seed pairs; entries",
         "independently seeded runs and $k(k-1)/3$ seed pairs; entries"),
        ("Table 1 footnote: matched-dose 29 steps -> 28 steps",
         "DPO $0.0317$--$0.0356$ at $29$ steps; gap",
         "DPO $0.0317$--$0.0356$ at $28$ steps; gap"),
        ("Table 1 footnote c: GRPO step range 300 -> 301",
         "one shaped reward, $100$--$300$ steps (L7); the baseline",
         "one shaped reward, $100$--$301$ steps (L7); the baseline"),
        ("results-rewiring: edge-subset robustness 5,000 -> 5,001",
         "Subsets of $2{,}000$ and $5{,}000$ edges keep SFT detected.",
         "Subsets of $2{,}000$ and $5{,}001$ edges keep SFT detected."),
        ("SFT/DPO-ordering paragraph: 29 steps -> 28 steps",
         "At SFT's $29$ steps the gap between",
         "At SFT's $28$ steps the gap between"),
        ("light-GRPO paragraph: step 100 -> 010 (digit swap)",
         "displacement does not grow from step $100$ to $300$",
         "displacement does not grow from step $010$ to $300$"),
        ("Conclusion: run-exchangeability bound k+1 -> k+0",
         "certify no better than $p\\le1/(k+1)$.",
         "certify no better than $p\\le1/(k+0)$."),
        ("intuition paragraph: p<=1/k -> p<=2/k",
         "exchangeability alone is $p\\le1/k$ for one contrast",
         "exchangeability alone is $p\\le2/k$ for one contrast"),
        ("intuition paragraph: k+1 odd-one-out -> k+0",
         "and any of the $k+1$ runs could have been the odd one out.",
         "and any of the $k+0$ runs could have been the odd one out."),
        ("intuition paragraph: rank quantile 1/(m+1) -> 1/(m+0)",
         "And a rank quantile $1/(m+1)$, which treats",
         "And a rank quantile $1/(m+0)$, which treats"),
    ]
    caught = 0
    for name, o, n in plants:
        if o not in tex:
            print("PLANT NOT APPLICABLE:", name)
            continue
        if check(tex.replace(o, n, 1)):
            print("  [CAUGHT] " + name)
            caught += 1
        else:
            print("  [MISSED] " + name)
    total = len(plants)

    # null-edit control: reword an unrelated sentence; none of these rules may
    # fire. Moved 2026-10-05 (Round 3) off "The intuition: the band is built
    # from only $k$ runs..." -- that sentence is no longer unrelated, it is
    # now itself pinned (check_design_constants' intuition-paragraph pins),
    # so rewording it is a true positive, not a false one.
    null_o = "What the audit does not show matters as much."
    null_n = "What the audit does not show matters just as much."
    if null_o in tex:
        false_positives = check(tex.replace(null_o, null_n, 1))
        ok = not false_positives
        print(("  [OK     ] " if ok else "  [FALSE-POS] ") +
              "null-edit control (unrelated sentence reworded, these rules must stay quiet)")
    else:
        ok = True
        print("  [SKIPPED] null-edit control (anchor text not found)")
    total += 1
    passed = caught + (1 if ok else 0)
    print("selftest caught %d/%d (incl. 1 null-edit control)" % (passed, total))
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
