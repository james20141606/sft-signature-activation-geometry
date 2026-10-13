#!/usr/bin/env python3
"""Assert that retracted claims do not survive in main.tex, and that the clauses which
retract them have not been deleted.

    python3 code/scripts/check_retractions.py             # gate: exit 1 on violation
    python3 code/scripts/check_retractions.py --selftest  # prove the gate can FAIL

WHY THIS SHAPE (the v1 post-mortem, kept because it is the whole point).
Page compression in this repo uses "move-don't-delete": body paragraphs are moved
verbatim into an unabridged appendix archive. That preserves evidence, but it gives
every claim TWO homes, and when a claim is later retracted only the body copy tends to
get edited. Hand-sweeping is not enough -- v1 of this script caught a survivor I had
missed by eye.

But v1 was itself weak in two ways that a `--selftest` exposed and that re-reading the
code did not (it scored 4/6 against planted violations):

  * It had ONLY negative rules ("this retracted phrase must not appear"). Deleting a
    *correcting* clause -- e.g. dropping "we draw no dissociation" from the intro --
    could therefore never fail it. That whole class was unreachable.
  * Its exemption was a 700-character window around a long list of loose markers
    ("rather than", "not as a", "draw no"). Ordinary neighbouring prose satisfied it, so
    a reinstated false ATTRIBUTION sat inside the window and passed.

v2 fixes both: exemptions are SAME-SENTENCE and the marker list is short and explicitly
retraction-flavoured; rules are split FACTUAL vs WORDING; and POSITIVE rules require the
correcting clauses to still be present, each anchored to the region that must carry it.

RULE CLASSES
  FACTUAL  - wrong everywhere, including inside the archive. A retracted attribution, a
             wrong hyperparameter or a false statement is not made acceptable by living
             in a verbatim copy.
  WORDING  - a superseded framing. Permitted only where the SAME SENTENCE marks it as
             superseded -- in the archive as in the body.
  POSITIVE - a clause that must be PRESENT in a named region. These are what make the
             gate able to fail on deletion rather than only on reinstatement.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEX = os.path.join(ROOT, "main.tex")
PDF = os.path.join(ROOT, "main.pdf")

# SCOPES BEYOND main.tex.
# A retraction guard that reads only the .tex is blind to every claim that reaches the
# reader through a FIGURE. That is not hypothetical: after the dissociation was withdrawn
# from the prose, figures/fig_dissociation.py still titled itself "Behavior and SAE-space
# footprint dissociate" and annotated "NO internal footprint" and "3x smaller behavior
# change", so the rendered paper contradicted its own caption -- and this gate passed,
# because it was looking in the wrong file. Figure text is paper text.
FIGURE_GLOBS = ["figures/*.py", "figures/specs/*.svg", "figures/specs/*.json",
                "figures/*.tex", "figures/specs/*.tex"]


def _neutralise_escapes(src: str) -> str:
    """Turn literal two-character escapes (backslash-n/t/r) into spaces, LENGTH-PRESERVED.

    Figure sources embed line breaks inside string literals as the two characters \\ and n.
    That makes the character preceding a word the LETTER 'n', which silently defeats any
    rule with a leading word boundary: `\banti-ordered` matches "are anti-ordered" but NOT
    "footprint:\\nanti-ordered" -- and the latter is exactly how this repo's figure titles
    were written. A scope can therefore be scanned correctly by a rule that cannot fire on
    the defect shape that lives there.

    Replacing each escape with TWO spaces keeps every offset and line number intact, so
    reported locations stay correct. Fixing it here fixes every rule at once, rather than
    patching one regex and leaving the next one exposed.
    """
    return re.sub(r"\\[ntr]", "  ", src)


def figure_sources():
    """(path, text) for every committed figure specification."""
    import glob as _glob
    out = []
    for pat in FIGURE_GLOBS:
        for f in sorted(_glob.glob(os.path.join(ROOT, pat))):
            try:
                raw = open(f, encoding="utf-8").read()
                out.append((os.path.relpath(f, ROOT), _neutralise_escapes(raw)))
            except (OSError, UnicodeDecodeError):
                pass
    return out


def pdf_text():
    """Text of the COMPILED paper -- the artifact a reviewer actually reads."""
    import subprocess
    if not os.path.exists(PDF):
        return None
    r = subprocess.run(["pdftotext", PDF, "-"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None

# Short and explicitly retraction-flavoured. Deliberately EXCLUDES loose connectives
# ("rather than", "not as a", "would be false", "draw no") -- those are ordinary prose
# and were exactly what blinded v1.
RETRACTION_MARKERS = [
    "withdraw", "withdrawn", "we retract", "retracted",
    "superseded", "earlier draft", "previous draft", "prior draft", "no longer",
]

FACTUAL = "FACTUAL"
WORDING = "WORDING"

# (pattern, human name, kind)
NEGATIVE = [
    # --- FACTUAL: wrong everywhere, archive included
    (r"collapse gate of~\\citet\{yue2025", "|d|>=0.5 gate mis-attributed to yue2025/wang2025", FACTUAL),
    (r"rank-\$64\$|r\{=\}64|\\alpha\{=\}128", "wrong LoRA rank/alpha (receipt: r=16, alpha=32)", FACTUAL),
    (r"all nine metrics", "stale nine-metric count (mean L0 is TopK-pinned)", FACTUAL),
    (r"at least as null as", "false step-300 uniformity claim", FACTUAL),
    (r"SFT and DPO Leave", "pre-scoping title", FACTUAL),
    # The GRPO co-headline was removed after a re-review judged the reference band too
    # heterogeneous and the matched check too dependent to carry half the title. Guard the
    # title line itself, not prose: GRPO/RL may of course be discussed in the body.
    (r"\\title\{[^}]*(?:GRPO|Run Variation)", "GRPO back in the title (retired co-headline)", FACTUAL),
    (r"-30\.7", "z=-30.7 (precision-inadequate weighted cosine)", FACTUAL),
    # 2026-09-25 cold read: one heavy cross-family checkpoint cannot carry a dose-response
    # reading; App. F says "we do not attribute to intensity", so no caption may say it scales.
    (r"(?:scales with|tracks|tracking)\s+post-training\s+(?:\\emph\{)?intensity",
     "residual read as tracking post-training intensity (one heavy checkpoint)", FACTUAL),
    # --- WORDING: superseded framings (body only; archive needs a same-paragraph marker)
    (r"\banti-?\}?-?ordered\b", "anti-ordered dissociation", WORDING),
    (r"double dissociat\w*", "'double dissociation'", WORDING),
    (r"our dissociat\w*", "'our dissociation'", WORDING),
    (r"dissociation is the payoff", "'dissociation is the payoff'", WORDING),
    (r"Behavioral gain does not predict", "Fig.2 dissociation headline", WORDING),
    (r"three times SFT|\$3\\times\$ (?:the )?(?:smaller|SFT)", "'3x SFT' behavioral ratio", WORDING),
    (r"title's ``DPO", "DPO-in-title framing", WORDING),
    # DECOUPLING. The six papers must each stand alone: no claim may depend on, or point
    # at, a sibling submission. Anonymised references ("a companion study", "(anonymized)")
    # still couple the papers -- a reviewer cannot check them, and they break if the
    # sibling is not accepted. Keyed on the reference shape, not on named papers.
    (r"cross-paper|companion stud\w*|companion project|companion training|companion GRPO"
     r"|sibling(?: [\w-]+)? (?:paper|study|submission|analysis)|our other (?:paper|study|submission)"
     r"|a companion|the companion|concurrent submission",
     "CROSS-PAPER coupling (papers must stand alone)", FACTUAL),
    # GENERIC rule, not another exact phrase. The v2 rules were a blocklist of the exact
    # retracted wordings, so a PARAPHRASE walked straight through: the Fig. 2 caption
    # said "the dissociation is carried by the ordering", which matched none of
    # 'anti-ordered' / 'double dissociation' / 'our dissociation' / 'is the payoff', and
    # an AC found it instead of the gate. Enumerating phrasings cannot be completed.
    # This rule instead flags ANY assertion of a dissociation and exempts only the
    # sentences that deny or scope one (plus the \ref{fig:dissociation} label).
    (r"dissociat\w*", "an ASSERTED dissociation (paper claims none)", WORDING),
    # F2's RETIRED VERB. The POSITIVE rule below only ever REQUIRED the paper to say
    # "displaces" somewhere in the body; it could not notice "rewires" being reasserted
    # beside it, so the retirement was guarded in one direction only and survivors sat in
    # contribution bullets, table captions and figure axis labels while the gate passed.
    # This is the negative half. Exempted, by construction rather than by sentence:
    #   * LaTeX identifiers that keep the old name -- \ref{sec:results-rewiring},
    #     \ref{tab:rewiring} (a label is not prose, and renaming them buys nothing);
    #   * receipt/script paths -- circuit_rewiring_*.json, sft_rewiring_localize.py,
    #     tulu_rewire. In .tex the underscore kills the leading \b; pdftotext renders the
    #     same paths with SPACES ("notes/circuit rewiring sft seednull.json"), which
    #     revives it -- hence BOTH a "circuit "-lookbehind and a lookahead on the receipt
    #     stems, so a line wrap falling on either side of the word is still exempt.
    # The sentence that DOCUMENTS the retirement is exempted through REWIRING_OK below.
    (r"(?<!circuit )(?<!check )(?<!results-)(?<!tab:)\brewir\w*\b"
     r"(?! (?:sft|dpo|qwen|llama|randsae|bfcl|4seed|mc)\b)(?!\\?_)",
     "RETIRED verb 'rewiring' reasserted (branch F2 retired it for 'displacement')",
     WORDING),
    # CATEGORICAL RL-NEGATIVE assertions. The abstract disclaims equivalence and reports a
    # failure to detect; body prose that says RL "does not" / "adds nothing" / "is not
    # written by RL" asserts absence instead, which the design does not license. Keyed on
    # the assertion, not on enumerated phrasings, so paraphrases are caught too.
    # Revision R1 (2026-10-07): the DPO dose-dependence reading was withdrawn under the
    # registered V2 wording rule (PREREG_graph_build_audit_20261004 S3): on one GPU
    # architecture the counts are 3/4, 1/4, 3/4 and the archived 0/4, 2/4, 4/4 curve lay
    # within the build difference. Keyed on the assertion shapes the submitted text used.
    (r"band membership (?:is|was) dose-dependent|band membership depends on dose"
     r"|band reading depends on dose|verdict depends on the dose|dose-dependent DPO reading"
     r"|the band moves, the effect does not grow|band reading is dose-dependent",
     "DPO dose-dependence reasserted (withdrawn under registered branch V2)", WORDING),
    # S2: the GRPO s1 'marginal case outside the band' was within the build difference.
    (r"except one marginal case \(s1|s1 just outside the band",
     "GRPO s1 marginal exceedance reasserted as a live reading (S2)", WORDING),
    (r"not RL\b|not by RL\b|nulling RL\b|nulls RL\b|RL does not\b|RL leaves (?:none|no)\b"
     r"|adds almost nothing|contributes no\b|no RL signature|without an RL signature",
     "CATEGORICAL RL-negative assertion (design licenses only failure-to-detect)", WORDING),
]

# Sentence-level exemptions for the categorical-RL rule: language that scopes the claim to
# a failure to detect, to an instrument limit, or that is explicitly describing what the
# paper does NOT claim.
# Tight on purpose. An earlier draft of this list included "resolve", "beyond", "regime"
# and "consistency check", which are ordinary words in this paper's prose and wrongly
# cleared a live categorical assertion ("the RLVR stage itself adds almost nothing"
# followed later in the same sentence by "consistency check"). Only phrases that
# themselves scope the claim to a NON-detection belong here.
RL_NEGATIVE_OK = [
    "not distinguishable", "indistinguishable", "failure to detect", "fail to detect",
    "not detected", "cannot detect", "no equivalence", "not a demonstration",
    "not a theorem", "is not claimed", "we do not claim", "does not licen",
    "not certif", "does not certify", "smaller than the run", "below our resolution",
    "at our precision", "under this audit", "distinguishable from",
]

# Sentence-level exemptions for the generic dissociation rule: negations and scope
# language. A sentence that DENIES a dissociation is exactly what the paper should say.
DISSOCIATION_OK = [
    "no behavioral", "not read it as", "we do not", "do \\emph{not}", "draw no",
    "claims none", "cannot", "no dissociation", "is not claimed", "needs a powered",
    "fig:dissociation", "fig_dissociation", "includegraphics", "establishing a",
    "earning one", "earning a", "draw \\emph{no}", "read it as",
    "we claim no", "not a measured dissociation", "would need",
]

# Sentence-level exemptions for the retired-verb rule. Only two things may still say
# "rewiring": the passage that RECORDS the retirement (branch F2's own consequence -- the
# paper has to be able to explain what it retired and why), and the figure docstring that
# explains why the old axis was removed. Deliberately does NOT include the receipt paths;
# those are handled structurally in the pattern, so a claim sentence that happens to cite a
# receipt in the same breath is still caught.
REWIRING_OK = [
    "retire the word", "retired the word", "we retire",
    "branch f2 fired", "registered consequence was to retire",
]

def flex(pat: str) -> str:
    """Let every literal space in a NEGATIVE pattern match any run of whitespace, so a
    violation whose words are split across a source line break ("RL does\\nnot") is still
    found in the .tex scope (verification 2026-09-24, mutation M3b). Spaces inside
    fixed-width lookbehinds (?<!...) are left literal, since Python lookbehind cannot
    take a variable-width \\s+."""
    out, i = [], 0
    for m in re.finditer(r"\(\?<[!=][^)]*\)", pat):
        out.append(pat[i:m.start()].replace(" ", r"\s+"))
        out.append(m.group(0))
        i = m.end()
    out.append(pat[i:].replace(" ", r"\s+"))
    return "".join(out)


# (pattern, human name, region). The region must CONTAIN a match or the gate fails.
POSITIVE = [
    (r"no behavioral/internal dissociation|draw no behavioral",
     "intro must still disclaim the dissociation", "intro"),
    (r"secondary|pilot", "abstract must still label the GRPO non-detection secondary/pilot",
     "abstract"),
    (r"no equivalence (?:test|margin)|perform no equivalence|define no equivalence",
     "abstract must still disclaim the equivalence test", "abstract"),
    (r"dissociation", "limitations/L5 must still address the dissociation", "limitations"),
    (r"uninformative by construction|TopK-pinned|pinned by construction",
     "body must still state mean L0 is TopK-pinned", "body"),
    (r"commit to \\textbf\{edge Jaccard\}|edge Jaccard.{0,120}scored statistic",
     "body must still commit to a precision-adequate statistic", "body"),
    (r"r\{=\}16|rank-\$16\$", "body must still state the true LoRA rank (16)", "body"),
    # F2 fired (functional_transfer_k5000.json): the registration retires the verb
    # "rewires" from the abstract and title-adjacent prose. This rule keeps the REPLACEMENT
    # present; the "RETIRED verb" NEGATIVE rule above keeps the old one out. One without the
    # other is half a guard -- the paper passed this rule for weeks while "rewiring" sat in
    # a contribution bullet, two table captions and a figure axis label.
    # 2026-09-24: the body now names the object by its own definition (Sec. 3: "a
    # co-activation proxy, not a gradient or causal attribution"), so "displaces the
    # co-activation graph" is accepted; the rule still requires the verb DISPLACES.
    (r"displaces the (?:sparse )?co-activation|displaces the attribution graph",
     "the abstract/conclusion must say DISPLACES, not rewires (branch F2)", "body"),
    # Rule description updated 2026-08-04: DPO HAS a per-algorithm null since the
    # seed-null landed; what must survive is the Tulu-staircase disclosure that only
    # one public DPO checkpoint exists there (read against a run-pair null).
    (r"single DPO checkpoint|one DPO checkpoint|one public DPO policy",
     "L8 must still disclose the Tulu staircase has only one public DPO checkpoint", "body"),
    # The 'stale on purpose' equivalent: seed s1 exists at BOTH step 200 and step 300, so
    # circuit_rewiring_qwen_6seed.json (s1@200) and circuit_rewiring_4seed_step300.json
    # (s1@300) report the same pair name with different values ON PURPOSE. If the paper
    # ever stops explaining that, the two receipts read as a contradiction.
    (r"s1 was additionally re-run to step|s1 re-run to the matched step",
     "paper must keep explaining that s1 exists at two steps (two receipts, same pair name)",
     "anywhere"),  # lives in App. A (setup), not the body
    # The abstract states that the base control and the width control RETRACT two
    # readings the pipeline first produced. Without this rule the sentence could be
    # flipped into an endorsement and pass (verification 2026-09-24, mutation M7).
    # 2026-09-26 tone pass: the abstract states both corrected readings as findings (the
    # retraction itself is disclosed in Sec. 4.1 and App. H); the rule still requires BOTH.
    (r"already in the zero-RL base[^.]*dictionary-width control removes an apparent scaling law",
     "abstract must still state that concentration is a base property and the width scaling law an artifact",
     "abstract"),
    # 2026-09-25 round-3 verification: six main-text qualifiers were present but UNPINNED
    # -- deleting any of them passed every checker (verifier mutations). Each is anchored to
    # its own sentence and to the region before \appendix, so an appendix copy of the same
    # words cannot mask a deletion from the body.
    # 2026-09-26: corrected from n=80 (4 x 20 assumption) to the 100-item held-out split.
    (r"accuracy rises over base,[^.]*excludes zero\s*\(\$n\{=\}100\$",
     "GRPO accuracy sentence must keep its sample size (n=100)", "maintext"),
    (r"Table~\\ref\{tab:randsae\};\s*one random draw",
     "random-dictionary control must still say it is ONE random draw", "maintext"),
    (r"no equivalence\s+margin defined, these are non-detections",
     "GRPO paragraph must still call within-band results non-detections", "maintext"),
    (r"1/35\$[^.]*computed after the fact, not registered",
     "1/35 must still be labelled post hoc (computed after the fact, not registered)", "maintext"),
    (r"up to several times more than count-matched random features\s*\(an upper bound",
     "hub-ablation specificity must stay an UPPER BOUND ('up to', control not magnitude-matched)",
     "maintext"),
    # Revision R1: the replacement wording and the build disclosure must stay present.
    (r"show no consistent dose\s+ordering", "abstract must state the DPO counts show no consistent dose ordering", "abstract"),
    (r"built on one GPU\s+architecture", "abstract must say every band uses graphs built on one GPU architecture", "abstract"),
    (r"\\emph\{Build rule:\}[^.]*not across architectures", "Sec.~3 must keep the build rule (graphs not bit-identical across architectures)", "maintext"),
    (r"lay within the build difference.{0,80}?we withdraw the", "Sec.~4.2 must keep withdrawing the archived dose-dependent reading", "maintext"),
    (r"under run exchangeability,\s*\$k\$ runs certify no better than",
     "conclusion must keep the exchangeability condition on the 1/(k+1) ceiling", "conclusion"),
]


def regions(tex: str) -> dict:
    def idx(pat, default):
        m = re.search(re.escape(pat), tex)
        return m.start() if m else default
    bib = idx(r"\bibliography{references}", len(tex))
    arch_start = idx(r"\label{app:extended}", bib)
    arch_end = idx(r"\section{Extended related work}", bib)
    if arch_end < arch_start:
        arch_end = bib
    return {
        "abstract": tex[idx(r"\begin{abstract}", 0): idx(r"\end{abstract}", 0)],
        "intro": tex[idx(r"\section{Introduction}", 0): idx(r"\section{Related Work}", bib)],
        "limitations": tex[idx(r"\section{Limitations}", 0): idx(r"\section{Conclusion}", bib)],
        "maintext": tex[:idx(r"\appendix", bib)],
        "conclusion": tex[idx(r"\section{Conclusion}", 0): bib],
        "body": tex[:bib],
        "archive": tex[arch_start:arch_end],
        "anywhere": tex,
        "_archive_span": (arch_start, arch_end),
    }


def sentence_around(tex: str, pos: int) -> str:
    """The sentence containing `pos` -- far tighter than v1's 700-char window."""
    start = max(tex.rfind(". ", 0, pos), tex.rfind(".\n", 0, pos),
                tex.rfind("\n\n", 0, pos), 0)
    nxt = [p for p in (tex.find(". ", pos), tex.find(".\n", pos), tex.find("\n\n", pos)) if p != -1]
    end = min(nxt) if nxt else len(tex)
    # Collapse whitespace: in the PDF scope a line wrap can fall inside an exemption
    # phrase ("retire the\nword"), which made a correctly exempted sentence fail
    # (2026-09-24, after a reflow). Normalising only affects the exemption lookup.
    return " ".join(tex[start:end].split()).lower()


def scan(tex: str) -> list:
    reg = regions(tex)
    a0, a1 = reg["_archive_span"]
    bad = []

    for pat, name, kind in NEGATIVE:
        for m in re.finditer(flex(pat), tex, re.I):
            in_archive = a0 <= m.start() < a1
            # Exemption is SAME-SENTENCE everywhere, including the archive. A
            # paragraph-level exemption is not safe: a paragraph that legitimately
            # reports one retraction becomes a shelter for a *different* claim
            # reinstated beside it -- the selftest's V3 passed under exactly that
            # loophole. A superseded archive copy must carry its marker in its own
            # sentence ("Superseded: an earlier draft read this as ...; withdrawn").
            sent = sentence_around(tex, m.start())
            if name.startswith("an ASSERTED dissociation") and \
               any(k in sent for k in DISSOCIATION_OK):
                continue  # sentence denies or scopes a dissociation -- that is correct
            if name.startswith("CATEGORICAL RL-negative") and \
               any(k in sent for k in RL_NEGATIVE_OK):
                continue  # sentence scopes the claim to a failure to detect
            if name.startswith("RETIRED verb") and \
               any(k in sent for k in REWIRING_OK):
                continue  # sentence records the retirement rather than undoing it
            # Coupling is never excused by a retraction marker: "we withdraw X --- the
            # sibling steering analysis clears 0/19" still points at another paper
            # (cold read 2026-09-25 found exactly that, sheltered by "withdraw").
            if not name.startswith("CROSS-PAPER") and \
               any(k in sent for k in RETRACTION_MARKERS):
                continue  # reported as retracted, not asserted
            line = tex[: m.start()].count("\n") + 1
            snip = " ".join(tex[max(0, m.start() - 80): m.end() + 80].split())
            where = "ARCHIVE" if in_archive else "BODY"
            bad.append(f"[{kind}/{where}] {name} @ main.tex:{line}\n      ...{snip}...")

    # ---- figure specifications
    for relpath, src in figure_sources():
        for pat, name, kind in NEGATIVE:
            for m in re.finditer(flex(pat), src, re.I):
                sent = sentence_around(src, m.start())
                if name.startswith("an ASSERTED dissociation") and \
                   any(k in sent for k in DISSOCIATION_OK):
                    continue
                if name.startswith("CATEGORICAL RL-negative") and \
                   any(k in sent for k in RL_NEGATIVE_OK):
                    continue
                if name.startswith("RETIRED verb") and \
                   any(k in sent for k in REWIRING_OK):
                    continue
                if any(k in sent for k in RETRACTION_MARKERS):
                    continue
                line = src[: m.start()].count("\n") + 1
                snip = " ".join(src[max(0, m.start() - 70): m.end() + 70].split())
                bad.append(f"[{kind}/FIGURE] {name} @ {relpath}:{line}\n      ...{snip}...")

    # ---- compiled PDF: catches anything that reaches the reader by any route
    ptxt = pdf_text()
    if ptxt is None:
        bad.append("[SCOPE] main.pdf absent or unreadable -- compiled-text scope not checked")
    else:
        for pat, name, kind in NEGATIVE:
            if name in ("pre-scoping title", "GRPO back in the title (retired co-headline)"):
                continue  # \title{} is a source construct, not PDF text
            for m in re.finditer(flex(pat), ptxt, re.I):
                sent = sentence_around(ptxt, m.start())
                if name.startswith("an ASSERTED dissociation") and \
                   any(k in sent for k in DISSOCIATION_OK):
                    continue
                if name.startswith("CATEGORICAL RL-negative") and \
                   any(k in sent for k in RL_NEGATIVE_OK):
                    continue
                if name.startswith("RETIRED verb") and \
                   any(k in sent for k in REWIRING_OK):
                    continue
                if any(k in sent for k in RETRACTION_MARKERS):
                    continue
                snip = " ".join(ptxt[max(0, m.start() - 70): m.end() + 70].split())
                bad.append(f"[{kind}/PDF] {name} in compiled main.pdf\n      ...{snip}...")

    for pat, name, region in POSITIVE:
        if not re.search(pat, reg[region], re.I):
            bad.append(f"[POSITIVE/{region}] MISSING: {name}")
    return bad


def selftest(tex: str) -> int:
    """Plant known violations and confirm the gate FAILS on each.

    SCORING IS A BASELINE-AWARE FINDINGS-SET DELTA, not an exit status and not a
    non-empty findings list. Both weaker schemes stop measuring anything the moment the
    gate has a legitimate standing failure, and they do not merely inflate -- they can
    invert, reporting that the rules which caught nothing caught everything. A delta also
    catches the subtler case where a plant fires some OTHER rule and is scored as a catch
    for the rule it was meant to test.

    For each plant we record:
      caught  -- the finding set grew (delta non-empty) relative to the unplanted baseline
      on-target -- the delta contains a finding tagged with the scope/rule the plant targets

    A plant that is `caught` but not `on-target` is reported, because it means the scope or
    rule under test is NOT the thing that fired.
    """
    baseline = set(scan(tex))
    if baseline:
        print("NOTE: gate has %d standing finding(s); delta scoring subtracts them by "
              "construction, so the numbers below remain meaningful." % len(baseline))
        for b in sorted(baseline)[:3]:
            print("   standing: " + b.splitlines()[0])

    intro_clause = r"We draw no behavioral/internal dissociation from the cross-state comparison."
    arch_anchor = r"\paragraph{Behavioral sanity check, in full (provenance and intervals).}"

    def tex_plant(mut):
        def run():
            corrupted = mut(tex)
            if corrupted == tex:
                return None          # plant did not modify anything -> BROKEN
            return set(scan(corrupted))
        return run

    def figure_plant():
        import glob as _glob, shutil
        figs = sorted(_glob.glob(os.path.join(ROOT, "figures", "*.py")))
        if not figs:
            return None
        victim, backup = figs[0], figs[0] + ".selftest.bak"
        shutil.copy2(victim, backup)
        try:
            with open(victim, "a", encoding="utf-8") as fh:
                fh.write('\nax.set_title("Behavior and footprint dissociate")\n')
            return set(scan(tex))
        finally:
            shutil.move(backup, victim)

    def figure_plant_escaped():
        """The steering shape: the defect sits immediately after a literal \\n escape, so the
        preceding character is the letter 'n' and any leading \\b is dead."""
        import glob as _glob, shutil
        figs = sorted(_glob.glob(os.path.join(ROOT, "figures", "*.py")))
        if not figs:
            return None
        victim, backup = figs[0], figs[0] + ".selftest2.bak"
        shutil.copy2(victim, backup)
        try:
            with open(victim, "a", encoding="utf-8") as fh:
                fh.write('\nax.set_title("Behavior and footprint:' + chr(92) + 'nanti-ordered on internals")\n')
            return set(scan(tex))
        finally:
            shutil.move(backup, victim)

    def figure_plant_coupling():
        import glob as _glob, shutil
        figs = sorted(_glob.glob(os.path.join(ROOT, "figures", "*.py")))
        if not figs:
            return None
        victim, backup = figs[0], figs[0] + ".selftest3.bak"
        shutil.copy2(victim, backup)
        try:
            with open(victim, "a", encoding="utf-8") as fh:
                fh.write('\nax.set_title("panel from the companion study")\n')
            return set(scan(tex))
        finally:
            shutil.move(backup, victim)

    def figure_plant_rewiring():
        """Figure axis labels are where the retired verb actually survived longest."""
        import glob as _glob, shutil
        figs = sorted(_glob.glob(os.path.join(ROOT, "figures", "*.py")))
        if not figs:
            return None
        victim, backup = figs[0], figs[0] + ".selftest4.bak"
        shutil.copy2(victim, backup)
        try:
            with open(victim, "a", encoding="utf-8") as fh:
                fh.write('\nax.set_ylabel("edge rewiring (1 - Jaccard)")\n')
            return set(scan(tex))
        finally:
            shutil.move(backup, victim)

    def pdf_plant_coupling():
        real = pdf_text()
        if real is None:
            return None
        g = globals(); saved = g["pdf_text"]
        g["pdf_text"] = lambda: real + "\nFeature 35239 also passes a companion study's causal gate.\n"
        try:
            return set(scan(tex))
        finally:
            g["pdf_text"] = saved

    def pdf_plant():
        real = pdf_text()
        if real is None:
            return None
        g = globals(); saved = g["pdf_text"]
        g["pdf_text"] = lambda: real + "\nThe stages are anti-ordered on behavior versus internals.\n"
        try:
            return set(scan(tex))
        finally:
            g["pdf_text"] = saved

    # (name, expected tag in the delta, runner)
    plants = [
        ("V1 retracted WORDING reinstated in body (abstract)", "/BODY]",
         tex_plant(lambda t: t.replace(r"\begin{abstract}",
             r"\begin{abstract}" + "\nThe two stages are anti-ordered on behavior versus internals.", 1))),
        ("V2 correcting clause DELETED from intro", "POSITIVE/intro",
         tex_plant(lambda t: t.replace(intro_clause, "", 1))),
        ("V3 retracted WORDING left ONLY in the archive, unmarked", "/ARCHIVE]",
         tex_plant(lambda t: t.replace(arch_anchor,
             arch_anchor + " The calibration establishes a double dissociation between the two stages.", 1))),
        ("V4 retracted ATTRIBUTION reinstated (factual, anywhere)", "FACTUAL/",
         tex_plant(lambda t: t.replace(r"collapse gate (a $|d|=0.5$ threshold we pre-registered ourselves",
             r"collapse gate of~\citet{yue2025,wang2025} (a $|d|=0.5$ threshold we pre-registered ourselves", 1))),
        ("V5 wrong LoRA rank reinstated in body", "wrong LoRA rank",
         tex_plant(lambda t: t.replace(r"rank-$16$", r"rank-$64$", 1))),
        ("V6 z=-30.7 reinstated as a live claim", "z=-30.7",
         tex_plant(lambda t: t.replace(r"\section{Conclusion}",
             r"\section{Conclusion}" + "\nThe weighted cosine separates SFT at $z=-30.7$.", 1))),
        ("V7 TopK-exclusion statement deleted", "POSITIVE/body",
         tex_plant(lambda t: re.sub(r"uninformative by construction|TopK-pinned|pinned by construction",
                                    "measured normally", t))),
        ("V8 s1 two-step provenance explanation deleted", "POSITIVE/anywhere",
         tex_plant(lambda t: t.replace("s1 was additionally re-run to step", "s1 was trained to step", 1)
                              .replace("s1 re-run to the matched step", "s1 at the matched step", 1))),
        ("V9 dissociation reasserted by PARAPHRASE (blocklist-proof)", "ASSERTED dissociation",
         tex_plant(lambda t: t.replace(r"\section{Conclusion}",
             r"\section{Conclusion}" + "\nThe dissociation is carried by the ordering of the two stages.", 1))),
        ("V10 categorical RL-negative assertion reinstated in body", "CATEGORICAL RL-negative",
         tex_plant(lambda t: t.replace(r"\section{Conclusion}",
             r"\section{Conclusion}" + "\nThe footprint is written by the supervised stages, not RL.", 1))),
        # Anchor on the \title{ macro rather than on the current subtitle wording: keying on
        # prose meant a legitimate retitle silently broke the plant (caught as BROKEN, not
        # as a false pass -- but a plant that needs re-anchoring on every edit is fragile).
        ("V11 GRPO put back into the title (retired co-headline)", "GRPO back in the title",
         tex_plant(lambda t: t.replace(r"\title{", r"\title{Light GRPO Is Indistinguishable from Measured Run Variation --- ", 1))),
        ("V12 abstract stops labelling the GRPO result secondary/pilot", "POSITIVE/abstract",
         tex_plant(lambda t: t[:t.index(r"\end{abstract}")].replace("secondary", "primary").replace("pilot", "main")
                              + t[t.index(r"\end{abstract}"):])),
        ("V13 cross-paper coupling reinstated in body", "CROSS-PAPER",
         tex_plant(lambda t: t.replace(r"\section{Conclusion}",
             r"\section{Conclusion}" + "\nThis replicates a companion study's causal screen.", 1))),
        ("V14 coupling by the anonymised form ('our other submission')", "CROSS-PAPER",
         tex_plant(lambda t: t.replace(r"\section{Conclusion}",
             r"\section{Conclusion}" + "\nSee our other submission for the matching analysis.", 1))),
        ("V15 retired verb 'rewires' reasserted in the conclusion", "RETIRED verb",
         tex_plant(lambda t: t.replace(r"\section{Conclusion}",
             r"\section{Conclusion}" + "\nLoRA-SFT rewires the attribution graph at twenty standard deviations.", 1))),
        ("V16 retired verb reasserted in a table caption", "RETIRED verb",
         tex_plant(lambda t: t.replace(r"\textbf{[Archived build.] Co-activation displacement is within run noise",
                                       r"\textbf{[Archived build.] Co-activation rewiring is within run noise", 1))),
        # 2026-09-25: deletion plants for the six main-text qualifier pins (one each).
        ("V25 abstract drops the width-artifact finding", "POSITIVE/abstract",
         tex_plant(lambda t: t.replace("and a\ndictionary-width control removes an apparent scaling law in it.", "", 1))),
        ("V26 abstract drops the base-property finding", "POSITIVE/abstract",
         tex_plant(lambda t: t.replace("is already in the zero-RL base, and a", "is present, and a", 1))),
        ("V17 n=100 dropped from the GRPO accuracy sentence", "POSITIVE/maintext",
         tex_plant(lambda t: t.replace(r"excludes zero ($n{=}100$; App.", r"excludes zero (App.", 1))),
        ("V17b sample size reverted to the superseded n=80", "POSITIVE/maintext",
         tex_plant(lambda t: t.replace(r"excludes zero ($n{=}100$; App.", r"excludes zero ($n{=}80$; App.", 1))),
        ("V18 'one random draw' dropped from the random-dictionary control", "POSITIVE/maintext",
         tex_plant(lambda t: t.replace(r"(Table~\ref{tab:randsae}; one random draw)", r"(Table~\ref{tab:randsae})", 1))),
        ("V19 non-detection sentence turned categorical", "POSITIVE/maintext",
         tex_plant(lambda t: t.replace("these are non-detections", "GRPO leaves no footprint", 1))),
        ("V20 post-hoc label dropped from 1/35", "POSITIVE/maintext",
         tex_plant(lambda t: t.replace("c; computed after the fact, not registered)", "c)", 1))),
        ("V21 'up to'/upper-bound dropped from the hub-ablation ratio", "POSITIVE/maintext",
         tex_plant(lambda t: t.replace("moves the hub up to several times more", "moves the hub several times more", 1))),
        ("V22 exchangeability condition dropped from the conclusion", "POSITIVE/conclusion",
         tex_plant(lambda t: t[:t.index(r"\section{Conclusion}")]
                   + t[t.index(r"\section{Conclusion}"):].replace("under run exchangeability, $k$ runs", "$k$ runs", 1))),
        ("V23 coupling by 'the sibling <x> analysis' (cold read 2026-09-25)", "CROSS-PAPER",
         tex_plant(lambda t: t.replace("--- a corrected steering analysis, computed on",
                                       "--- the sibling steering analysis, computed on", 1))),
        ("V24 residual re-read as tracking post-training intensity", "post-training intensity",
         tex_plant(lambda t: t.replace("not light GRPO; with one heavy checkpoint",
                                       "not light GRPO, and the residual scales with post-training intensity; with one heavy checkpoint", 1))),
        ("V27 DPO dose-dependence reasserted in the body (S1)", "DPO dose-dependence",
         tex_plant(lambda t: t.replace(r"\section{Conclusion}",
             r"\section{Conclusion}" + "\nDPO's band membership is dose-dependent across the three doses.", 1))),
        ("V28 GRPO s1 marginal case reasserted in the body (S2)", "GRPO s1 marginal",
         tex_plant(lambda t: t.replace(r"\section{Conclusion}",
             r"\section{Conclusion}" + "\nGRPO stays inside except one marginal case (s1).", 1))),
        ("V29 build rule deleted from Sec. 3 (S7)", "POSITIVE/maintext",
         tex_plant(lambda t: t.replace(r"\emph{Build rule:}", "", 1))),
        ("V30 abstract 'no consistent dose ordering' reverted", "POSITIVE/abstract",
         tex_plant(lambda t: t.replace("show no consistent dose\nordering", "are dose-ordered", 1))),
        ("S6 retired verb planted in a FIGURE axis label", "RETIRED verb", figure_plant_rewiring),
        ("S1 violation planted in a FIGURE source", "/FIGURE]", figure_plant),
        ("S4 CROSS-PAPER coupling planted in a FIGURE source", "CROSS-PAPER", figure_plant_coupling),
        ("S5 CROSS-PAPER coupling planted in COMPILED PDF text", "CROSS-PAPER", pdf_plant_coupling),
        ("S3 FIGURE violation behind a literal \\n escape (leading-\\b killer)", "/FIGURE]",
         figure_plant_escaped),
        ("S2 violation planted in COMPILED PDF text", "/PDF]", pdf_plant),
    ]

    caught = on_target = broken = 0
    for name, expect, run in plants:
        after = run()
        if after is None:
            print(f"  [BROKEN ] {name}: plant did not apply")
            broken += 1
            continue
        delta = after - baseline
        got = bool(delta)
        hit = any(expect in d for d in delta)
        caught += got
        on_target += hit
        tag = "CAUGHT " if got else "MISSED "
        if got and not hit:
            tag = "OFFTGT "
        detail = ""
        if delta:
            detail = "   -> " + sorted(delta)[0].splitlines()[0]
        print(f"  [{tag}] {name}{detail}")
        if got and not hit:
            print(f"           expected a finding tagged {expect!r}; the delta does NOT "
                  f"contain one -- the rule under test did not fire")

    n = len(plants)
    print(f"\nselftest (baseline-aware findings-delta scoring):")
    print(f"  caught     {caught}/{n}   (finding set grew)")
    print(f"  on-target  {on_target}/{n}   (the rule under test actually fired)")
    if broken:
        print(f"  BROKEN     {broken}   (plant did not apply)")
    ok = (caught == n and on_target == n and broken == 0)
    print("  verdict: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", "--plant", action="store_true", dest="selftest",
                    help="plant known violations and confirm the gate fails on each")
    args = ap.parse_args()
    tex = open(TEX).read()
    if args.selftest:
        return selftest(tex)
    bad = scan(tex)
    for b in bad:
        print("VIOLATION  " + b)
    print("retraction guard: %d negative (%d factual, %d wording) + %d positive rules: %s"
          % (len(NEGATIVE), sum(1 for r in NEGATIVE if r[2] is FACTUAL),
             sum(1 for r in NEGATIVE if r[2] is WORDING), len(POSITIVE),
             "PASS" if not bad else "FAIL"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
