#!/usr/bin/env python3
"""Pin every functional-transfer sweep number in main.tex to the receipts.

Registered in notes/PREREG_functional_transfer_full_20260924.md. Strings are produced by
code/scripts/functional_transfer_tex.py from notes/functional_transfer_full_*_k*.json via
the committed reader (functional_transfer_read.decide), so a hand-typed number, a row or
column swap that preserves the value set, a stale pilot sentence in the body, or a verdict
label that the sweep reader did not produce all fail this gate.

  python3 code/scripts/check_functional_transfer.py [--selftest]
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from functional_transfer_tex import KS, ROWS, strings  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# The four sentences that described the pilot may not come back into the body.
STALE_BODY = [r"pilot intervention", r"causal pilot", r"pilot causal", r"In a pilot,", r"65\\times",
              r"13\.5\\times", r"neither registered branch"]


def norm(t):
    return " ".join(t.split())


def verdict(cells):
    br = [cells[("A", k)]["res"]["branch"] for k in KS]
    return br[0] if len(set(br)) == 1 else "MIXED"


def check(tex, s=None, cells=None):
    if s is None:
        s, cells = strings()
    bad = []
    t = norm(tex)
    a = norm(tex[tex.index(r"\begin{abstract}"):tex.index(r"\end{abstract}")])
    body = norm(tex[:tex.index(r"\bibliography{")])
    app = norm(tex[tex.index(r"\label{app:functional}"):])
    app = app[:app.index(r"\end{table}")] if r"\end{table}" in app else app
    for c, k in ROWS:
        row = norm(s[f"row:{c}:{k}"])
        if row not in app:
            bad.append(f"tab:functional row {c} k={k} != receipt: {row}")
    if s["abstract_ratio"] not in a:
        bad.append(f"abstract ratio: expected {s['abstract_ratio']!r}")
    if s["n"] not in a:
        bad.append(f"abstract n: expected {s['n']!r}")
    # 2026-09-25 de-numbering: the J values moved out of the body into tab:functional
    # (pinned by the row checks above); the body must carry the receipt-derived direction.
    if "body_J_dir" in s and s["body_J_dir"] not in body:
        bad.append(f"body J direction: expected {s['body_J_dir']!r}")
    if s["body_ratio"] not in body:
        bad.append(f"body ratio: expected {s['body_ratio']!r}")
    if s["app_range"] not in norm(tex[tex.index(r"\label{app:functional}"):]):
        bad.append(f"appendix ratio range: expected {s['app_range']!r}")
    v = verdict(cells)
    if f"registered verdict: {v}" not in body:
        bad.append(f"body verdict != sweep reader's {v}")
    if f"sweep verdict: \\textbf{{{v}}}" not in app:
        bad.append(f"appendix verdict label != sweep reader's {v}")
    for pat in STALE_BODY:
        m = re.search(pat, body)
        if m:
            bad.append(f"stale pilot wording in body: {m.group(0)!r}")
    # direction: when the primary configuration returns F5 (reversed) at any budget, the
    # abstract and conclusion must say so; "did not resolve/support" would hide a resolved sign
    n_f5 = sum(cells[("A", k)]["res"]["branch"] == "F5" for k in KS)
    if n_f5:
        concl = norm(tex[tex.index(r"\section{Conclusion}"):tex.index(r"\bibliography{")])
        intro = norm(tex[tex.index(r"\section{Introduction}"):tex.index(r"\section{Related")])
        disc = norm(tex[tex.index(r"\label{sec:discussion}"):tex.index(r"\section{Limitations}")])
        for where, txt in (("abstract", a), ("introduction", intro), ("discussion", disc),
                           ("conclusion", concl)):
            if "reversed" not in txt:
                bad.append(f"{where}: {n_f5} primary budget(s) are F5 but 'reversed' is absent")
        # anywhere in the main text: the F2/MIXED template clauses would hide a resolved sign
        m = re.search(r"did not (resolve|support) stage specificity|found no stage specificity", body)
        if m:
            bad.append(f"main text hides a resolved direction: {m.group(0)!r}")
    return bad


def swapJ(t, s):
    r5, r10 = s["row:A:5000"].split(" & "), s["row:A:10000"].split(" & ")
    r5[6], r10[6] = r10[6], r5[6]
    return t.replace(s["row:A:5000"], " & ".join(r5), 1).replace(
        s["row:A:10000"], " & ".join(r10), 1)


def main():
    tex = open(os.path.join(ROOT, "main.tex")).read()
    s, cells = strings()
    if "--selftest" in sys.argv:
        plants = [
            ("abstract ratio", lambda t: t.replace(s["abstract_ratio"],
                                                   s["abstract_ratio"].replace("mean,", "mean ,"), 1)),
            ("row A10k branch", lambda t: t.replace(s["row:A:10000"].split(" & ")[-1],
                                                    "F9 \\\\", 1)),
            # value-set-preserving swap of the J cells of two primary rows
            ("swap J between A5k and A10k", lambda t: swapJ(t, s)),
            ("stale pilot sentence", lambda t: t.replace(r"\end{abstract}",
                                                         "A pre-registered pilot intervention.\n\\end{abstract}", 1)),
            ("verdict label", lambda t: re.sub(r"sweep verdict: \\textbf\{[A-Z0-9]+\}",
                                               lambda m: "sweep verdict: \\textbf{F2}"
                                               if "F2" not in m.group(0) else
                                               "sweep verdict: \\textbf{F1}", t, count=1)),
        ]
        plants.append(("body ratio", lambda t: t.replace(s["body_ratio"],
                                                          s["body_ratio"].replace("in median", "in medians"), 1)))
        plants.append(("body verdict", lambda t: t.replace("registered verdict: ", "registered verdict: F", 1)))
        plants.append(("appendix range", lambda t: t.replace(s["app_range"], s["app_range"].replace("--", "--4", 1), 1)))
        plants.append(("abstract n", lambda t: t.replace(s["n"], s["n"].replace("816", "861"), 1)))
        plants.append(("abstract direction hidden", lambda t: t.replace(
            "interaction reversed at two budgets", "interaction did not resolve at two budgets", 1)))
        plants.append(("intro stale pilot clause", lambda t: t.replace(
            "a registered causal sweep found the stage-specificity interaction reversed at two of three budgets.",
            "a pilot causal test did not support stage specificity.", 1)))
        plants.append(("discussion F2 clause", lambda t: t.replace(
            "our registered causal sweep found the stage-specificity interaction reversed at two of three budgets.",
            "our registered causal sweep found no stage specificity (reversed at two budgets).", 1)))
        plants.append(("conclusion direction hidden", lambda t: t.replace(
            "stage-specificity interaction reversed at two of three",
            "stage-specificity interaction unresolved at two of three", 1)))
        if "body_J_dir" in s:
            plants.append(("body J direction", lambda t: t.replace(
                s["body_J_dir"], s["body_J_dir"].replace("negative", "positive"), 1)))
            plants.append(("body J CI clause", lambda t: t.replace(
                s["body_J_dir"], s["body_J_dir"].replace("excluding zero", "covering zero"), 1)))
        caught = 0
        for name, f in plants:
            mutated = f(tex)
            if mutated == tex:
                print("PLANT NOT APPLICABLE:", name)
                continue
            if check(mutated, s, cells):
                caught += 1
            else:
                print("NOT CAUGHT:", name)
        print(f"selftest caught {caught}/{len(plants)}")
        return 0 if caught == len(plants) else 1
    bad = check(tex, s, cells)
    for b in bad:
        print("MISMATCH", b)
    print(f"functional-transfer sweep pins ({len(ROWS)} table rows, abstract, body, verdict "
          f"{verdict(cells)}): {'FAIL' if bad else 'PASS'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
