# PREREGISTRATION: an edge-level functional transfer test (does the structural rewiring carry computation?)

Committed BEFORE the test is implemented and before any transfer quantity exists.

## Why

arXiv:2606.06267 ("Many Circuits, One Mechanism") varies token frequency, finds structurally
distinct circuits implementing the SAME computation, and concludes "structural differences
between circuits are not sufficient evidence for distinct mechanisms", prescribing edge-level
evaluation plus cross-condition transfer. Our headline -- LoRA-SFT displaces the co-activation
attribution graph ~20 SD beyond its own seed null -- is exactly a structural edge-level claim.
This test asks whether the displaced edges carry different computation, or whether the
displacement is "phantom".

## Design (2x2, both directions, faithfulness gated FIRST)

For each model M in {base, SFT} and each circuit C in {C_base, C_SFT}:
- selected edges carry CLEAN activations; every unselected edge receives its matched
  RESAMPLE-corrupted activation (resample patching, not zero ablation);
- both circuits carry exactly the same number of edges;
- size-matched RANDOM-edge masks run as a control in every cell;
- evaluation on a FRESH held-out Pile split, disjoint from the split the graphs were built on.

Primary readout, at the final token:
    K(M,C) = E_x [ KL( p_M(.|x) || p_{M,C}(.|x, x_resample) ) ]        lower = better
    R(M,C) = 1 - K(M,C) / K(M, empty)                                  normalized recovery
where C=empty is the all-resample-ablated baseline.

**FAITHFULNESS GATE, read before any transfer statistic**: each own-condition circuit must
reach R >= 0.80 AND beat its size-matched random mask. If it does not, no transfer number is
read (branch F4).

Decisive statistic (diagonal advantage), with a paired-prompt bootstrap CI:
    I = [K(SFT, C_base) - K(SFT, C_SFT)] + [K(base, C_SFT) - K(base, C_base)]

Secondary: top-1 agreement with the full model; held-out-token NLL.

## Doses and staging (stop-gated)

Pilot: k=5,000 edges, 100 held-out prompts. Continue ONLY if the gate passes. Full run then
adds the 2x2 matrix over k in {2,000, 5,000, 10,000} on the full held-out set -- the same
cutoffs the structural result already survives.

## Branches (all before data)

- **F1 STAGE-SPECIFIC**: I > 0 with the bootstrap CI excluding zero, gate passed in all four
  cells -> the displaced edges carry different computation; "rewiring" is earned and the
  headline stands as a mechanism claim.
- **F2 PHANTOM**: gate passes and the CI on I covers zero -> structural displacement without
  functional specialization. Reported positively and prominently: seed calibration detects
  real STRUCTURAL change, and structural change still does not identify a new mechanism. The
  paper's claim is rewritten to "calibrated structural displacement", the word "rewires" is
  retired from the abstract and title-adjacent prose, and 2606.06267 is credited for the test.
- **F3 ASYMMETRIC**: only one circuit clears the faithfulness gate -> extraction-quality
  asymmetry, not specialization; reported as such, no transfer claim.
- **F4 NOT-A-CIRCUIT**: neither own-condition circuit clears the gate -> the top-k
  co-activation object is not an executable circuit; we retain ONLY the structural claim and
  say so in the body, not a footnote.

Every branch is informative and F2/F4 are the ones that cost us the most; they are registered
first-class for that reason.

Reader: code/scripts/functional_transfer_read.py, committed with the implementation before any
receipt, refuses partial input, checks the faithfulness gate before reading I, prints the
branch verbatim.

**Scope bound stated now**: our graph edges are non-negative co-activation products, not
executable connections. If per-edge gating is not implementable on this object, the test runs
on the residual-component graph via factorized resample patching and is reported as functional
CORROBORATION of the structural result, not validation of the exact 20-SD object -- and that
distinction goes in the paper.

No transfer quantity exists at the time of this registration.

## AMENDMENT 1 — the 2x2 as registered is underpowered; test the DIFFERENTIAL edges instead

Committed after a POWER PRE-FLIGHT on the already-committed graphs (no new data, no transfer
quantity computed, nothing tuned to an outcome) and BEFORE any implementation.

**The finding.** C_base and C_SFT overlap far more than the displacement statistic suggests:

| k | edge Jaccard | source-feature Jaccard | base-only srcs | SFT-only srcs |
|---|---|---|---|---|
| 2,000 | 0.931 | 0.921 | 44 | 27 |
| 5,000 | 0.941 | 0.947 | 69 | 24 |
| 10,000 | 0.937 | 0.937 | 113 | 60 |

The registered 2x2 keeps ~94% of the same features clean in both arms. Under that design a null
on I would be a power failure wearing the costume of branch F2, and F2 is the branch that would
cost us the abstract's verb. Registering the fix now is cheaper than misreading it later.

**The replacement primary test: differential-edge necessity.** Let D_SFT = edges in C_SFT and
not in C_base, D_base = edges in C_base and not in C_SFT (|D| ~ 630 at k=10,000, the 6.3%
displacement). Everything else in the design is unchanged (resample patching, held-out Pile
split, size-matched random control, faithfulness gate read first).

For model M in {base, SFT} and edge set D in {D_SFT, D_base, R (size-matched random,
frequency-matched by source-activation decile)}: corrupt ONLY D's source features (their
clean SAE contributions replaced by resampled ones at the source layer, reconstruction error
left intact) and measure

    Delta(M, D) = E_x [ KL( p_M(.|x) || p_M(.|x, D corrupted) ) ].

Decisive statistic, paired-prompt bootstrap CI:

    J = [Delta(SFT, D_SFT) - Delta(SFT, D_base)] - [Delta(base, D_SFT) - Delta(base, D_base)]

J > 0 means the SFT-specific edges matter more in the SFT model than in the base model,
relative to the base-specific edges -- an interaction, so any main effect of "SFT-only features
are simply more important everywhere" is differenced out.

**Faithfulness gate, unchanged in spirit, restated for this design**: Delta(M, D) for each
own-condition differential set must exceed its size- and frequency-matched random control R in
the same model, or that cell is uninterpretable (branch F3/F4 as before).

**Branches remap, thresholds unchanged**: F1 if J > 0 with CI excluding zero; F2 if the gate
passes and the CI covers zero (structural displacement without functional specialization --
abstract verb retired as already registered); F3 if only one cell beats its random control;
F4 if neither does.

Everything the original registration fixed -- doses k in {2,000, 5,000, 10,000}, held-out
split, the stop-gated pilot, the reader, and the requirement that the gate be read before J --
carries over unchanged.
