# PREREGISTRATION: a MATCHED-CONSTRUCTION DPO null (closes the L8 band asymmetry)

Committed BEFORE any reshuffled DPO seed is trained. Git history is the timestamp.
Companion to `PREREGISTRATION_dpo_seed_null.md` and `PREREGISTRATION_dpo_dose_matched.md`;
neither is reopened. The published 12-step D2 verdict and the dose-ladder receipts stand.

## Why

The paper's lead is the disjointness of displacement ranges at the identical 29 optimizer
steps: SFT 0.0588–0.0630 vs DPO 0.0323–0.0365 (both from committed receipts,
`notes/circuit_rewiring_sft_seednull.json` and `notes/circuit_rewiring_dpo_dose29.json`).
The paper itself discloses the one asymmetry that cuts the wrong way (main.tex, "One
asymmetry between our two nulls"): `dpo_warmup.py` uses a fixed, unshuffled index order
identical across seeds, so the DPO band varies by initialization and execution
nondeterminism but NOT by data order, while the SFT band (trl `SFTTrainer`) does include
data-order variance. A variance component structurally absent from one band is
anti-conservative for detecting DPO. The disclosed fix is a matched-construction DPO null
-- per-run seeded reshuffling -- and this registration runs it. External reviewer-model
advice (internal review round 4) independently ranked this the project's first remaining
experiment: it closes an internal-validity gap no second backbone can repair.

## Design (everything not named here is byte-identical to the dose-29 rung)

- Seeds **0–3**, chosen because they are the published null's seeds. Run **once**. No
  reshuffle-seed search of any kind: the reshuffle RNG is seeded with the run seed.
- **29 optimizer steps** only (the primary rung and the dose of the disjointness lead).
  No 12/60 rungs: the question is the band's construction, not the dose curve, and the
  dose curve is already measured.
- One code change, committed with this file: `--reshuffle-per-epoch` in
  `code/training/dpo_warmup.py`. When set, the fixed index order is reshuffled once per
  pass through the data with `random.Random(seed)`; default off preserves the committed
  runs byte-identically. This adds to the DPO band the same variance component class the
  SFT band already contains (data order across runs), which is the entire point.
- Recipe, pair set (`data/bfcl_dpo_pairs.parquet`, NOT rebuilt), lr 5e-6, beta 0.1,
  batch 4 x accum 8, LoRA r16/a32, graph protocol (qwen-scope l14->22, n=500), and
  scoring (`scripts/circuit_rewiring.py`) unchanged.
- Graph receipts: `notes/basediff_dpo_mc_seed{0..3}_step29_l14to22.json`.
  Rewiring receipt: `notes/circuit_rewiring_dpo_mc29.json`, seed family
  DPOMC_s0..3, focus DPOMC_s0, same base and SFT inputs as the dose-29 run.

## Registered readout

Displacement = 1 - edge_jaccard, read per seed from `post_contrasts` of the rewiring
receipt by the committed reader `code/scripts/mc_null_read.py` (written and committed
with this file, BEFORE any receipt exists; refuses partial input). Frozen comparators,
already in committed receipts, restated so they cannot drift:

- SFT 29-step displacement range over seeds s0–s3: **[0.0588, 0.0630]**
  (1 - {0.9412, 0.9378, 0.9393, 0.9370}).
- DPO fixed-order 29-step range: **[0.0323, 0.0365]** — published, not reopened.

## Branches (all written before data)

- **MC1 — disjointness survives construction matching**: the matched-construction DPO
  displacement range [min, max over the 4 seeds] is disjoint from [0.0588, 0.0630]
  (i.e. max < 0.0588). The disclosed asymmetry closes: the paper's disjointness lead
  keeps its caveat paragraph but the "not run here" sentence is replaced by the measured
  result, and the anti-conservativeness concern retires.
- **MC2 — overlap**: any matched-construction seed displaces >= 0.0588. The
  "resolved separation" lead is downgraded IMMEDIATELY (internal review round 4's condition,
  accepted here in advance): the paper reverts to the calibrated seed-null statement
  scoped to fixed-order nulls, reports the overlap in the same paragraph that today
  reports disjointness, and the abstract's disjointness sentence is rewritten. No
  reshuffle re-run, no seed extension, no search for a friendlier band.
- **MC3 — uninformative**: fewer than 4 complete graph receipts (training failure,
  eviction, graph-step failure after retry). Reported as attempted-and-uninformative;
  the existing disclosure stays verbatim.

Evidential basis: 4 runs x 1 displacement each vs a frozen 4-run comparator range; this
is a range-disjointness readout, deliberately distribution-free, same as the published
lead. No z is computed on n=4.

No matched-construction quantity exists at the time of this registration:
`notes/circuit_rewiring_dpo_mc29.json` does not exist, and no `dpo-mc-*` checkpoint
directory exists on the cluster.
