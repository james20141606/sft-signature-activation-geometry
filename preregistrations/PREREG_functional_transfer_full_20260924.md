# PREREG correction + full sweep: differential-edge necessity (functional transfer)

Written 2026-09-24 (Beijing) and committed BEFORE any full-sweep or smoke receipt exists.
It corrects and completes, and does not replace:

- `PREREGISTRATION_functional_transfer.md` (commit 2c4f19c) and its AMENDMENT 1 (0a35185):
  design, doses k in {2,000, 5,000, 10,000}, the faithfulness gate read before J, the
  interaction J, branches F1-F4, "the full held-out set".
- The stop-gated pilot, JID 15406147 (k = 5,000, 100 prompts, receipt
  `notes/functional_transfer_k5000.json`).

## 0. What is already known when this is written (stated so it cannot be hidden)

The pilot's numbers are known: gate passed in both cells; D_SFT moved the output 0.550 nats
(base) / 0.524 (SFT), 65x / 90x the matched random control; J = -0.02701, bootstrap CI
[-0.05544, -0.00120], permutation p = 0.0501. The paper currently reports them. This
correction was therefore written with the pilot in view. The pilot is not evidence in the
sweep's verdict; the sweep is read only through the rules below. The pilot's 100 prompts
(rows 5,000-5,099) are part of "the full held-out set" as originally defined, so they are
inside the sweep's sample (about 2% of it); this is disclosed, not corrected.

## 1. Defects in the pilot, found before any new data

1. **Reader catch-all (the one this correction exists for).** `functional_transfer_read.py`
   ended in an `else` that printed "F2 PHANTOM" for every gate-passing outcome that was
   not F1. F2 as registered requires the CI to cover zero; the pilot's CI excludes zero on
   the negative side, and p = 0.0501 is not < 0.05. The reader called it F2 anyway
   (ELN ~L4290). **Fixed** in the same commit as this file: the branch table is exhaustive
   (Sec. 4), `classify()` asserts that at most one condition holds, and anything the table
   does not name is labelled U, never defaulted into F2. The arithmetic is unchanged: the
   fixed reader reproduces the pilot's printed J, CI and p to the digit and now reads the
   pilot as **U (DISCORDANT)** (`--selftest`, 18/18).
2. **Mismatched SFT pairing.** D_SFT is defined from `notes/basediff_sft_l14to22.json`,
   the graph of the audited SFT run (`hf_model_path: grpo_project/checkpoints/sft-qwen3-1p7b`,
   adapter sha256 5a89aeeb...). The pilot sbatch ran the "SFT" arm with
   `checkpoints/sft-seed0` (adapter sha256 e254626c...), a different seeded SFT run whose own
   graph is `notes/basediff_sft_seed0_l14to22.json`. The registered design ("M in {base,
   SFT}", "C_SFT") means the SFT model whose graph defines C_SFT, so the pilot's SFT arm
   measured the audited run's differential sources inside another run. Corrected below
   (config A).
3. **Frequency matching on one prompt.** The pilot ranked SAE features by mean |z| on the
   first prompt only (64 tokens), so every feature silent there tied at zero and was
   ordered by index. Corrected: rank by the base model's firing rate over the 500
   graph-building prompts (tie-break mean activation), draw R without replacement within
   +-200 ranks. (The pilot's R_SFT had 103 features for 104 targets because duplicates
   were collapsed.)
4. **Model loading.** The pilot used `HookedTransformer.from_pretrained` (weight
   processing); the graph builder used `from_pretrained_no_processing`. The sweep uses the
   graph builder's loader.

Items 2-4 change the instrument, so the sweep is not a continuation of the pilot's numbers;
the pilot is reported as a pilot, with these defects, in the appendix.

## 2. Design (fixed)

Implementation: `code/scripts/functional_transfer_full.py` (this commit). Intervention and
statistic are the pilot's: at `blocks.14.hook_resid_post`, for source-feature set S, add
`(z_resample[S] - z_clean[S]) @ W_dec[S]` at every position (qwen-scope layer-14 SAE,
reconstruction error untouched); Delta(M, S) = mean over prompts of
KL(p_M(.|x) || p_M(.|x, S corrupted)) at the final token.

- **Configurations.**
  - **A (primary, the only one that enters the verdict):** base graph
    `attribution_graph_v1_qwen3-1p7b_qwenscope_l14to22.json` + audited SFT graph
    `basediff_sft_l14to22.json`; SFT model = the audited adapter
    (`grpo_project/checkpoints/sft-qwen3-1p7b`, the run behind the paper's 19.9).
  - **B (descriptive):** SFT graph `basediff_sft_seed0_l14to22.json` + `checkpoints/sft-seed0`
    (a second matched SFT run).
  - **C (descriptive, k = 5,000 only):** the pilot's pairing (A's graphs + sft-seed0) at the
    sweep's n, to show what the pairing defect did.
- **Doses.** k in {2,000, 5,000, 10,000} top edges per graph, as registered. D_SFT / D_base
  = sources of edges in one top-k set and not the other, minus sources in both (pilot rule).
- **Held-out set and n.** "The full held-out set" = rows 5,000-9,999 of
  `NeelNanda/pile-10k` (the pilot's offset; the graphs used the first ~500 rows). A row is
  eligible iff its first 2,000 characters give >= 64 tokens under `model.to_tokens`; the
  first 64 tokens are used (fixed length, so the pilot's pairwise truncation disappears).
  No other exclusion. **n = the number of eligible rows**, fixed by this rule before any
  forward pass: a tokenization-only pass writes
  `notes/functional_transfer_full_heldout_count.json` (n and the sha256 of the row list),
  which is committed before the main job is submitted. The sweep reader refuses any
  primary receipt whose n or row hash differs from that file. Resample partner of eligible
  prompt i is eligible prompt (i+1) mod n.
- **Seeds.** Statistics seed 0; R draws seeded by (0, graph id, k, set id).
- **Smoke.** n = 64, all configurations, written to `notes/smoke/`; checks the pipe only.
  Its numbers are never read for a decision or quoted.

## 3. Faithfulness gate (read first, per k, config A)

Gate_SFT: mean Delta(SFT, D_SFT) > mean Delta(SFT, R_SFT). Gate_base: mean
Delta(base, D_base) > mean Delta(base, R_base). (As registered: a mean comparison.)

## 4. Test and per-k decision rule (config A)

j_i = [Delta_i(SFT, D_SFT) - Delta_i(SFT, D_base)] - [Delta_i(base, D_SFT) - Delta_i(base, D_base)];
J = mean_i j_i. Paired-prompt percentile bootstrap 95% CI [lo, hi] (B = 10,000, seed 0);
two-sided within-prompt label permutation (sign flip of j_i, B = 10,000), p = (#|perm| >= |J| + 1)/(B + 1).
alpha = 0.05 per k.

| branch | condition |
|---|---|
| F4 NOT-A-CIRCUIT | neither gate passes |
| F3 ASYMMETRIC | exactly one gate passes |
| F1 STAGE-SPECIFIC | both gates; J > 0, lo > 0, p < 0.05 |
| F5 REVERSED (new) | both gates; J < 0, hi < 0, p < 0.05 |
| F2 PHANTOM | both gates; lo <= 0 <= hi and p >= 0.05 |
| U DISCORDANT (new) | both gates; anything else (CI and permutation disagree, or J outside its CI) |

F5 and U are added because the registered table named neither and the pilot fell between
them; they carry no claim beyond their definitions.

## 5. Sweep verdict

All three k in the same branch X -> verdict X. Otherwise -> **MIXED**. No multiplicity
correction: a directional verdict requires agreement at all three k. Reader:
`code/scripts/functional_transfer_sweep_read.py` (this commit), the only verdict path.

## 6. Numbers the paper quotes (fixed now)

- Load-bearing ratio: config A, **k = 10,000** (the edge budget of every structural number
  in the paper), the SFT model's own-condition cell: mean Delta(SFT, D_SFT) / mean
  Delta(SFT, R_SFT) and the ratio of medians. Rounded to an integer if >= 10, else one decimal;
  medians one decimal.
- J, CI and p at each k (appendix table; main text gives J at the three k).
- Configs B and C only in the appendix table, labelled descriptive.

## 7. What the paper says under each verdict (main text net +0 lines; appendix gets a subsection + table)

The sentences replaced are the four that describe the pilot: abstract ("A pre-registered
pilot intervention found ... did not support stage specificity"), Sec. 4 "Displacement is
not mechanism" (the pilot clause), Discussion ("our one causal pilot did not support stage
specificity"), Conclusion ("A pre-registered pilot intervention finds ..."). The appendix
paragraph "Does the displacement carry computation?" becomes a subsection with the design,
the four pilot defects, the table, the verdict, this file's path and the receipt paths.

The verb stays **displaces** under every verdict, F1 included: the original registration's
scope bound says a source-side intervention is functional corroboration, not validation of
the co-activation object, and `check_retractions.py` keeps "rewires" retired.

Abstract template: "A pre-registered causal sweep (three edge budgets, {n} held-out prompts)
found the displaced sources load-bearing ({R}x a matched random control in mean, {Rm}x in
median) {PHRASE}, so we claim calibrated structural displacement, not a changed mechanism."

| verdict | PHRASE (abstract / Sec. 4) | Discussion / Conclusion clause |
|---|---|---|
| F2 | "but not stage-specific" | "found no stage specificity" |
| F5 | "but less so in the SFT model than in the base, the reverse of stage specificity" | "found the reverse of stage specificity" |
| U or MIXED | "but did not resolve stage specificity" (+ per-k branches in Sec. 4) | "did not resolve stage specificity" |
| F1 | "and more so in the SFT model than in the base"; the clause after the comma becomes "which corroborates a stage-specific change at the source layer, though the graph remains a co-activation proxy" | "found stage-specific source effects, which corroborate but do not validate the co-activation graph" |
| F3 | "found only one of the two differential source sets load-bearing, so no functional claim is made" (ratio parenthetical dropped) | "found only one differential set load-bearing" |
| F4 | "found the displaced sources no more load-bearing than matched random sources, so we claim calibrated structural displacement only" | "found the displaced sources no more load-bearing than random ones" |

Every verdict is entered, including the ones that weaken the paper (F3, F4, MIXED). The only
conditions under which nothing enters the paper: the primary receipts are not back, read
and integrated with all gates green by **2026-09-25 12:00 Beijing**, or the job fails. Then
the paper keeps its current pilot text, and the ELN records why.

## 8. Receipts

`notes/functional_transfer_full_{A,B}_k{2000,5000,10000}.json`,
`notes/functional_transfer_full_C_k5000.json`: pilot schema plus the feature sets, SHA-256
of the script, reader, both preregistrations, all graphs and both adapters, the held-out row
hash, seed, SLURM job id, host, GPU, package versions. Sbatch:
`code/scripts/slurm/run_sig_functional_transfer_{smoke,full}.sbatch`.
