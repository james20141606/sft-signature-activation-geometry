# Pre-registration: sftband_20260926 (full-parameter SFT band, k=20 LoRA-SFT band, readout robustness)

**Written 2026-09-26 03:05-03:40 +08:00, before any job for this campaign has been
submitted and before any of the new statistics below has been computed on any data.** torch is
unreachable at the time of writing (institutional VPN not connected), so no checkpoint, graph or state for
this campaign exists yet. The only numbers quoted here are ones that are already in the paper or in
committed receipts. If this file changes after a result exists, git history will show it.

Owner: strengthen agent for #60295 (STRENGTHEN_BRIEF.md). The tex is owned by the writer agent;
this campaign produces receipts only.

## 0. Why

The three most frequent (c)-type objections in `final_review_20260924/ai_review_sim/signature_cc.md`
that fit in one day of compute:

* W2 (A, C): SFT/DPO are LoRA-only; LoRA may carry intruder dimensions; no full-parameter SFT
  seed band. -> **Analysis F** (full-parameter SFT, k=4, same recipe).
* W8 + B-W2 (A, B): k=4 is too few; the band SD comes from six dependent pairs; by Prop. 1 the SFT
  claim can never certify better than 1/5. The paper itself says one recipe certifies p<=0.05
  only with k>=19 reruns, all outside (Sec. 6). -> **Analysis L** (LoRA-SFT band at k=20).
* W5 / W28 / C-W4 (A, B, C): the band is only shown for co-activation edge Jaccard; would another
  diffing readout on the same states agree? -> **Analysis R** (four further readouts, same
  checkpoints, same probe states).

Not run, with reasons: more GRPO reruns (a 300-step run is ~7.5 h on 4 GPUs at the measured
~90 s/step, `run_grpo_resume_seed_step300.sbatch`; it cannot finish before the 11:00 cut and
violates the single-GPU <=3 h rule); crosscoders (training one per rerun pair is days of compute).

## 1. Fixed pipeline (identical to the paper's)

* Backbone `Qwen/Qwen3-1.7B`; SAE `qwen-scope-3-1.7b-base-w32k-l50`, layer14 -> layer22;
  `m2_attribution_graph.py --n-prompts 500 --seq-len 64 --dtype bfloat16 --keep-top-n 10000
  --per-state-top-k 2000`, pile-10k, keep-side first, one (last-position) state per prompt.
  New flag `--dump-states` only records the states the graph is built from; it does not change
  the graph (pass-through generator).
* SFT recipe (`code/training/sft_warmup.py` via `code/scripts/sft_seed.py`, env
  `$WORK/envs/shaped-2026`): 900 BFCL rows (`data/bfcl_sft_train.parquet`), 1 epoch
  = 29 optimizer steps, lr 5e-5, batch 8 x grad-accum 4, max-len 2048, bf16, seed set before the
  model is built.

## 2. Analyses and estimands

Displacement D = 1 - S for a similarity S; contrast = D(base, run); band = D over all C(k,2)
pairs of same-recipe runs. "Outside" = contrast strictly greater than the band maximum.
E_b = all k contrasts outside (the Prop. 1(b) event); if E_b, the Prop. 1(b) ceiling under run
exchangeability is 1/(k+1). z = (contrast - band mean)/band SD is descriptive only. The base is
the runs' common origin, so as in the paper the ceiling is stated "under run exchangeability" and
is not claimed conservative.

### Analysis L — LoRA-SFT at k=20 (primary statistic: edge Jaccard)
* Runs: LoRA-SFT seeds 0..19. Seeds 0-3 reuse the existing adapters
  (`checkpoints/sft-seed{0..3}`) if present on /scratch, else are retrained at the same seed;
  seeds 4..19 are new (16 runs, identical command, only `--seed` differs).
* Every graph (including base and seeds 0-3) is rebuilt in this campaign.
* Contrast for run s uses the base graph built **in the same Slurm job** as run s, so a GPU-model
  difference between jobs can only enter the band (pairs across jobs, widening it), never a
  contrast. Sensitivity: the same analysis with one common base (the one from the
  existing-checkpoint job) is also reported.
* Training gate: a seed is excluded (and k reduced, ceiling recomputed as 1/(k+1)) only if its
  adapter file is missing or its trainer log is absent. Exclusions are listed.
* Branches (edge Jaccard, same-job bases):
  * **L1 support**: 20/20 outside (E_b). Paper may say the SFT displacement exceeds every one of
    the 190 seed pairs of a 20-run band, which under run exchangeability certifies p <= 1/21.
  * **L2 partial**: 16-19 of 20 outside. No certificate. Paper reports the count and keeps
    resting the SFT claim on effect size, replication and the band-free comparison.
  * **L3 weaken**: <= 15 of 20 outside. Paper must narrow: the four-run band overstated the
    separation; the SFT claim is reported as "most reruns" with the count.

### Analysis F — full-parameter SFT, k=4 (primary statistic: edge Jaccard)
* Same recipe, same data, same 29 steps, seeds 0..3; the only change is that all weights are
  trained (`--full-param`). Declared implementation choices, fixed now: fp32 master weights with
  bf16 autocast (the LoRA arm's trainable adapter weights are also fp32), gradient checkpointing
  (memory only), `--logging-steps 1`, `--save-strategy no` (no optimizer checkpoint), weights saved
  in bf16 (the graph builder loads every checkpoint in bf16). LoRA dropout (0.05) has no full-param
  counterpart; Qwen3 has no other dropout. lr stays 5e-5 (not re-tuned: changing it would make this
  a different recipe).
* **Gate T (instrument competence), per run**: last logged training loss < first logged loss, and
  relative weight change ||theta - theta0|| / ||theta0|| > 0 (from `train_receipt.json`, written by
  the trainer process; the loss curve is the trainer's own log). If any run fails T, the arm is
  **INVALID**: reported as a failed arm, no inference, not as F3.
* Branches (edge Jaccard; band = 6 full-param pairs; same-job base):
  * **F1 support**: 4/4 outside. The band verdict extends from LoRA-SFT to full-parameter SFT at
    the same recipe (k=4: descriptive; Prop. 1 ceiling 1/5).
  * **F2 partial**: 1-3 of 4 outside. Report the count; the LoRA-only limitation stays open.
  * **F3 weaken**: 0/4 outside. The SFT band verdict is specific to LoRA-SFT at this recipe; the
    paper must scope every "SFT" claim to LoRA-SFT and say full-parameter SFT at the same recipe
    was not resolved from its own reruns.
* Descriptive only (no branch): full-param vs LoRA mean displacement; full-param band width vs the
  LoRA band width.

### Analysis R — readout robustness (same checkpoints, same 500 probe states)
Readouts (distances; definitions in `code/scripts/sftband_readouts.py`, committed with this file):
`wcos` (1 - weighted edge cosine, recomputed at full float precision from the graph JSONs),
`feat` (relative L2 distance between the mean layer-22 SAE code vectors, i.e. the per-feature
activation difference on the same base SAE), `dense` (mean paired relative L2 distance of the
layer-22 residual states; dictionary-free), `cka` (1 - linear CKA of the layer-22 states;
dictionary-free).
* Families scored: LoRA-SFT (k as in L), full-param SFT (k=4), GRPO (the paper's 6-run band:
  baseline@100, s1@200, s3@100, s4@300, s5@300, SHAPED-lambda0.1@150), DPO at 29 steps (k=4,
  fixed-order runs). A family whose checkpoints are missing on /scratch is reported as missing.
* **SFT branches** (LoRA-SFT family):
  * **RS1 robust**: under all four alternative readouts, all LoRA contrasts are outside.
  * **RS2 partial**: under 1-3 of the four readouts all contrasts are outside; name which fail.
  * **RS3 statistic-specific**: under none of the four. The paper must say the SFT verdict is
    specific to the co-activation statistic.
* **GRPO branches**:
  * **RG1 consistent**: no alternative readout puts all 6 GRPO contrasts outside the GRPO band.
    The non-detection is not an artefact of edge Jaccard among these readouts.
  * **RG2 readout-dependent**: >=1 readout puts all 6 outside. The paper must say the GRPO
    non-detection is specific to the co-activation statistic, and name the readout(s) that separate
    (the GRPO band mixes steps and one reward variant, L7, which makes it wider, so a separation
    under a wider band is not an artefact of band heterogeneity).
* Full-param SFT and DPO@29 under the alternative readouts: reported descriptively (no branch).

### Analysis R0 — stored-graph `wcos` at full precision (CPU, runnable before the cluster is back)
The paper withdrew the weighted-cosine z because the stored receipt rounded it to 4 decimals
(null SD below the reporting step). Recomputing `wcos` at full precision from the committed graph
JSONs for the existing families (SFT k=4, DPO@29 k=4, DPO@12 k=4, GRPO 6-run) is a readout
check that needs no GPU. Reported descriptively with the same outside/E_b counts; it does not
reinstate the withdrawn z=-30.7 (that was against the GRPO-family band).

## 3. Continuity and hardware (reported, not a gate on the branches)
* Each job rebuilds the base graph; pairwise D between the per-job bases and the stored base
  (`notes/attribution_graph_v1_qwen3-1p7b_qwenscope_l14to22.json`) is reported.
* Rebuilt vs stored graphs for LoRA seeds 0-3, audited SFT, GRPO and DPO checkpoints: D_edge reported.
* GPU model per job from `nvidia-smi` in the Slurm log; environment versions logged.

## 4. What does not change the pre-registered reading
* A result under a descriptive quantity (z, band widths, ratios) never overrides a branch.
* No threshold, readout, family membership or base choice is changed after results exist. If a
  pipeline defect is found after results exist, the defect and the corrected run are both reported.

## 5. Paper wording per branch (English, for the writer agent)
* L1: "Raising SFT's band from 4 to 20 seeded reruns (190 seed pairs), all 20 base-to-SFT contrasts
  still lie outside it, the Proposition 1(b) event, which under run exchangeability certifies
  $p\le1/21$ (App. X)."
* L2: "With 20 seeded reruns, n of 20 base-to-SFT contrasts lie outside the 190-pair band, so the
  Proposition 1(b) certificate is not obtained; the SFT claim rests on effect size and replication
  as before (App. X)."
* L3: "With 20 seeded reruns only n of 20 base-to-SFT contrasts lie outside the band; the four-run
  band overstated the separation, and we report SFT's displacement as exceeding most, not all, of
  its reruns (App. X)."
* F1: "Full-parameter SFT with the same data, steps and learning rate also exceeds its own four-run
  band (4/4 contrasts outside; App. X), so the SFT verdict is not specific to the LoRA adapter."
* F2: "Full-parameter SFT with the same recipe puts n of 4 contrasts outside its own band (App. X);
  whether the verdict extends beyond LoRA remains open."
* F3: "Full-parameter SFT with the same recipe is not resolved from its own four reruns (0/4
  outside; App. X); our SFT claims are about LoRA-SFT."
* INVALID: "A full-parameter SFT arm failed its pre-registered training gate and is not interpreted
  (App. X)."
* RS1/RG1: "The band verdicts do not depend on the co-activation statistic: under a weighted-edge,
  a per-feature activation-difference and two dictionary-free readouts of the same states, SFT's
  contrasts stay outside its band and GRPO's do not all leave theirs (App. X)."
* RS2/RS3/RG2: state which readouts agree and which do not, per the branch definitions above, with
  no summary adjective.
