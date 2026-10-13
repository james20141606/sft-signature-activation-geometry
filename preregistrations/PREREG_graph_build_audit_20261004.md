# Pre-registration: graph_build_audit_20261004 (Llama same-environment rebuild, Qwen build-discrepancy root cause, DPO 12/60-step run-graph verification, seed-null battery build check)

**Written 2026-10-04 (+08:00), before any job for this audit exists.** the cluster is down, so nothing
has been run for it. No GPU is in use and no new statistic has been computed. Every number quoted
below already exists in a committed receipt or in the submitted PDF. The facts used to write it are
metadata only: git first-add dates, sbatch partition lists, the top-level keys of graph and
featstats JSONs, and the builder's commit list. If this file changes after a result exists, git
history will show it.

Paper: #60295, submitted PDF sha256 `74d70fdcb0e1…`, tex `a2df410`, frozen until reviews (10-29).
Results from this audit are rebuttal or revision evidence only. They do not change the submitted PDF.

## 0. Why

`results/sftband_20260926/samebuild_recompute.json` (not pre-registered; ELN 2026-09-26T18:05)
and the BootLoops audit (`project_notes/next_cycle_20260927/bootloops_audit_20261003/signature.md`,
F1, F2, F5) found the following:

* The archived Qwen base graph `notes/attribution_graph_v1_qwen3-1p7b_qwenscope_l14to22.json`
  (first added 2026-06-28) does not rebuild in the current environment: D_edge = 0.0367.
* The other 06-28 graphs do not rebuild either: audited SFT 0.0344, GRPO s4 0.0325, s5 0.0307,
  SHAPED 0.0334.
* Graphs built 06-30 or later rebuild bit-exactly: SFT seeds 0-3, DPO@29 seeds 0-3, GRPO s1@200
  and s3@100. The four per-job base rebuilds (four H200 nodes) are bit-identical to each other.
* Against a same-environment base, the DPO 12/29/60-step counts read 1/4, 1/4, 3/4 (paper: 0/4, 2/4,
  4/4). GRPO reads 0/5 (paper: 1/6). The audited SFT z reads 17.8 (paper: 19.9).
* Four things are still open. This file registers them:
  (1) the Llama second-backbone replication has never been continuity-checked;
  (2) the cause of the Qwen discrepancy is undiagnosed;
  (3) the 12- and 60-step DPO run graphs are only assumed same-environment;
  (4) the seed-null activation battery mixes 06-28 and 07-01 metric files.

Definitions used throughout:
* D_edge(G, H) = 1 - |E(G) ∩ E(H)| / |E(G) ∪ E(H)| over the top-10,000 edge sets. This is the
  paper's statistic, computed at full float precision from graph JSONs.
* "Outside" uses the paper's one-sided rule: contrast c_s > max of the seed-pair distances (in D).
  `check_rewiring.py`'s two-sided rule is not used.
* "Bit-exact" means identical edge lists and identical stored weights.

## 1. Llama-3.1-8B base and seed graphs rebuilt in one environment (audit F2)

**What is being checked.** The abstract claims that "the pre-registered estimator replicates on
Llama-3.1-8B with Llama-Scope". Its receipt `notes/circuit_rewiring_llama_sft_seednull.json` pairs
two graph sets built a month apart:
* the base `notes/tulu_v2_base_l16to25.json`, first added 2026-07-02, built by the Tulu staircase
  job;
* four LSFT seed graphs `notes/basediff_llama_sft_seed{0..3}_l16to25.json`, built 2026-08-03 by
  `run_sig_llama_sft_seednull.sbatch`. That script reuses the base because of the line "skip if the
  Tulu v2 base graph already exists".

The two sets also carry different metadata fields: the base has no `dataset`, `keep_side` or
`states_per_prompt` fields.

Stored readout, edge-Jaccard distance:
* band over 6 seed pairs: 0.0222-0.0257;
* contrasts: 0.0434-0.0444;
* r_stored = min contrast / band max = 1.69 at the receipt's four decimals. The paper reports this
  as "1.7×".

**Step 1.0 (CPU, before any GPU job).** Recompute r_stored at full precision from the five stored
graph JSONs. That full-precision value is the r_stored used in the rule below; both values are
reported. Also run the availability check: Llama-3.1-8B base weights, the Llama-Scope `l16r_32x` and
`l25r_32x` SAEs in the HF cache, the four adapters `checkpoints/lsft-seed{0..3}`, and the /scratch
quota (it was at 5.00/5.00 TB on 09-26).

**Step 1.1 (one GPU job).** Run on the partition used on 08-03 (h200). Use the flags of the 08-03
job (`GCOMMON` in `run_sig_llama_sft_seednull.sbatch`), the same operating point (normalized,
fold-norm-scale) and the builder at HEAD. The job builds:
* the base graph twice (B1, B2);
* the four LSFT seed graphs.

The job writes an environment record into the job log and a sidecar JSON:
* python, torch, CUDA, cuBLAS, transformers, transformer_lens, sae_lens and safetensors versions;
* `nvidia-smi` GPU name and driver;
* the builder's git sha and file sha256;
* HF snapshot revisions of the model and both SAEs;
* dtype and attention implementation.

**Readouts.**
* C_det = D_edge(B1, B2).
* C_base = D_edge(stored base, B1).
* C_s = D_edge(stored LSFT_s, rebuilt LSFT_s) for s = 0..3.
* Same-environment band: the 6 pairwise D_edge values among the rebuilt LSFT graphs.
* Same-environment contrasts: c_s = D_edge(B1, rebuilt LSFT_s).
* K_out = #{s : c_s > band max}.
* r_same = min c_s / band max.

**Decision rule** (threshold tied to the paper's 1.7× margin). Let
h = 1 + (r_stored - 1)/2, the ratio at which half of the stored excess over the band maximum would
remain (≈1.35 if r_stored = 1.69).

| Branch | Condition | Consequence |
|---|---|---|
| LL0 (gate) | C_det > 0 | The environment is not deterministic within one job. Stop. No branch. Report C_det and the GPU. |
| LL-C | C_base = 0 and every C_s = 0 | The stored receipt is already same-environment. The 1.7× reading stands as published, and the cross-environment caveat in the rebuttal is withdrawn. |
| LL1 | K_out = 4 and r_same ≥ h | The replication holds under a same-environment build. The revision quotes r_same (2 dp) in place of 1.7× if they differ. |
| LL2 | K_out = 4 and 1 < r_same < h | The replication holds by count, but less than half the stored margin remains. The abstract claim stands. The "1.7×" sentence is replaced by r_same, and the rebuttal says the margin shrank. |
| LL3 | K_out ≤ 3 | The replication does not hold under a same-environment build. The revision removes "replicates on Llama-3.1-8B" from the abstract and introduction, and the rebuttal Q1/Q3 drop item (ii). |
| LL-NR | an adapter, the base weights or an SAE is missing | NOT_RUN. If only adapters are missing, rebuild the base alone and report C_base. Seed graphs then stay unverified, and no LL branch is assigned. |

LL-C is checked first; if it holds, LL1-LL3 are not needed, because they would reproduce the stored
numbers. A descriptive quantity (z, band width, r) never overrides the branch. No family membership,
layer pair, operating point or threshold is changed after results exist.

**Runtime: unknown — to be measured by a smoke.**
* The smoke is one base-graph build with the job's flags, timed end to end from `sacct`, including
  model and SAE load.
* The job's time limit is set from that measurement, not from the Qwen-1.7B jobs (17-26 min on
  09-26). The `elapsed_seconds` field inside the graph JSONs (35-65 s for Llama) covers only the
  graph loop, not loading or adapter merge, so it is not used either.
* GPU-utilisation risk: the 09-26 existing-checkpoint job ran at gpuutil 38, below the 50%
  cancellation line, because it did many short model loads. The smoke also records gpuutil. If it
  is below 50%, the graphs are batched per model load or the job is split.

## 2. Root cause of the Qwen discrepancy (planted-truth step 7; audit F1 item 5)

**Known facts.**
* Bad builds: the 06-28 graphs (base, audited SFT, GRPO baseline, s4@300, s5@300, SHAPED). They were
  committed 2026-06-28 09:57-10:01 PDT. Per the ELN, the base, audited SFT and GRPO baseline were
  built around 04:00-05:40 EDT (01:00-02:40 PDT), and s4, s5 and SHAPED later that day. The ELN
  timestamps are approximate.
* Good builds: the 06-30 graphs (s1@200, s3@100, JID 12113165) and everything later.
* Top weights differ by about 0.1-0.5%, which reshuffles edges near the 10,000 cut.
* Builder commits around that date:
  * `733ecb1` 06-26
  * `adab697` 06-28 01:14 PDT
  * `8074f5d` 06-28 01:58 PDT
  * then none until `548e470` (07-02)
* The sbatch partition lists for both the 06-28 and the 06-30 jobs allow
  `l40s/a100/h100/h200`. The ELN cost lines for 06-28 jobs say "l40s". The 09-26 rebuilds all ran
  on H200.
* The 09-26 ELN records that the cluster copy of the builder was an older file than git HEAD
  ("docstring + --panel-features alias") and was overwritten.

**Candidate causes and the test that discriminates each.** Tests are run in this order: CPU tests
first, and GPU tests only for candidates the CPU tests leave standing.

| # | Candidate | Discriminating test | Predicts |
|---|---|---|---|
| R1 | GPU architecture (bf16 GEMM kernel choice differs by arch: L40S/Ada vs H100/H200/Hopper vs A100) | CPU: `sacct -j <06-28 JIDs>,12113165 --format=JobID,NodeList,Partition` → GPU model per job. GPU: rebuild the base in the current env on the 06-28 GPU type. | 06-28 jobs on a different arch from 06-30 and 09-26; rebuild on that arch = stored (D_edge 0) |
| R2 | Library versions in `rs-qwen-env` (torch, CUDA/cuBLAS, transformers, transformer_lens, sae_lens) changed after 06-28 | CPU: `conda-meta/history` and `*.dist-info` mtimes in `$WORK/envs/rs-qwen-env`; list install events between 06-28 and 06-30 and after | An install event between the bad and good builds touching a numeric library; GPU confirmation only if R1 is eliminated (cloned env pinned to the pre-change versions) |
| R3 | Builder code version (06-28 builds may predate `8074f5d`, or used the cluster-side copy) | CPU: `git diff 733ecb1 adab697 8074f5d HEAD -- code/scripts/m2_attribution_graph.py` restricted to the numerical path (state extraction, dtype, top-k, tie-breaking, weight accumulation); match each 06-28 job's start time to the commit then on torch. GPU: rebuild with the 06-28 script version in the current env | A numerical diff between the script that built the 06-28 graphs and the one that built 06-30; rebuild with old script = stored |
| R4 | Probe data (pile-10k snapshot or prompt selection; the 06-28 and 06-30 JSONs both lack the `dataset`/`keep_side` fields, so the field set alone does not separate them) | CPU: HF dataset cache snapshot revision(s) and their dates; tokenize the first 500 prompts under the 06-28 and HEAD scripts and compare token-id hashes | Different prompt token hashes; this would usually move weights by more than 0.5% |
| R5 | SAE weights revision (`qwen-scope-3-1.7b-base-w32k-l50`) | CPU: HF cache `refs/main` and snapshot list; sha256 of the snapshot blobs vs any older snapshot | Two snapshots, the older one dated before 06-30 |
| R6 | Model weights revision (`Qwen/Qwen3-1.7B`) | CPU: same as R5 for the model repo | as R5 |
| R7 | Nondeterministic kernels within one environment | Already tested on H200 (four per-job rebuilds identical; July/August graphs bit-exact). Remaining GPU test: build the base twice in one job on the 06-28 GPU type | D_edge > 0 between two same-job builds on that arch |
| R8 | Attention implementation or autocast default (sdpa/eager/flash) varying with arch or transformers version | GPU: rebuild with the implementation forced to each option, on the arch found under R1 | Stored graph reproduced under one setting only |
| — | Adapter merge path | Eliminated for the base graph, which has no adapter. Relevant only to run graphs. | — |

**Decision rule.**
* **Confirmed:** one factor, changed alone in the current environment, reproduces the stored 06-28
  base graph bit-exactly. That means D_edge = 0 and every stored weight equal at its stored
  precision.
* **Contributing:** the factor lowers D_edge from 0.0367 but not to 0. It is reported as
  contributing, not as the cause.
* **Unresolved:** no single factor reproduces the stored graph. The report lists the eliminated
  candidates and makes no causal claim.

Whatever the outcome, the revision scores every contrast against a base built in the same job as
its run graphs, as `PREREG_sftband_20260926.md` §2 already does. The root cause decides only whether
the June graphs can be regenerated, and whether the other repos' June graphs carry the same risk.

**Runtime: unknown — to be measured by a smoke.** The CPU tests need cluster login access but no GPU.
For each GPU test, one timed base build on the chosen partition is the smoke.

## 3. DPO 12- and 60-step run graphs: same-environment verification (audit F1, F10)

**What is being checked.** `samebuild_recompute.json` marks `dpo12`, `dpo60` and `dpo_mc29` as
`run_graphs_verified_same_env: false`. The 12-step graphs `basediff_dpo_seed{0..3}_l14to22.json`
were first added 07-31. The 60-step graphs `basediff_dpo_seed{0..3}_step60_l14to22.json` were
first added 08-01. Their 1/4 and 3/4 same-environment counts assume these graphs would rebuild
unchanged.

**Job.** One GPU job, H200, builder at HEAD, flags as in `PREREG_sftband_20260926.md` §1. It
builds:
* the base once;
* the 8 DPO graphs (seeds 0-3 at 12 and 60 steps) from their adapters;
* as a separately reported add-on, the 4 matched-construction 29-step graphs.

It writes the same environment record as §1.

**Readouts.**
* Per graph: D_edge(stored, rebuilt).
* For each dose: the band from the rebuilt seed graphs, contrasts against the rebuilt base, and the
  count outside, reported as a/4.

**Decision rule.**

| Branch | Condition | Consequence |
|---|---|---|
| V1 | all 8 graphs rebuild bit-exactly | The 1/4 (12 steps) and 3/4 (60 steps) same-environment counts stand as computed. The "not verified" label is removed. |
| V2 | any graph differs | The counts are recomputed from the rebuilt graphs and the rebuilt base and reported. Every changed graph is listed. The rebuilt counts replace the assumed ones. |
| V-NR | any adapter missing | NOT_RUN for that dose. The label stays. |

**Pre-registered wording for the dose curve**, applied to the final same-environment counts
(c12, c29, c60). c29 is the verified 1/4 unless §2 or a later rebuild changes it.
* If c12 ≤ c29 ≤ c60 and c60 - c12 ≥ 2: keep "band membership is dose-dependent", with the new
  counts.
* Otherwise: replace it with "under a same-environment base the counts (c12/4, c29/4, c60/4) show
  no consistent dose ordering; the 0/4, 2/4, 4/4 curve was within the build difference".

In both cases the revision reports both curves, the stored-base one and the same-environment one.
Neither curve is chosen after the fact.

The matched-construction add-on is reported descriptively; its stored and same-environment counts
are both 0/4 today.

The step-300 GRPO triplet (s1@300 first added 07-14; s4@300 and s5@300 from 06-28) is outside this
item. It needs the s1@300 checkpoint and is listed here only so that nobody reads it as covered.

**Runtime: unknown — to be measured by a smoke** (one DPO graph build, timed from `sacct`). The
same gpuutil caveat as §1 applies.

## 4. Seed-null activation battery: are the 06-28 and 07-01 metric files the same build? (audit F5)

**What is being checked.** `notes/featstats_seednull.json` (App. F; first added 07-01) takes the
base→GRPO delta from:
* `featstats_base_l22.json` and `featstats_grpo_l22.json`, both first added 06-28 from the trio
  job;

and the five-run band from GRPO, s1, s3, s4 and s5:
* `featstats_s{1,3,4,5}_l22.json`, first added 07-01 from the 7-checkpoint job `12125673`.

The 06-28 files carry `fvu` and `fve` fields that the 07-01 files lack. So at least the script
version differed, but whether the numbers differ is unknown. The GRPO baseline checkpoint has been
deleted, so `featstats_grpo_l22.json` cannot be recomputed.

**Job.** One GPU job runs `code/scripts/sae_feature_stats.py` at HEAD with the stored files' settings
(n_prompts 500, seq_len 64, layer 22, pile) on:
* the base (no checkpoint needed);
* s1@200, s3@100, s4@300 and s5@300 from the merged HF checkpoints used on 09-26.

It writes the same environment record as §1.

**Readout.** For each file and metric: the recomputed value rounded to the stored file's decimals,
compared with the stored value. "Equal" means identical after rounding to the stored precision.
`n_tokens` must match exactly (31,906 in the stored files).

**Decision rule.**

| Branch | Condition | Consequence |
|---|---|---|
| B1 | base and all four s* reproduce on every metric | The 06-28 and 07-01 files are the same build at stored precision. The GRPO file, built in the same job as the base, is inferred same-build and labelled "inferred". The battery stands as built. |
| B2 | base differs on any metric, s* reproduce | The 06-28 files are a different build. The contrast pair (base, GRPO; both 06-28) is internally consistent, but four of the band's ten pairs cross builds. The revision reports the battery descriptively, as a same-environment four-run (s1, s3, s4, s5) band with base→s_i deltas, and removes "carry the claim" and "within a measured band" wording for the three metrics at the precision floor. |
| B3 | any s* differs | The 07-01 files are a different build too. Same consequence as B2, computed only from recomputed files. |
| B-NR | an s* checkpoint missing | NOT_RUN for that run. If the base alone can be recomputed, report it as a partial B1/B2 indication without a branch. |

The battery is not upgraded under any branch. B1 only removes the build question. The other F5
items (GRPO-only band, max-pairwise band, three metrics at the precision floor) stay as disclosed.

**Runtime: unknown — to be measured by a smoke** (one featstats run on the base, timed from `sacct`;
the stored `elapsed_seconds` of 17-31 s covers only the metric loop).

## 5. What does not change the pre-registered reading

* A descriptive quantity (z, band width, ratio, mean) never overrides a branch.
* No threshold, readout, family membership, layer pair, operating point or base choice changes after
  results exist. If a pipeline defect is found after results exist, both the defect and the
  corrected run are reported.
* Unregistered readings found along the way are reported and labelled "not pre-registered", as
  `samebuild_recompute.json` was.
* Each job's command lives in a script under `code/scripts/slurm/`, and each result gets an ELN
  entry with its JID.

## 6. Operational note (added 2026-10-04T03:57:04+08:00, before any job; does not change any rule above)

* Every job script for §1-§4 exports the HF cache variables one per line with explicit paths, as
  in `run_sig_llama_sft_seednull.sbatch:20-22` and the fixed `run_sig_functional_transfer.sbatch:14-16`:
  `export HF_HOME=$WORK/.cache/huggingface`, then
  `export HF_HUB_CACHE=$WORK/.cache/huggingface/hub` on its own line. `HF_HUB_OFFLINE`
  also goes on its own line.
* The same-line form `export HF_HOME=… HF_HUB_CACHE=$HF_HOME/hub` expands `$HF_HOME` before the
  assignment, so it is forbidden (`project_notes/next_cycle_20260927/HF_ENV_DEFECT_INVENTORY.md`).
  Before submitting, each new script must give 0 hits on that inventory's grep.
* The environment record of §1 includes the resolved `HF_HOME`/`HF_HUB_CACHE` values. A cache
  path is part of the build environment (R4-R6 in §2).
