# graph_build_audit_20261005: CPU-side records for PREREG_graph_build_audit_20261004 §2

Collected 2026-10-05 04:12–05:30 (+08:00) on the cluster login node. **No GPU job was submitted.**
The GPU steps of §1, §3, §4 and the GPU part of §2 wait for the author's D4 approval.

Prereg: `notes/PREREG_graph_build_audit_20261004.md` (33836ca; §6 cea06cb).
Structured receipt: `cpu_records.json`.

Raw records in this directory:
- `sacct_graph_jobs.txt`: sacct for every graph and featstats job.
- `sacct_sig_20260628_0806.txt`: all `sig-*` jobs from 06-28 to 08-06.
- `node_gres.txt`: `scontrol show node` GPU type for each build node.
- `rs_qwen_env_distinfo_after_20260501.txt`: package install times in `rs-qwen-env`.

## Which GPU built which graph (sacct + job-log "saved →" lines)

| Graph(s) | JID | Node / GPU | Built (EDT) | Rebuild on H200 (09-26) |
|---|---|---|---|---|
| Qwen base `attribution_graph_v1_…l14to22` | 11897524 | <NODE> / **L40S** | 06-27 01:39 | **differs**, D 0.0367 |
| GRPO baseline@100 | 11953743 | <NODE> / **A100** | 06-28 04:48 | untestable (checkpoint deleted) |
| audited SFT | 11953881 | <NODE> / **A100** | 06-28 04:55 | **differs**, D 0.0344 |
| GRPO s4@300 | 11967628 | <NODE> / **L40S** | 06-28 12:52 | **differs**, D 0.0325 |
| GRPO SHAPED@150, s5@300 | 11967672 | <NODE> / **L40S** | 06-28 12:55 | **differ**, D 0.0334, 0.0307 |
| GRPO s1@200, s3@100 | 12113165 | <NODE> / H200 | 06-30 17:53 | bit-exact |
| LoRA-SFT seeds 0–3 | 14851207 | <NODE> / H200 | 07-27 21:02 | bit-exact |
| DPO 29-step seeds 0–3 | 15103738 | <NODE> / H100 | 08-01 01:05 | bit-exact |
| DPO 60-step seeds 0–3 | 15103738 | <NODE> / H100 | 08-01 01:05 | not tested (same job as the 29-step graphs) |
| DPO 12-step seeds 0–3 | 15102095 | <NODE> / **A100** | 07-31 23:57 | not tested |
| DPO matched-construction 29-step | 15240602 | <NODE> / H200 | 08-03 19:23 | not tested |
| Llama base `tulu_v2_base_l16to25` | 12312283 | <NODE> / **A100** | 07-02 20:27 | not tested |
| Llama LSFT seeds 0–3 | 15245979 | <NODE> / H200 | 08-03 21:53 | not tested |
| featstats base, GRPO | 11984904 | <NODE> / **A100** | 06-28 21:18 | — |
| featstats s1, s3, s4, s5 | 12125673 | <NODE> / **L40S** | 06-30 22:29 | — |
| (shaped) mwhanna 1.7B l14→22 | 11898371 | <NODE> / **L40S** | 06-27 02:19 | — |

All jobs used `$WORK/envs/rs-qwen-env/bin/python`, `--dtype bfloat16` and the same graph flags.

**Pattern.** Each of the 5 graphs that failed to rebuild was built on L40S or A100. Each of the 10
that rebuilt bit-exactly was built on H100 or H200. The two architectures share compute capability
sm_90.

## Candidates ranked by what the records support

| Rank | Candidate | Records | Basis |
|---|---|---|---|
| 1 | **R1 GPU architecture** | **supported, not confirmed** | Splits 15/15 tested graphs perfectly. The deciding pair is s4/s5/SHAPED (L40S, 06-28 12:52–13:00) against s1/s3 (H200, 06-30 17:53). Both used builder commit 8074f5d, the same env (no install between 06-28 04:21 and 07-17) and the same model, SAE and dataset revisions. The only recorded differences are GPU, node and two days. |
| 2 | R7 nondeterminism off Hopper | not separable from R1 by records | Determinism is shown only on H200: four per-job rebuilds were identical, and the July/August graphs rebuild bit-exactly. Two same-job builds on L40S would separate R7 from R1. |
| 3 | R8 attention / kernel defaults | no setting change on record | TransformerLens computes attention itself, and bf16 appears in every script. An arch-dependent GEMM or split-K choice would be how R1 acts, not a separate cause. |
| 4 | R3 builder code version | disfavoured | `git diff 733ecb1..HEAD` leaves the default numeric path unchanged: pile-10k loader, first 64 tokens, last-position state, SAE encode, top-k. s4/s5/SHAPED and s1/s3 share a commit but have opposite outcomes. Residual: the June cluster-side copy was overwritten on 09-26 (today's cluster copy, sha256 `075b502c…`, equals local HEAD). |
| 5 | R2 library versions | disfavoured | Every numeric package dates from ≤ 2026-04-30: torch 2.6.0+cu124, cuBLAS 12.4.5.8, transformers 4.57.1, TransformerLens 3.1.0, sae_lens 6.43.0, safetensors 0.7.0. Conda history has one event (2025-11-13). Installs inside the window were non-numeric only: ftfy, wordfreq, langcodes, locate and msgpack (06-28 04:21), datalad (07-17), h5py, nibabel. |
| 6 | R4 probe data | disfavoured | pile-10k last hub commit is 2022-10-14 (`127bfedc`). The local snapshot is that same revision, re-created 07-07. Loader code unchanged. |
| 7 | R5 SAE weights | eliminated (records) | Hub repo last commit 2026-05-13. The `.locks` blob ids `7d850585…` (layer14) and `fea01bf5…` (layer22) equal today's hub LFS oids. |
| 8 | R6 model weights | eliminated (records) | Qwen3-1.7B main last changed 2025-07-26 (`70d244cc`), and the local snapshot is `70d244cc`. |

What the records favour is **R1, GPU architecture** (L40S/A100 against Hopper). They cannot separate
it from date, because no non-Hopper graph built after 06-30 has been rebuild-tested yet. Under the
prereg, "confirmed" still needs the GPU test: an L40S rebuild in the current env that reproduces the
stored base bit-exactly.

## Consequences for the other open items (facts from the table; not results)

- **DPO 12-step:** the four graphs were built on A100. If R1 holds, the "same-environment" 1/4 in
  `samebuild_recompute.json` pairs A100 run graphs with an H200 base, so it is still mixed-build. The
  60-step graphs come from the same H100 job as the verified 29-step graphs. §3's 12-step rebuild on
  H200 is also the discriminator between date and architecture.
- **Llama replication:** the base was built on A100 (07-02) and the seeds on H200 (08-03), the same
  split as the Qwen failure.
  - Weight source differed: the 07-02 base read the official `meta-llama` cache, while the 08-03 seeds
    read the NousResearch mirror through a symlink created 08-03 21:53.
  - All four safetensors shards have identical LFS oids (`f8b9704a…`, `c28b25e7…`, `d8e9504d…`,
    `e4486f35…`; official queried with the HF token on the cluster), so the weights themselves are not a
    difference.
- **Battery:** the 06-28 files are A100, the 06-30 files are L40S, and neither is Hopper.
- **shaped:** the mwhanna 1.7B graph behind the 1.7B M1 panels was built on L40S.

## Prerequisites found missing (must be fixed before any GPU job)

- The `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50` snapshot is gone from the HF cache; only `.locks`
  remain.
- `models--NousResearch--Meta-Llama-3.1-8B` is absent, so `models--meta-llama--Llama-3.1-8B` is a
  dangling symlink.
- `fnlp/Llama3_1-8B-Base-LXR-32x` is absent; only `.locks` remain.
- Available:
  - all adapters: lsft 0–3, dpo 12/60/mc29 0–3, audited SFT;
  - merged GRPO s1@200, s3@100, s4@300, s5@300. SHAPED@150 merged dir missing; not needed.
- /scratch was at 82.3% (4.11/5.00 TB).

## Prepared, not submitted (copied to the cluster; new file names only, no overwrite)

| Step | Script | Smoke command | Full run |
|---|---|---|---|
| 0 cache (CPU, online) | `run_sig_gba_prep_cache.sbatch` | — | `sbatch code/scripts/slurm/run_sig_gba_prep_cache.sbatch` |
| §1 Llama | `run_sig_gba_llama.sbatch` | `sbatch --export=ALL,SMOKE=1,TAG=smoke …/run_sig_gba_llama.sbatch` | `sbatch --time=<from smoke> --export=ALL,TAG=full,GBA_TIME_FROM_SMOKE=<JID> …` |
| §3 DPO | `run_sig_gba_dpo.sbatch` | `SMOKE=1` (base + 1 DPO graph) | same pattern |
| §4 battery | `run_sig_gba_featstats.sbatch` | `SMOKE=1` (base featstats) | same pattern |
| §2 R1/R7 | `run_sig_gba_arch.sbatch` | `ARCH=l40s,SMOKE=1` | `ARCH=l40s` on <SLURM_PARTITION>; `ARCH=a100` with `-p <SLURM_PARTITION>` |

Script details:
- Shared setup: `gba_common.sh`. It exports the HF variables one per line with explicit paths (prereg
  §6), refuses a full run without `GBA_TIME_FROM_SMOKE`, writes `gba_env_record.py` output per job, and
  never substitutes a missing checkpoint (it exits 3 = NOT_RUN).
- Readouts: `code/scripts/gba_read.py`, stdlib only.
  - The selftest passes. It was mutation-tested with 8 single-line mutants. Two survived the first
    version of the selftest (the battery branch order and the four-file minimum); both were closed by
    new tests and are now killed.
  - On the stored receipts it reproduces the known values: stored base D 0.0367, s1 bit-exact,
    per-job bases identical, DPO 12/60 = 0/4 and 4/4 against the stored base.
- **§1.0 result (CPU, prereg-specified):** r_stored at full precision = **1.6923**, so the LL threshold
  is h = **1.3462**.

Runtime for every GPU job is unknown until its smoke is measured, as the prereg requires. Graph-only
jobs with repeated model loads risk the < 50% GPU-utilisation cancellation; the smoke's `sacct`
gpuutil decides whether to batch.

## GPU phase (D4 approved; relayed by the coordinator 2026-10-05)

**Cache prep (CPU), JID 19175467, <NODE>, 2 min 17 s.**
- Downloaded into the audit's private hub cache `signature/hf_cache/hub`. The three repos that
  already existed in the shared cache were symlinked in, not copied.
- All 8 blobs match `expected_blobs.txt` (`[HASH OK]` ×8). Those expected values are the June–August
  `.locks` ids, which equal the current hub LFS oids.

**Correction to the CPU records.** `transformer_lens` 3.2.1 and `sae_lens` 6.43.0 are imported from
`~/.local` (installed 2026-05-10), not from the env. Nothing in `~/.local` was installed after
05-10, so the R2 verdict does not change (see `cpu_records.json`).

**Smokes**

| JID | Step | GPU | Measured wall | gpuutil | Status |
|---|---|---|---|---|---|
| 19175665 | §4 featstats base | H200 <NODE> | 115 s per unit; 2 min 11 s job | 48 | done |
| 19175668 | §2 L40S base build | L40S <NODE> | 108 s per graph; 2 min 05 s job | 73 | done |
| 19175661 | §3 DPO (base + DPO12 s0) | H200 <NODE> | — | — | running when the cluster SSH socket dropped |
| 19175657 | §1 Llama base | h200 | — | — | pending when the socket dropped |
| 19175671 | §2 A100 audited SFT | a100 | — | — | pending (QOSGrpGRES) |

**Smoke 19175668 result.**
- The Qwen base graph built on L40S in today's environment (builder at HEAD, torch 2.6.0+cu124)
  is **bit-exact with the stored 06-27 base**: D_edge 0, identical weights.
- The same graph against the 09-26 H200 rebuild gives D_edge 0.0367.
- Receipt: `run/arch_l40s_smoke_vs_stored.json`.
- Only one factor was changed, GPU architecture. Under the prereg §2 definition this **confirms R1 for
  the base graph**.
- The same-job repeat on L40S (R7) is in the full job and has not run yet.

**Remaining smokes (read 2026-10-05 22:1x CST)**

| JID | Step | GPU | Measured | gpuutil |
|---|---|---|---|---|
| 19175657 | §1 Llama base | H200 <NODE> | 153 s per graph; 2:50 job | 92 |
| 19175661 | §3 Qwen base + DPO12 s0 | H200 <NODE> | 90 s + 94 s; 3:13 job | 35 |
| 19175671 | §2 A100 audited SFT | A100 <NODE> | 115 s; 2:10 job | 53 |

**Continuity checks on the smoke graphs** (`run/smoke_continuity.json`; unregistered extra pairs, reported descriptively)

| Rebuild | Against | D_edge | Bit-exact |
|---|---|---|---|
| audited SFT on **A100** (today) | stored audited SFT (A100, 06-28) | 0 | **yes** |
| Qwen base on H200 (today) | 09-26 H200 rebuild `base_X` | 0 | yes |
| DPO 12-step seed0 on H200 (today) | stored (A100, built **07-31**) | 0.0348 | no |
| Llama base on H200 (today) | stored `tulu_v2_base` (A100, 07-02) | 0.0225 | no |

What these show:
- Rebuilding on the stored graph's own architecture reproduces it.
- The DPO12 graph was built on A100 a month after 06-30 and still does not reproduce on Hopper.
  That breaks the date-vs-architecture collinearity: the cause is architecture, not date.

**Full jobs submitted** (`--time` = roughly 2× the projection from the measured per-graph time)

| JID | Step | Content | Projected wall | `--time` |
|---|---|---|---|---|
| 19223775 | §1 Llama | base ×2 + 4 LSFT on H200 | ~18 min | 45 min |
| 19223776 | §3 DPO | base + DPO12 ×4 + DPO60 ×4 + mc29 ×4 on H200 | ~21 min | 50 min |
| 19223777 | §4 battery | featstats base + s1 + s3 + s4 + s5 on H200 | ~10 min | 25 min |
| 19223778 | §2 L40S | base ×2 (R7) + GRPO s4@300 | ~6 min | 20 min |
| 19223779 | §2 A100 | audited SFT ×2 (R7) + DPO12 s0 | ~6 min | 20 min |

Projected total ≈ 1.0 GPU-h. gpuutil risk: the §3 smoke ran at 35. The job is kept to one short run
(~21 min) rather than batched, because batching would change the per-prompt forward that the
bit-exact comparison depends on.

## Full-job verdicts (2026-10-05, after D4)

| JID | Job | Wall | gpuutil | Readout | Branch |
|---|---|---|---|---|---|
| 19223778 | §2 L40S | 5:49 | 78 | `run/arch_l40s_readout_full.json`: base vs stored D 0 (bit-exact); s4@300 vs stored D 0; same-job repeat D 0 | R1 confirmed for the base; R7 excluded on L40S |
| 19223779 | §2 A100 | 6:10 | 50 | `run/arch_a100_readout_full.json`: audited SFT vs stored D 0; DPO12 s0 (stored A100, 07-31) D 0; same-job repeat D 0 | R1 confirmed on A100; R7 excluded on A100 |
| 19223775 | §1 Llama | 13:08 | 91 | `run/llama_readout_full.json` | **LL1** |
| 19223776 | §3 DPO | 21:02 | 35 | `run/dpo_readout_full.json` | **V2** |
| 19223777 | §4 battery | 8:21 | 47 | `run/battery_readout_full.json` | **B3** |

- **§2 root cause: R1, GPU architecture, confirmed** by the prereg's single-factor rule.
  - Every stored graph tested on its own architecture rebuilds bit-exactly: L40S base, L40S s4@300,
    A100 audited SFT, A100 DPO12 s0.
  - The same graphs on Hopper differ by D 0.03–0.037.
  - R7 is excluded: same-job repeats are identical on both non-Hopper architectures.
  - DPO12 was built on A100 on 07-31 and still does not rebuild on Hopper, so the cause is the
    architecture, not the date.
- **§1 LL1.**
  - C_det = 0. C_s = 0 for all four LSFT seeds (H200-built). C_base = 0.0225 (stored base was A100-built).
  - Same-architecture band 0.0222–0.0257, contrasts 0.0442–0.0452, K_out 4/4.
  - r_same = 1.722 ≥ h = 1.346 (r_stored full precision 1.692).
- **§3 V2.**
  - DPO 12-step graphs (A100-built) differ on H200: D 0.031–0.035.
  - DPO 60-step and mc29 graphs are bit-exact.
  - Same-architecture counts: 12-step 3/4, 60-step 3/4, mc29 0/4. With the verified 29-step 1/4, the
    curve is **3/4, 1/4, 3/4**.
  - Pre-registered wording: "no consistent dose ordering; the 0/4, 2/4, 4/4 curve was within the build
    difference".
- **§4 B3.**
  - All five recomputed featstats files differ from the stored ones at stored precision. Stored base and
    GRPO were A100-built; stored s1–s5 were L40S-built.
  - Battery becomes descriptive. Same-architecture descriptive table:
    `run/battery_same_env_descriptive.json` (not a test).
- **GPU use:** smokes 12.5 min + full jobs 54.5 min ≈ 1.1 GPU-h. Two jobs ran below 50% gpuutil
  (35, 47). Both were kept short instead of batched, and neither was cancelled.
