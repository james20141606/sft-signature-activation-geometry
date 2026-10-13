# SFT Leaves a Reproducible Signature in Activation Geometry: Calibrating Post-Training Interpretability Against Run-to-Run Noise

Xupeng Chen (New York University), Chenqian Le (New York University), Jiayi Cheng (New York University),
Runhao Li (University of Southern California), Michael Yue (Columbia University)

A model diff credits post-training with whatever separates a base checkpoint from one post-trained
descendant, presuming a rerun would land close by. On the sparse co-activation graph of `Qwen3-1.7B`
it does not: two LoRA-SFT reruns sit over half as far apart as either sits from the base. We score
each base-to-post contrast against the run-to-run band of its own recipe, and show that k runs
certify no better than p = 1/(k+1) under run exchangeability, so the band z-score is a descriptive
scale. SFT leaves a large, reproducible signature: base-to-SFT displaces the graph by 17.8
pairwise-distance SDs beyond the mean distance between four independent LoRA-SFT runs, and the
pre-registered estimator replicates on Llama-3.1-8B with Llama-Scope. At 29 steps each, four SFT and
four DPO runs occupy disjoint displacement ranges. All bands use graphs built on one GPU
architecture. DPO band counts (3/4, 1/4, 3/4 contrasts outside at 12, 29, 60 steps) show no
consistent dose ordering; light GRPO is a secondary non-detection, not an absence; and the graph
concentration read as an RL signature is already in the zero-RL base. A pre-registered causal sweep
on 4,816 held-out prompts found the displaced sources load-bearing, but their stage-specificity
interaction reversed at two edge budgets and the third failed its faithfulness gate, so the paper
claims calibrated structural displacement, not a changed mechanism.

This repository holds the co-activation graphs, the receipts (the JSON outputs every number in the
paper is read from), the readers and checkers that pin the paper's numbers to those receipts, the
graph builder, the figure scripts, the SFT/DPO trainers with their small BFCL-derived inputs, the job
specifications, and the pre-registrations. It holds no model or SAE weights and no checkpoints.

## Citation

```bibtex
@article{chen2026sftsignature,
  title   = {SFT Leaves a Reproducible Signature in Activation Geometry: Calibrating Post-Training Interpretability Against Run-to-Run Noise},
  author  = {Chen, Xupeng and Le, Chenqian and Cheng, Jiayi and Li, Runhao and Yue, Michael},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## Verify the release (offline, no GPU)

From the repository root, with Python 3.9 or newer and only the standard library:

```bash
python3 code/scripts/verify_release.py   # required files + MANIFEST.sha256 + core statistics + DPO dose curves vs. main.tex
```

`verify_release.py` checks that every file listed in `MANIFEST.sha256` is present with its hash,
runs `check_rewiring.py`, runs the `samearch.py` consistency checks, and compares the DPO
12/29/60-step band counts (same-architecture and archived) with the sentences in `main.tex`.

Further readers and checkers (standard library only unless noted):

```bash
python3 code/scripts/samearch.py             # Table 1 (one GPU architecture) and the archived readings
python3 code/scripts/samearch.py --selftest  # same-arch bases identical, H200 build records,
                                             # archived recompute == committed notes/ receipts
python3 code/scripts/check_table_main.py     # Table 1, archived table, Table 2 vs. main.tex
python3 code/scripts/check_main_numbers.py   # numbers quoted in the body vs. receipts
python3 code/scripts/check_rewiring.py       # SFT/GRPO/DPO run-null summaries, DPO dose curve, Llama replication
python3 code/scripts/functional_transfer_read.py notes/functional_transfer_k5000.json
python3 code/scripts/mc_null_read.py
python3 code/scripts/functional_transfer_sweep_read.py   # registered sweep verdict (MIXED); a few minutes
python3 code/scripts/functional_transfer_tex.py
python3 code/scripts/check_functional_transfer.py
python3 code/scripts/check_retractions.py      # withdrawn claims stay withdrawn in main.tex and main.pdf (needs pdftotext)
```

`check_rewiring.py` recomputes the Qwen SFT/GRPO/DPO run-null summaries, the DPO 12/29/60-step
literal band-membership curve, the matched-29-step SFT/DPO range separation, the
matched-construction control, and the Llama SFT replication directly from the released JSON arrays.
The DPO receipt field `rl_signature_beyond_seed_noise` uses a pre-registered **large-effect** rule
(`base_to_focus < band_min - 0.02`); it is not a literal band-membership flag.
`functional_transfer_sweep_read.py` is the only verdict path for the registered functional-transfer
sweep (`preregistrations/PREREG_functional_transfer_full_20260924.md`): it refuses any primary
receipt whose n or held-out row hash differs from `notes/functional_transfer_full_heldout_count.json`.

`code/scripts/check_retractions.py` additionally reads the compiled paper (`main.pdf`, the arXiv
PDF) and needs `pdftotext` (poppler).

With `numpy` installed, `python3 code/scripts/audit_conventions.py` reruns the top-k convention
audit. With `numpy`, `matplotlib`, `svgutils`, headless Google Chrome and `pdffonts` (poppler),
each `code/analysis/fig*.py` regenerates its data figure from `notes/` and `results/` into a
`figures/` directory at the repository root; `code/analysis/fig1_r1_strip.py` draws Figure 1 from
the same-architecture graphs under `results/` (the source of Table 1). The style module
`code/figstyle/compose.py` expects Chrome at its macOS path; edit `CHROME` there on other systems.

## Directory map

| Path | Contents |
|---|---|
| `main.tex`, `main.pdf` | the paper source and compiled PDF (arXiv version); the checkers cross-check stated numbers against them. `main.tex` builds with the ICLR 2027 style files, which are not included |
| `notes/*.json` | every co-activation graph and receipt the paper names under `notes/` |
| `results/sftband_20260926/` | discussion-period runs registered in `preregistrations/PREREG_sftband_20260926.md` (20-run LoRA-SFT band, full-parameter SFT, alternative readouts): verdict and readout receipts, every graph, build manifest; full-parameter checkpoints were deleted after graphing (`run/meta/full_s*_weights.sha256` kept) |
| `results/sftband_20260926/r0_stored/` | weighted-edge-cosine check at full precision (analysis R0); rerun with `python3 code/scripts/sftband_readouts.py --manifest results/sftband_20260926/r0_stored/manifest.json --out /tmp/r0.json --stats edge wcos` (needs `numpy`) |
| `results/graph_build_audit_20261005/` | the registered build audit (`preregistrations/PREREG_graph_build_audit_20261004.md`): rebuilt graphs, per-job build records (`run/meta/env_*.json`: GPU, driver, package versions), readouts of `code/scripts/gba_read.py`, same-architecture battery files |
| `code/attribution_fn.py`, `code/attribution_graph.py` | graph construction (Eq. 1: per-state top-2,000 products, then global top-10,000 edges) |
| `code/random_sae_baseline.py` | random-direction encoder for the random-dictionary control |
| `code/scripts/m2_attribution_graph.py` | graph-building driver (needs `torch`, `sae_lens`, `transformers`, `peft`, `datasets` and the public model/SAE weights) |
| `code/scripts/*.py` | readers, checkers, seeding wrapper, H3 probe, provenance stamping, `verify_release.py` |
| `code/analysis/`, `code/figstyle/`, `code/fig1_rc.py` | figure generators (read only `notes/` and `results/`) and their style modules |
| `code/training/`, `code/tests/test_dpo_loss.py` | SFT/DPO trainers, data construction, and the DPO-loss unit tests |
| `code/scripts/slurm/` | job specifications for the seeded arms, the transcoder scale sweep and the functional-transfer sweep |
| `data/` | the 900-row SFT input and the 398-pair DPO preference set (BFCL-derived, Apache-2.0; see below) |
| `preregistrations/` | all registrations. The paper cites three of them as `notes/PREREG_*.md` (`functional_transfer_full_20260924`, `graph_build_audit_20261004`, `sftband_20260926`); they are here under `preregistrations/` |
| `LICENSE`, `LICENSES/` | MIT license for this repository's code and receipts; Apache-2.0 text for the BFCL-derived files |
| `MANIFEST.sha256` | SHA-256 of every file (except itself and `.gitignore`) |

## Naming and placeholders in the receipts

Receipts, graph JSONs and scripts label the paper's **shaped-lambda0.1** GRPO arm `SHAPED` (for
example the pair `base->SHAPED` and the key `shaped_features`). Cluster-specific values recorded in
receipts and job scripts are replaced by placeholders: `$WORK` (working directory on the cluster),
`$HOME`, `<HOST>` / `<NODE>` (compute-node names), `<SLURM_ACCOUNT>` and `<SLURM_PARTITION>`; job
scripts need those filled in before they can be submitted. `$WORK/grpo_project` is the separate
project in which the GRPO-family checkpoints and the original SFT input were produced.

## Third-party data and licenses

The two files under `data/` are modified derivatives of the Berkeley Function Calling Leaderboard
(BFCL) data, released by Gorilla LLM (UC Berkeley) under the Apache License 2.0: the release files
`BFCL_v4_simple_python.json`, `BFCL_v4_multiple.json`, `BFCL_v4_parallel.json` and
`BFCL_v4_parallel_multiple.json` with their `possible_answer/` counterparts, from the gorilla
repository (https://github.com/ShishirPatil/gorilla, `berkeley-function-call-leaderboard/bfcl_eval/data`;
also at https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard). Each
category was shuffled with seed 0 and split 90/10 into training rows (900 in total: 360/180/180/180)
and a held-out validation split (100 items: 40/20/20/20). The upstream license text is in
`LICENSES/BFCL-Apache-2.0.txt`; the upstream dataset page lists no NOTICE file. Modifications:

- `bfcl_sft_train.parquet`: 900 BFCL training rows converted to chat-completion format, with the
  first listed ground-truth call as the assistant turn (`code/training/prepare_sft_from_bfcl.py`).
- `bfcl_dpo_pairs.parquet`: 398 preference pairs built from those rows, with the gold call as
  `chosen` and a verifiably wrong greedy generation of the public base model Qwen3-1.7B as
  `rejected` (`code/training/build_dpo_pairs.py`).

```
@misc{berkeley-function-calling-leaderboard,
  title={Berkeley Function Calling Leaderboard},
  author={Fanjia Yan and Huanzhi Mao and Charlie Cheng-Jie Ji and Tianjun Zhang and Shishir G. Patil and Ion Stoica and Joseph E. Gonzalez},
  howpublished={\url{https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html}},
  year={2024},
}
```

`notes/sft_rewired_feature_interp.json` originally quoted short max-activating contexts from
`NeelNanda/pile-10k`, a sample of the Pile (Gao et al., 2020). The quoted text (`context` and
`token` fields) is removed here; feature ids, firing rates, activation values and prompt/position
indices remain, and `code/scripts/feature_interp.py` regenerates the contexts from the public
corpus. No model or SAE weights are included.

## Scope of reproducibility

The released graph JSONs make the headline statistics reproducible without a GPU. The graph
builder, trainers, job specifications, pre-registrations and small input parquets document and
permit rerunning the seeded LoRA arms when the named public base models, SAE weights, a compatible
CUDA environment and a Slurm cluster are supplied. The GRPO checkpoints, the large
model/checkpoint artifacts and the exact historical conda environments are not included, so this
repository does not claim bit-for-bit end-to-end training reproduction. Co-activation graphs
rebuild bit-for-bit on the GPU architecture that built them but not across architectures; the
paper's Table 1 reads graphs built on one H200. The audited original SFT checkpoint predates
pre-model-construction seeding; the four run-null training jobs are the reproducible seeded set.
The transcoder scale rows of the scale table come from the included sweep job scripts; no per-run
summary JSON was committed for them. The full lab notebook is not part of this release.

## License

Code and the result files authored for this paper: MIT (`LICENSE`). The BFCL-derived files in
`data/`: Apache License 2.0 (`LICENSES/BFCL-Apache-2.0.txt`).

## Contact

Xupeng Chen, xc1490@nyu.edu
