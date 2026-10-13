# PREREGISTRATION: the SFT calibrated estimator on a second backbone (Llama-3.1-8B)

Committed BEFORE any Llama SFT seed is trained. Git history is the timestamp.

## Why, and what exactly replicates

arXiv:2605.28860 independently reports the SFT>RL ordering without seed calibration; our
residual claim is the calibration. The second backbone therefore replicates the **calibrated
estimator** — LoRA-SFT under an identical recipe, four seeds, displacement read against the
family's own 6-pair seed band — not merely an ordering. (internal review round 4; run once, then stop:
no third backbone, no layer search, no seed extension.)

## Predeclared instrument (no selection anywhere)

- **Backbone**: meta-llama/Llama-3.1-8B (base). **SAEs**: fnlp Llama-Scope via sae-lens
  release `llama_scope_lxr_32x`.
- **Layer rule, fixed by formula**: fractional-depth transplant of the Qwen instrument
  (l14->l22 of 28) => src = round(14/28*32) = **16**, dst = round(22/28*32) = **25**.
  This coincides with the instrument already committed for the Tulu staircase
  (`run_sig_tulu_stair_v2norm.sbatch`), whose measured norm folds are reused verbatim:
  fold_src = 5.2784 (L16R), fold_dst = 2.0237 (L25R). No other layer pair will be run.
- **Instrument caveats carried over from the committed Llama chain**: metrics are
  encode-only (the sae-lens llama_scope decode path is known-broken, FVU ~4e4, and is not
  used); absolute L0/FVU are not comparable across SAE families; only within-family deltas
  are read.
- **Training recipe: byte-identical flags to the Qwen SFT seed null** (sft_warmup.py:
  epochs 1.0, lr 5e-5, batch 8 x accum 4, LoRA r16/alpha32, seeded via sft_seed.py), on the
  registered 900-row `data/bfcl_sft_train.parquet`. Only `--base-model` changes. h200-class
  nodes only, so memory never forces a recipe change.
- Graphs: base + 4 seeds, `m2_attribution_graph.py`, n=500 pile prompts, seq 64, same
  keep/top-k as every committed graph. Scoring: `circuit_rewiring.py`, family LSFT_s0..3,
  focus LSFT_s0 -> `notes/circuit_rewiring_llama_sft_seednull.json`.

## Registered readout and branches

Displacement d = 1 - edge_jaccard. Band = the 6 seed-pair displacements. Contrasts = the 4
base->LSFT_s* displacements. Verdict computed ONLY by `code/scripts/llama_repl_read.py`
(committed with this file, before any receipt; refuses partial input; prints the branch
verbatim and the candidate set it saw).

- **R1 — the calibrated estimator replicates**: every contrast exceeds the seed-pair
  maximum (min contrast > max seed-pair). The paper's SFT separation rule, on a second
  backbone family and a second SAE family.
- **R2 — partial**: some but not all contrasts exceed the seed-pair maximum. Reported as
  partial replication; no cherry-picking of the passing seeds.
- **R3 — no separation**: no contrast exceeds the seed-pair maximum. The SFT signature is
  then not backbone-general at this recipe and scale — reported as the finding, not buried.
- **R4 — uninformative**: fewer than 4 contrasts or 6 seed pairs in the receipt, or the
  receipt's own edge_jaccard precision_adequate flag is false. Attempted-and-uninformative.

No Llama-SFT quantity exists at the time of this registration:
`notes/circuit_rewiring_llama_sft_seednull.json` does not exist and no `lsft-seed*`
checkpoint directory exists on the cluster.

## AMENDMENT 1 (before any seed trained; JID 15241235 failed at the tokenizer load)

The first submission failed in 58 s at `AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")`:
the repo is gated (403) for this account, and the scratch HF cache has since been purged of both
the NousResearch mirror and the fnlp Llama-Scope SAEs that the committed Tulu chain used (the
official-path symlink the job created was dangling). No adapter, graph or receipt was produced.

Three changes, none touching the registered estimator:
1. **Model ID**: `NousResearch/Meta-Llama-3.1-8B` (ungated mirror) everywhere — training AND
   graphs. This is the same artifact the committed Tulu instrument actually loaded (its
   meta-llama path was a symlink into this mirror's cache).
2. **The graph step runs online on first use** so sae-lens can re-download
   `llama_scope_lxr_32x` (cache purged); later seeds hit the cache.
3. The job removes a dangling official-path symlink if present and creates none.

Layer rule, folds, recipe flags, seeds, branches, reader: unchanged.

## AMENDMENT 2 (before any seed trained; JID 15241385 failed at data formatting)

The training data is a chat-completion parquet; trl formats it with the tokenizer's chat
template. Qwen3-1.7B-Base ships its family's template in the base tokenizer, so the Qwen arm
never named this dependency; Llama-3.1-8B base ships none, and the run fails before step 1
(no adapter was written; the registration window is intact).

The recipe rule, made explicit: **format with the backbone family's canonical chat template.**
Implementation: `--chat-template-from NousResearch/Meta-Llama-3.1-8B-Instruct` (ungated
mirror of the canonical Llama-3.1 template; its header special tokens exist in the base
vocabulary). `sft_warmup.py` gains that optional flag, default None, so every committed Qwen
run is byte-identical. Nothing else changes.

Disclosed limit, fixed now: this makes "byte-identical recipe" precise as "identical flags
and formatting rule"; the template STRING necessarily differs by family, exactly as it
already does between the two families' instruct models.

## AMENDMENT 3 (before any seed trained; JID 15241489 OOM at step 0 of 29)

The registration asserted "h200-class nodes only, so memory never forces a recipe change".
That bet failed: at 8B, batch 8 x seq 2048 x a 128K vocabulary, the fp32 cross-entropy
allocation exceeds the H200's headroom (4.42 GiB short at step 0; no optimizer step ran, no
adapter was written; the registration window is intact). The log confirms the dose: 29
optimizer steps, exactly the Qwen arm's count.

**Change: --batch-size 4 --grad-accum 8** (was 8 x 4). The effective batch stays 32, the
optimizer-step count stays 29, and the change applies uniformly to all four seeds -- the
estimator is within-family (band from seed pairs, contrasts against the same band), so a
uniform accumulation split does not touch what is compared. Disclosed rather than hidden:
per-step batch COMPOSITION under trl's seeded shuffling differs from what bs=8 would have
produced; it differs identically for all four seeds.

Nothing else changes. This is the third and final permitted plumbing amendment class; a
fourth infrastructure failure escalates to R4 (attempted-and-uninformative) rather than
another patch.

## CLOSE-OUT: R4 by the Amendment-3 escalation clause (JID 15243747)

The fourth infrastructure failure occurred: seed-0 TRAINING now succeeds (29 steps, loss
1.474->1.234, adapter saved), but the graph step fails at transformer_lens name resolution --
`NousResearch/Meta-Llama-3.1-8B` is not an official TL model name. Root cause: AMENDMENT 1
changed the graph step's --backbone away from the committed Tulu instrument's proven
configuration (offline + cache symlink + `meta-llama/Llama-3.1-8B`, the TL-resolvable name),
which had used the symlink precisely because TL needs the official name while the weights
come from the ungated mirror.

Per Amendment 3's own clause -- "a fourth infrastructure failure escalates to R4 rather than
another patch" -- **this registration closes as R4: attempted-and-uninformative.** No gate
quantity was ever produced under it. A successor registration (v2, separate file) may re-run
the identical estimator with the graph step reverted to the committed instrument's exact
configuration; that is a new registration on the record, not a patch to this one.
