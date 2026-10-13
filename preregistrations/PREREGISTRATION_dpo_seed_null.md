# PREREGISTRATION: a per-algorithm seed null for DPO (closes L8)

Committed BEFORE any DPO seed is trained. Git history is the timestamp.

## Why

L8 is the paper's stated asymmetry: SFT has a 4-run per-algorithm null and GRPO a
6-run one, but **DPO has neither**. Its Tulu-3 stage increment (0.796) is scored
against the RLVR run-pair null (0.903), which is not DPO's own — and rather than
disclose that while keeping DPO in the title, we removed DPO from the title.

The AC's writing-only route to borderline is already done. This is the
experimental route: give DPO the same treatment SFT got, and let the result decide
whether DPO returns to the title.

This is exactly the standard the paper demands of others: a base→post difference
means nothing until it is scored against the run-to-run variability of the
algorithm that produced it.

## Design (mirrors the SFT seed null exactly; that is the point)

- **Recipe**: LoRA-DPO on Qwen3-1.7B, r=16 alpha=32 on q/k/v/o_proj, 1 epoch,
  bf16 — matched to the audited SFT recipe wherever DPO permits, with beta=0.1
  (trl default) and lr 5e-6 (10x below SFT's 5e-5, the standard DPO/SFT ratio;
  fixed here, not tuned).
- **Seeds**: 0, 1, 2, 3 through `code/scripts/sft_seed.py` (a generic runpy
  delegate that calls `set_seed` BEFORE the model is built, so LoRA init is
  seeded — the same wrapper the SFT null used, unchanged).
- **Null**: all 6 pairs among the 4 runs, identical to the SFT construction.
- **Contrast**: base→each DPO run, on the same pile-arm attribution graph
  (`m2_attribution_graph.py`, qwen-scope l14→22, n=500, seq 64, keep-top-n 10000),
  scored by `circuit_rewiring.py --seed-family ... --focus DPO`.
  Every knob above is copied from `run_sig_sft_seed_null.sbatch`, unchanged.

### Preference data — constructed from receipts, not judgement

The BFCL SFT parquet (900 rows) carries a **programmatically checkable** gold
function call per row. Preference pairs are therefore built without any judge:

- **chosen** = the gold assistant turn (the same string SFT trained on).
- **rejected** = a base-model greedy generation for that prompt that is
  **verifiably wrong** — string-inequality to gold after normalisation. Rows where
  the base model produces the gold call are DROPPED (no pair exists), and the
  dropped count is reported.

This makes the preference signal a fact about the base model's errors, not about
a preference model we would then have to validate. Pair construction is seeded
(greedy decoding, temperature 0) and runs ONCE, before any seed is trained, so all
four seeds see the identical preference set — the null must isolate seed variance,
not data variance.

**Registered floor**: if fewer than 200 usable pairs survive, the null is
underpowered relative to SFT's 900-row recipe; we report the count and declare
the arm inconclusive rather than training on a thin set.

## Interpretation branches (written before data; each with its evidential basis)

Basis for all branches: 6 null pairs over 4 specified runs, 5 base→DPO contrasts
(4 seeds + the public Tulu-3 DPO stage where comparable), same graph and statistic
as the SFT arm.

- **D1. DPO separates from its own seed null** (all base→DPO contrasts outside the
  6-pair band, as SFT's five contrasts were) → DPO returns to the title alongside
  SFT, and L8 closes as *resolved by measurement*. The claim is scoped to
  LoRA-DPO at 1.7B on BFCL-derived preferences, NOT to the public Tulu-3 DPO
  stage, which remains scored against the run-pair null it has.
- **D2. DPO does not separate** → a *calibrated* DPO null result: the preference
  stage's footprint is within its own run noise on this recipe. That is a real
  finding — it would make the paper's stage-resolved picture SFT-only among
  offline methods, and it is reportable with the same standing as the GRPO null.
  L8 closes as resolved; DPO stays out of the title for a measured reason rather
  than an absent measurement.
- **D3. The band is degenerate** (seed pairs indistinguishable from base→DPO in
  BOTH directions, or the DPO runs barely move from base at all — e.g. mean
  |Δ| below the graph statistic's own precision) → the recipe did not train;
  report as an instrument/recipe failure, no claim about DPO. A positive control
  applies: the four DPO runs must differ from base by more than the SFT seed
  band's width, or the arm is declared unable to resolve anything.
- **D4. Fewer than 200 usable preference pairs** → inconclusive, per the floor
  above; no branch is read.

The verdict is computed by `code/scripts/circuit_rewiring.py` on committed graphs
and reported verbatim. No threshold below is adjustable after seeing a number:
"outside the band" means outside the 6-pair min/max, the same rule the SFT arm used.

## What this is NOT

- Not a claim about production-scale DPO (Tulu-3's stage stays as it is).
- Not a new statistic: same graph, same edge cutoff, same rewiring measure.
- Not a re-opening of the SFT result, which stands unchanged either way.

## Cost

SFT seeds took ~40 s each; DPO is ~2-3x that per step with the reference model,
so ~4 short training runs plus 4 attribution graphs — one GPU job, comparable to
`run_sig_sft_seed_null.sbatch`.

## AMENDMENT 1 (before any DPO seed was trained; no null number exists)

Pair construction ran (JID 15094123) and returned 900 usable pairs, 0 dropped.
Inspecting them — which the registration required in spirit but did not require in
code — showed the set was **invalid for the registered purpose**:

```
rejected empty: 0
rejected mean chars: 400   chosen mean chars: 76
rejected that parse as a function call: 0 / 900
same function name, different args (near-miss): 0
CHOSEN : "recipe_search(diet=['Gluten Free'], time_limit=30, dish='cookie')"
REJECT : "<think>\nOkay, let's see. The user wants a gluten-free cookie recipe..."
```

Every rejected string is a Qwen3 `<think>` monologue, so the preference contrast
was **format** (concise call vs reasoning block), not **correctness**. The
registration's literal condition (string-inequality to gold) was satisfied; its
purpose was not. A DPO null built on this would measure what suppressing the
think-block does to activation geometry — a different experiment with a valid
timestamp on it.

Fixed at the source, not by post-filtering alone:
- generations now use `apply_chat_template(..., enable_thinking=False)`, so the
  base model emits a bare call and a wrong call is a wrong *call*;
- any generation that still does not parse as `name(` is dropped and counted
  separately as a format contrast rather than admitted;
- `</think>` residue is stripped before comparison.

The registered floor (<200 usable pairs → arm inconclusive) now does real work: if
the base model is usually *correct* once thinking is off, the pair set may fall
below it, and that is a reportable outcome rather than something to engineer around.

Evidence no outcome influenced this: no DPO seed has been trained, no attribution
graph for any DPO run exists, and `notes/circuit_rewiring_dpo_seednull.json` is
absent. Only pair construction has run.
