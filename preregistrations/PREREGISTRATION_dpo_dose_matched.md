# PREREGISTRATION: a DOSE-MATCHED DPO seed null (closes L8(a))

Committed BEFORE any dose-matched DPO seed is trained. Git history is the timestamp.
Companion to `PREREGISTRATION_dpo_seed_null.md`, which this does not reopen: the
12-step arm and its D2 verdict stand as published.

## Why

The first DPO null (JID 15102095, receipt `notes/circuit_rewiring_dpo_seednull.json`)
returned branch D2: base->DPO does not separate from DPO's own 6-pair seed band
(z = -1.02). Writing that up exposed a confound we then disclosed as L8(a) rather
than fixed:

> the DPO runs are **12 optimizer steps** on 398 pairs at lr 5e-6, against the SFT
> arm's **29 steps** on 900 rows at lr 5e-5. So "DPO writes less footprint than SFT"
> is not separable from "this DPO run was shorter."

A disclosed confound is weaker than a measured one. This registration fixes the
design that removes it. The paper's whole thesis is that a difference means nothing
until it is calibrated; a *non*-difference means nothing until the arm is shown to
have had the dose to produce one.

## The design decision, made before the run: step-matching, inside a ladder

Two readings of "dose-matched" were available and they are not equivalent.

**(A) Match optimizer steps.** Train DPO for 29 optimizer steps -- SFT's exact
count. The matched quantity is *number of parameter updates*.

**(B) Match displacement.** Train DPO until base->DPO displacement (1 - edge
Jaccard) reaches SFT's 0.0628, then ask whether *that* displacement escapes the
seed band recomputed at that dose. This is the sharper test of the question one
actually cares about -- *is DPO's footprint qualitatively different from SFT's, or
merely smaller?* -- because it removes size from the comparison entirely.

**We register (A) as PRIMARY, embedded in a three-rung dose ladder, and we report
(B) descriptively off the same runs.** The reason is not that (B) is the worse
question; it is that (B) is an **adaptive stopping rule**. To match displacement you
must build a graph, look at the number, and decide whether to keep training -- a
stopping decision made *after* seeing the outcome, on the same statistic that
carries the verdict. Pre-registering (B) honestly forces it into "fix a grid of
doses in advance, then report the rung whose displacement is closest to SFT's,"
which is precisely the ladder below. So the ladder subsumes (B) without the forking
path, and (A) falls out of it as one rung.

**Registered doses: 12 / 29 / 60 optimizer steps.**
- **12** -- the already-published arm. Its four adapters exist; it is NOT retrained.
- **29** -- SFT's optimizer-step count. **The primary comparison.**
- **60** -- ~2x SFT's dose, and the extrapolation rung. It exists so that a null at
  29 cannot be dismissed as "still too short," and so that if displacement is still
  climbing at 29 we can say whether it plateaus or keeps going.

**What (B) would have shown, stated now so it cannot be reframed later.** If some
rung reaches SFT's displacement of 0.0628 and the contrast at that rung still sits
inside its own dose-matched band, that is the strongest available form of the claim:
DPO moves the graph as far as SFT does and *still* cannot be told apart from its own
run-to-run noise, i.e. the difference between the stages is not one of size. If no
rung reaches 0.0628, displacement-matching is simply not achievable on this recipe
within this budget, and we will say that rather than extrapolate a curve to it.

## Protocol (every knob copied from the 12-step arm unless named here)

- **Recipe**: unchanged. LoRA r=16 alpha=32 on q/k/v/o_proj, lr 5e-6, beta 0.1,
  bs 4 x accum 8 (32 pairs per optimizer step), bf16, max_len 1024, on the same
  committed `data/bfcl_dpo_pairs.parquet` (398 pairs, the AMENDMENT 1 rebuild).
- **Seeds**: 0, 1, 2, 3, through `code/scripts/sft_seed.py` exactly as before, so
  `set_seed` runs before the model is built and LoRA init is seeded.
- **All four seeds get the identical treatment at every rung.** Each seed trains a
  single 60-step trajectory with adapters saved at step 29 and step 60. The null at
  each rung is therefore a null over four *specified* runs, as at 12 steps.
- **Data order**: `dpo_warmup.py` uses a fixed unshuffled index order and, past one
  epoch, repeats it. There is **no per-epoch reshuffling** -- registered here as a
  known property, not discovered later. It is the same construction the 12-step runs
  used; it makes the ladder a single deterministic trajectory per seed.
- **Graphs**: `m2_attribution_graph.py`, qwen-scope l14->22, n=500, seq 64,
  keep-top-n 10000 -- byte-identical arguments to the 12-step arm.
- **Scoring**: `circuit_rewiring.py --seed-family <the four runs at that rung>
  --focus <seed 0 at that rung>`, i.e. the same statistic (edge Jaccard), the same
  6-pair null construction, and the same scored run as before.

## Registered checks that run before any branch is read

1. **D3-style positive control, at every rung.** The four DPO runs at a rung must
   displace the graph from base by more than the SFT seed band's width, or that rung
   is declared unable to resolve and no branch is read from it. As in the 12-step
   arm we use the conservative reading of "width": the 95% interval width
   3.92 x 0.001367 = 0.00536. (The 12-step rung clears this at 6.1-6.6x.)
2. **Trajectory-continuity check.** Steps 1-12 of each new 60-step run must
   reproduce the committed 12-step run's per-step losses for that seed (same seed,
   same fixed data order, same lr, same optimizer). *Tolerance, fixed here because it
   depends on hardware we do not control:* the published runs ran on an **A100**
   (JID 15102095, `<SLURM_PARTITION>`). bf16 reduction order is GPU-model dependent, so we
   require agreement to **4 decimals if this job also lands on an A100**, and to **3
   decimals if it lands on another GPU model**, with the realised GPU recorded either
   way. If the check fails at the applicable tolerance, the ladder is NOT an
   extension of the published arm and we report the rungs as three separate arms
   rather than as a dose-response curve (branch M5).
3. **Monotonicity sanity check.** Displacement is expected to grow with dose. If it
   does not -- if base->DPO displacement at 29 or 60 is not above the 12-step value
   -- the recipe has plateaued or something is wrong, and we say so explicitly
   instead of reading a null off a run that stopped moving.
4. **Precision adequacy.** The scored statistic's null SD must exceed the 1e-4
   reporting step at each rung, exactly as the paper's precision rule requires. The
   weighted edge cosine is expected to fail this again and, as before, carries
   nothing.

## Interpretation branches (written before data; each with its evidential basis)

Basis for all branches: 6 null pairs over 4 specified runs at each rung, 4 base->DPO
contrasts per rung, same graph and statistic as the SFT and 12-step DPO arms.

- **M1. The matched-dose (29) contrast SEPARATES from its own dose-29 band.**
  Then the original D2 was a **dose artefact**, and L8(a) -- which we wrote as a
  limitation -- was the real story. Consequences, accepted in advance: the
  "three-for-three on per-algorithm nulls" framing is withdrawn; the abstract's DPO
  clause is withdrawn; DPO is reported as *detected once dose-matched*, and the
  question of whether it belongs in the title is reopened on the evidence. We would
  rather publish this than the version we currently have.
- **M2. The matched-dose (29) contrast still lands INSIDE its own dose-29 band.**
  The null survives dose-matching. The claim strengthens from "not resolved at this
  dose" to "not resolved at two doses, one of them matched to SFT's, and (if the
  60-step rung agrees) across a 5x dose range." L8(a) is downgraded from an open
  confound to a bounded one. This does **not** become "preference learning leaves no
  trace": it remains one LoRA recipe at 1.7B on one preference set, and it still
  does not transfer to the public Tulu-3 DPO stage.
- **M3. The rungs disagree** (e.g. inside at 29, outside at 60, or vice versa).
  Report the curve, not a verdict: the dose at which a footprint becomes resolvable
  is then the finding, and the headline becomes dose-dependent rather than
  algorithm-dependent. No single z is promoted.
- **M4. The band degenerates or training diverges at a higher dose** (loss not
  finite, or the four runs collapse onto each other / onto base such that check 1 or
  3 fails). Instrument/recipe failure at that rung; that rung is reported as failed
  and contributes no claim. The 12-step arm is unaffected.
- **M5. The trajectory-continuity check (2) fails.** The rungs are reported as three
  independent arms at three doses rather than as a dose-response curve, and the
  discrepancy is reported with the loss traces.

No threshold here is adjustable after seeing a number. "Inside the band" means inside
the 6-pair min/max at that rung, and the z is quoted against that rung's own mean and
SD -- the same rule the SFT, GRPO and 12-step DPO arms used.

## What this is NOT

- Not a reopening of the 12-step D2 result, which stands as published either way.
- Not a claim about the public Tulu-3 full-parameter DPO stage, at any rung. That
  stage keeps the run-pair null it has; substituting a 1.7B LoRA null for it would be
  the borrowed-null error this paper exists to criticize.
- Not a new statistic, graph protocol, or edge cutoff.
- Not a matched *learning rate*. lr stays at 5e-6 (10x below SFT's, the standard
  DPO/SFT ratio, fixed in the original registration and not tuned). Step count is the
  matched quantity; lr remains a disclosed difference between the arms and will stay
  disclosed in L8.

## Cost

4 seeds x 60 optimizer steps. The 12-step runs measured ~3 min/seed, so ~15 min/seed
plus 8 attribution graphs (~1 min each) -- roughly 70 minutes in one GPU job, against
this paper's ~35 GPU-hour total budget.

## OUTCOME (appended after the run; the design above is unchanged)

Run: JID 15103738, H100 (`<NODE>`), 2026-08-01T01:05:11 -> 01:28:28. Receipts
`notes/circuit_rewiring_dpo_dose{29,60}.json`. **Registration margin, stated rather
than rounded up: this file was committed at 01:05:07 and the job started at 01:05:10
-- three seconds.** No dose-matched adapter, graph or receipt existed at commit time,
so the substantive guarantee holds, but the margin is thin and we say so.

Checks, in the registered order:
1. **D3 positive control -- PASS at every rung** (6.0-6.8x the SFT band's 95% width).
2. **Trajectory continuity -- FAIL** -> branch **M5**. Diagnosed, not assumed (probe
   JID 15109030): two runs of seed 0 on *one node*, under the old and the new code
   path, on provably identical batches, agree for five optimizer steps and then
   diverge; so does a fresh run of the *unmodified* path against the published one on
   another node of the same GPU model. The cause is therefore neither the `--max-steps`
   edit nor the A100->H100 move but **non-bitwise-reproducible training**. The check
   assumed a determinism this system does not have -- a defect in this registration,
   not a finding about DPO. Rungs are reported as three independent arms, per M5.
3. **Monotonicity -- FAIL (displacement does not grow with dose)**, which the
   registration required be stated outright. 4-run mean displacement 0.0346 / 0.0343 /
   0.0349 at 12 / 29 / 60; all twelve contrasts within 0.0323-0.0365.
4. **Precision adequacy -- PASS** for edge Jaccard at every rung (null SD 22 / 12 / 12
   reporting steps); weighted cosine fails at every rung and carries nothing.

Branch: under the registered primary rule (scored run + the receipt's verdict field)
**M2** holds at every rung -- no rung separates. Under the rule the paper states in
prose for the SFT arm (every contrast outside the band, every seed pair inside) the
rungs **disagree** -- 0/4, 2/4, 4/4 contrasts outside at doses 12/29/60 -- which is
**M3**. Both readings have support, so we report both and take M3's prescription
(report the curve, promote no single z) as the more conservative.

The headline the ladder actually buys is the one that needs no band: at the identical
29 optimizer steps the four SFT and four DPO runs occupy **disjoint** displacement
ranges (0.0588-0.0630 vs 0.0323-0.0365; SFT's mean 1.8x DPO's), and dose does not
close the gap.

Two things found while analysing that cut against over-reading the dose-60 rung, both
now disclosed in the paper: the DPO seed band **rises and tightens with dose** (mean
0.96705 -> 0.96772 -> 0.96917, SD 0.0022 -> 0.0012 -> 0.0012) as the seeds converge,
so the dose-60 turn is driven by the denominator while displacement stands still; and
the DPO null holds **data order fixed** across seeds whereas the SFT null (trl
`SFTTrainer`, seeded sampler) does not, so the DPO band omits a variance component the
SFT band contains -- anti-conservative for detecting DPO. A matched-construction DPO
null with per-run reshuffling is the fix and is not run here.

## Evidence no outcome influenced this document

At the time of committing: no dose-matched DPO seed has been trained, no adapter
exists at step 29 or 60, no attribution graph exists for any dose-matched run, and
`notes/circuit_rewiring_dpo_dose_matched.json` is absent. Only the 12-step arm exists,
and its receipt is already committed and unchanged.
