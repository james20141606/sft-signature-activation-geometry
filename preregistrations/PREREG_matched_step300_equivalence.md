# Pre-registration — matched same-recipe, same-step (@300) equivalence analysis

**Written 2026-07-28, BEFORE seed s3 reaches step 300 and before any four-run null is computed.**
Committed in this state deliberately: an equivalence margin chosen after seeing the interval is
worthless, and this paper's entire thesis is that post-hoc thresholds manufacture findings. If this
file is edited after the run, the git history must show it.

## Why this run exists

The current matched-step null uses **three** same-recipe, same-step pairs from three runs
(s1/s4/s5 @ step 300), and all three base→post contrasts share one base checkpoint. The paper
therefore says, correctly, that this is "a descriptive consistency, not a test of equivalence"
and defines no margin. The AC's decisive objection is that a categorical RL-negative headline is
not licensed by an analysis that disclaims equivalence. Completing s3 to step 300 gives a
**four-run, six-pair** same-recipe same-step null and makes a real equivalence statement possible.

## Fixed in advance

**Statistic.** Edge-set Jaccard between 10,000-edge attribution graphs (qwen-scope l14→22,
n=500 pile prompts). Chosen because it is the paper's declared precision-adequate statistic: its
null SD is ~16x the 1e-4 reporting step, whereas the weighted-cosine null is quantized to three
distinct values and cannot carry a z (Sec. 4.3). No statistic switching after the fact.

**Comparison.** Mean of the four base→post@300 contrasts (base→s1, s3, s4, s5) against the mean of
the six seed–seed@300 pairs.

**EQUIVALENCE MARGIN — Δ = 0.0032 in edge Jaccard.**
Fixed now, and derived from an independently pre-existing quantity rather than from this run's data:
it is the audit's own published 2σ minimum detectable effect on the pile arm (2 × 0.0016, Sec. 4.3),
i.e. the smallest base→post shift the instrument was already claimed to flag at |z|≥2. It corresponds
to ~32 of 10,000 edges rewired beyond run noise. Practical meaning: an effect smaller than Δ is one
this instrument was never able to resolve, so declaring equivalence within Δ claims exactly the
sensitivity the paper already claims elsewhere, and no more.
For scale, SFT's actual effect is 0.0254 — **7.9 × Δ** — so the margin is far below the one effect
the audit is known to detect. Δ is **not** to be renegotiated after seeing the interval.

**Dependence-aware interval.** The six pairs share four underlying runs and the four base contrasts
share one base checkpoint, so the observations are not independent and a naive t-interval understates
the variance. We will report a **cluster/leave-one-run-out bootstrap**: resample *runs* (not pairs)
with replacement, recompute both means and their difference, and take the 90% percentile interval
(two one-sided tests at 5%, the standard TOST convention). We will report the leave-one-run-out
spread alongside it, and we will state n_runs = 4 as the true resolution limit.

## Both branches, committed now

**Branch A — the interval falls entirely inside ±Δ.** Then the base→RL change is statistically
equivalent to seed-to-seed run variation at the audit's own resolution. We may then say the RL
contrast is *equivalent to run variation within a pre-registered margin of 0.0032*, replacing the
current descriptive-consistency language. We still may **not** say "RL leaves no signature": Δ is a
resolution limit, not zero, and the claim stays scoped to this regime (≤300 steps, accuracy reward,
1.7B, general-text probe).

**Branch B — the interval is not contained in ±Δ (either it straddles a bound or lies outside).**
Then equivalence is **not** established. The paper will say so in the abstract, keep the RL result as
an explicit failure to detect with the interval quoted, and drop any wording that implies the RL
contrast has been shown equal to run noise. If the interval lies *below* −Δ, i.e. base→RL is
detectably more rewired than seed pairs, then **the RL-negative claim is wrong and must be removed
from the title, abstract and conclusion**, and the paper reports a detected RL effect instead.

**Branch C — s3@300 does not complete.** Report the three-pair result exactly as it stands today, with
no equivalence claim, and record the shortfall. No partial substitution of a different seed or step.

## What will NOT be done

- No swapping edge Jaccard for another statistic if the result is unwelcome.
- No widening Δ, and no adding runs from a different recipe or step to shrink the interval.
- No reporting a one-sided interval where a two-sided one was pre-registered.
- No dropping s1 (the known-marginal seed) from the four-run null.

## Execution

```bash
sbatch --export=ALL,SEED=3 code/scripts/slurm/run_grpo_resume_seed_step300.sbatch   # to step 300
sbatch --export=ALL,SEED=s3 code/scripts/slurm/run_sig_basediff_s1s3_step300.sbatch # 4th graph
python3 code/scripts/circuit_rewiring.py --graphs base=... GRPOs1=... GRPOs3=... \
    GRPOs4=... GRPOs5=... --focus GRPOs3 --out notes/circuit_rewiring_6pair_step300.json
python3 code/scripts/equivalence_matched_step300.py     # TOST + cluster bootstrap vs Δ=0.0032
```
