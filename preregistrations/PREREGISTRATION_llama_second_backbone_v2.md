# PREREGISTRATION v2: Llama second-backbone SFT replication (successor to the R4 close-out)

Committed BEFORE any Llama graph or gate quantity exists. Predecessor:
`PREREGISTRATION_llama_second_backbone.md`, closed as R4 by its own Amendment-3 escalation
clause after four infrastructure failures (gated repo; missing family template; OOM; a
transformer_lens name introduced by its own Amendment 1). Every failure was plumbing; no
estimator number was ever produced, so nothing here can be tuned toward a result.

**Everything is inherited unchanged from the predecessor and its amendments 1-3**: backbone
weights (NousResearch mirror), layer rule l16->l25 with folds 5.2784/2.0237, recipe (bs 4 x
accum 8, effective 32, 29 steps, lr 5e-5, LoRA r16/a32, family-canonical chat template),
seeds 0-3, branches R1/R2/R3/R4, and the committed reader `code/scripts/llama_repl_read.py`
(byte-unchanged).

**One change -- the graph step reverts to the committed Tulu instrument's exact
configuration** (`run_sig_tulu_stair_v2norm.sbatch`, receipts in the paper): TL backbone name
`meta-llama/Llama-3.1-8B`, weights via the cache symlink onto the (now present) mirror
-- created only after verifying the target exists -- and `HF_HUB_OFFLINE=1` for graphs. The
two SAEs are warm-cached online (CPU, no measurement) before the offline graph loop.

**Inherited state, disclosed**: the seed-0 adapter exists (trained under the final inherited
recipe after Amendment 3; the training arm of the predecessor succeeded). Seeds 1-3 untrained;
0 graphs; 0 gate quantities.

**Escalation clause carried over**: one further infrastructure failure closes THIS
registration as R4 with no successor.
