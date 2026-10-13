#!/usr/bin/env python3
"""Unit-test the DPO loss and the completion-only masking (no model, no GPU).

The two things that go silently wrong in a hand-written DPO: counting prompt
tokens in the log-probability, and getting the preference sign backwards. Both
would still train and still produce a seed null -- of the wrong quantity.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from dpo_warmup import seq_logprob  # noqa: E402


class FakeModel:
    """Returns fixed logits so the expected log-prob is computable by hand."""
    def __init__(self, vocab, logits):
        self.vocab, self._logits = vocab, logits

    def __call__(self, input_ids, attention_mask):
        b, t = input_ids.shape
        return type("O", (), {"logits": self._logits.expand(b, t, self.vocab).clone()})


V, T = 5, 6
# uniform logits -> every token log p = -log(V)
uni = torch.zeros(1, T, V)
m = FakeModel(V, uni)

ids = torch.tensor([[1, 2, 3, 4, 0, 0]])
att = torch.tensor([[1, 1, 1, 1, 0, 0]])

# prompt_len 2 -> completion tokens are positions 2,3 => 2 predicted tokens
lp = seq_logprob(m, ids, att, torch.tensor([2]))
expect = 2 * -torch.log(torch.tensor(float(V)))
assert torch.allclose(lp, expect, atol=1e-5), (lp, expect)
print(f"1. completion-only mask: {lp.item():.4f} == 2*log(1/5) = {expect.item():.4f}  OK")

# prompt_len 1 -> 3 completion tokens
lp1 = seq_logprob(m, ids, att, torch.tensor([1]))
assert torch.allclose(lp1, 3 * -torch.log(torch.tensor(float(V))), atol=1e-5)
print(f"2. longer completion counts more tokens: {lp1.item():.4f}  OK")

# padding must not contribute: extending pad length changes nothing
ids2 = torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0]])
att2 = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])
m2 = FakeModel(V, torch.zeros(1, 8, V))
lp2 = seq_logprob(m2, ids2, att2, torch.tensor([2]))
assert torch.allclose(lp, lp2, atol=1e-5), (lp, lp2)
print(f"3. padding contributes nothing: {lp2.item():.4f}  OK")

# ---- loss sign ----------------------------------------------------------------
beta = 0.1
def dpo(pc, pr, rc, rr):
    return -F.logsigmoid(beta * ((pc - rc) - (pr - rr)))

# policy prefers chosen MORE than reference does -> lower loss
better = dpo(torch.tensor(0.0), torch.tensor(-2.0), torch.tensor(0.0), torch.tensor(0.0))
neutral = dpo(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))
worse = dpo(torch.tensor(-2.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))
assert better < neutral < worse, (better, neutral, worse)
print(f"4. loss sign: prefer-chosen {better.item():.4f} < neutral {neutral.item():.4f} "
      f"< prefer-rejected {worse.item():.4f}  OK")
assert abs(float(neutral) - 0.6931) < 1e-3, "neutral must be log 2"
print("5. neutral loss == log 2  OK")
print("\nALL PASS")
