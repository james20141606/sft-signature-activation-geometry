"""Concrete `attribution_fn` implementations for Signature's build_graph.

The Signature paper's H1/H2 analysis needs a per-state attribution map
A[i,j] ≈ "how much does feature j at upstream layer L_src influence
feature i at downstream layer L_dst?". `attribution_graph.build_graph`
accepts any callable matching this signature; this module supplies the
canonical first-pass implementations.

Two methods are provided:

1. **`linear_probe_attribution`** (fast, no gradients) — for two SAEs at
   layers L_src < L_dst, treat the upstream feature activation `z_src` as
   inputs and the downstream `z_dst` as outputs; per-state A[i,j] =
   z_src[j] * |W_dst_decoder_row_i| * sign(z_dst[i]). Cheap correlational
   proxy; appropriate for an initial F1 distribution-shift figure.

2. **`attribution_patching`** (slower, gradient-based) — implements the
   first-order linear approximation (Geiger 2023 / Syed 2023): for each
   downstream feature i, the attribution of upstream feature j is
   `(∂z_dst[i] / ∂z_src[j]) * z_src[j]` where the gradient is taken at
   the current state. Requires both SAEs to have differentiable encode().

Both return dense `(d_sae_src, d_sae_dst)` arrays; build_graph applies
top-K per state before merging.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def linear_probe_attribution(
    state: dict,
    src_layer: int,
    dst_layer: int,
    *,
    sae_src,            # callable: hidden state at src_layer → (d_sae_src,)
    sae_dst,            # callable: hidden state at dst_layer → (d_sae_dst,)
    h_src: torch.Tensor,  # (d_in,) residual at src_layer for this state
    h_dst: torch.Tensor,  # (d_in,) residual at dst_layer for this state
) -> np.ndarray:
    """Cheap O(d_sae_src + d_sae_dst) attribution proxy.

    For each pair (i, j), A[j, i] = z_src[j] * z_dst[i]. This is the
    rank-1 outer product of the two SAE codes — the "co-activation" map.
    Useful as an entry point; downstream graph analysis can threshold this
    to find candidate edges before paying for proper gradient attribution.

    Returns array of shape (d_sae_src, d_sae_dst).

    NOTE: this dense path materialises a (d_sae_src × d_sae_dst) tensor.
    For large SAEs (d_sae ≥ 8192) prefer `linear_probe_attribution_sparse`
    which exploits SAE-code sparsity (typical L0 ≪ d_sae) to avoid the 4GB+
    outer-product allocation.
    """
    z_src = sae_src(h_src.unsqueeze(0)).squeeze(0)  # (d_sae_src,)
    z_dst = sae_dst(h_dst.unsqueeze(0)).squeeze(0)  # (d_sae_dst,)
    # outer product, on CPU as numpy for downstream sparse aggregation
    return torch.outer(z_src, z_dst).detach().float().cpu().numpy()


def linear_probe_attribution_sparse(
    state: dict,
    src_layer: int,
    dst_layer: int,
    *,
    sae_src,
    sae_dst,
    h_src: torch.Tensor,
    h_dst: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Sparse-aware variant of `linear_probe_attribution`.

    Exploits the fact that SAE codes are sparse (L0 ≪ d_sae). Instead of
    materialising the (d_sae_src × d_sae_dst) outer product, we compute it
    only over the non-zero indices of each code, yielding at most L0_src ×
    L0_dst non-zero entries (≤ 500 in practice for d_sae=32768, L0~22).

    Returns:
        rows  (E,)  int64 — src feature indices
        cols  (E,)  int64 — dst feature indices
        vals  (E,)  float64 — z_src[i] * z_dst[j] entries
        d_sae_src  int — full upstream SAE dim (for graph metadata)
        d_sae_dst  int — full downstream SAE dim
    """
    z_src = sae_src(h_src.unsqueeze(0)).squeeze(0)  # (d_sae_src,)
    z_dst = sae_dst(h_dst.unsqueeze(0)).squeeze(0)  # (d_sae_dst,)
    d_sae_src = int(z_src.shape[0])
    d_sae_dst = int(z_dst.shape[0])

    z_src_cpu = z_src.detach().float().cpu()
    z_dst_cpu = z_dst.detach().float().cpu()

    nz_src = z_src_cpu.nonzero(as_tuple=True)[0]   # (L0_src,)
    nz_dst = z_dst_cpu.nonzero(as_tuple=True)[0]   # (L0_dst,)
    if nz_src.numel() == 0 or nz_dst.numel() == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            d_sae_src, d_sae_dst,
        )

    v_src = z_src_cpu[nz_src]                       # (L0_src,)
    v_dst = z_dst_cpu[nz_dst]                       # (L0_dst,)
    # outer product on tiny dimensions
    outer = (v_src.unsqueeze(1) * v_dst.unsqueeze(0)).flatten().numpy()
    rows = nz_src.repeat_interleave(nz_dst.numel()).numpy().astype(np.int64)
    cols = nz_dst.repeat(nz_src.numel()).numpy().astype(np.int64)
    return rows, cols, outer.astype(np.float64), d_sae_src, d_sae_dst


def attribution_patching(
    state: dict,
    src_layer: int,
    dst_layer: int,
    *,
    sae_src,
    sae_dst,
    h_src: torch.Tensor,        # (d_in,) — must have requires_grad eligible path
    f_dst_to_resid: Callable[[torch.Tensor], torch.Tensor],
    target_idx_subset: np.ndarray | None = None,
) -> np.ndarray:
    """First-order Geiger/Syed attribution patching.

    For each target feature i in `sae_dst`'s output, compute
        ∂ z_dst[i] / ∂ z_src[j] * z_src[j]
    by running a small backward pass per i. The full d_sae_dst×d_sae_src
    grid is expensive — pass `target_idx_subset` (size k) to restrict to a
    candidate set produced by `linear_probe_attribution + top_movers`.

    `f_dst_to_resid(z_src)` is the path from `z_src` (decoded → residual at
    src_layer → forward through transformer to dst_layer's residual), which
    must be supplied by the caller since it depends on the model object.

    Returns (d_sae_src, d_sae_dst) — for indices not in target_idx_subset,
    columns are zero.
    """
    z_src = sae_src(h_src.unsqueeze(0)).squeeze(0).requires_grad_(True)
    h_dst_pred = f_dst_to_resid(z_src)
    z_dst = sae_dst(h_dst_pred.unsqueeze(0)).squeeze(0)

    d_sae_src = int(z_src.shape[-1])
    d_sae_dst = int(z_dst.shape[-1])
    out = np.zeros((d_sae_src, d_sae_dst), dtype=np.float32)
    targets = target_idx_subset if target_idx_subset is not None else np.arange(d_sae_dst)

    for i in targets:
        z_src.grad = None
        z_dst[int(i)].backward(retain_graph=(i != targets[-1]))
        if z_src.grad is None:
            continue
        a = (z_src.grad.detach() * z_src.detach()).cpu().numpy()
        out[:, int(i)] = a
    return out
