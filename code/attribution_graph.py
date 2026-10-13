"""H2: attribution graph diff — pre-RL vs post-RL.

Build a feature-to-feature attribution graph over a set of probe trajectories,
for each of the base and post-RL models. Compare graphs via:

    - Jaccard overlap of edge sets (binary, threshold-gated)
    - Weighted edge overlap (cosine of vectorised edge weights)
    - Set of "post-RL-only" edges — candidates for novel circuitry

A post-RL-only edge is a *positive* H2 candidate only if its causal necessity
is confirmed via attribution patching — see §method-h2.

This module deliberately does NOT prescribe the attribution method;
attribution patching, integrated gradients, and direct SAE-feature linear
probing are all valid. The attribution function is passed in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class AttributionGraph:
    """Sparse adjacency over SAE features.

    Edge weight A[i, j] = attributed influence of feature j (at upstream
    layer L_src) on feature i (at downstream layer L_dst), averaged over
    probe trajectories. Sparse representation: indices + values."""

    src_layer: int
    dst_layer: int
    edges_src: np.ndarray   # [E]
    edges_dst: np.ndarray   # [E]
    weights: np.ndarray     # [E]
    d_sae_src: int
    d_sae_dst: int


def jaccard_overlap(g1: AttributionGraph, g2: AttributionGraph, threshold: float) -> float:
    """Binary Jaccard overlap on edges with |weight| > threshold."""
    e1 = {(int(s), int(d)) for s, d, w in zip(g1.edges_src, g1.edges_dst, g1.weights) if abs(w) > threshold}
    e2 = {(int(s), int(d)) for s, d, w in zip(g2.edges_src, g2.edges_dst, g2.weights) if abs(w) > threshold}
    if not e1 and not e2:
        return 1.0
    return len(e1 & e2) / len(e1 | e2)


def weighted_overlap(g1: AttributionGraph, g2: AttributionGraph) -> float:
    """Cosine similarity over the union of edge keys, treating absent edges
    as zero weight. Range [-1, 1]; expect near-1 for similar graphs."""
    edges = (
        {(int(s), int(d)): float(w) for s, d, w in zip(g1.edges_src, g1.edges_dst, g1.weights)},
        {(int(s), int(d)): float(w) for s, d, w in zip(g2.edges_src, g2.edges_dst, g2.weights)},
    )
    keys = set(edges[0]) | set(edges[1])
    v1 = np.array([edges[0].get(k, 0.0) for k in keys])
    v2 = np.array([edges[1].get(k, 0.0) for k in keys])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float((v1 @ v2) / (n1 * n2))


def post_only_edges(
    pre_graph: AttributionGraph,
    post_graph: AttributionGraph,
    pre_threshold: float,
    post_threshold: float,
) -> list[tuple[int, int, float]]:
    """Edges that exist in post but not in pre at the configured thresholds.
    These are the H2 candidates whose causal necessity must then be tested
    via an ablation in `causal_ablation.py`."""
    pre_set = {
        (int(s), int(d))
        for s, d, w in zip(pre_graph.edges_src, pre_graph.edges_dst, pre_graph.weights)
        if abs(w) > pre_threshold
    }
    out: list[tuple[int, int, float]] = []
    for s, d, w in zip(post_graph.edges_src, post_graph.edges_dst, post_graph.weights):
        if abs(w) > post_threshold and (int(s), int(d)) not in pre_set:
            out.append((int(s), int(d), float(w)))
    return out


def build_graph(
    states_iter,                                  # iter of states (see data_loader)
    attribution_fn: Callable[..., np.ndarray],    # returns [d_sae_src, d_sae_dst]
    src_layer: int,
    dst_layer: int,
    keep_top_n: int = 10_000,
    per_state_top_k: int = 5_000,
) -> AttributionGraph:
    """Aggregate per-state attribution matrices into a sparse graph.

    `attribution_fn(state, src_layer, dst_layer)` is the user-supplied
    attribution method (attribution patching, integrated gradients, …).
    We sum over states and keep the top-``keep_top_n`` edges by |weight|.

    Memory strategy: each per-state attribution is filtered to its own top
    ``per_state_top_k`` edges (by |weight|) before being merged into the
    cumulative sparse accumulator. This keeps memory O(per_state_top_k) per
    state regardless of full d_sae, at the cost of dropping edges that are
    individually small but jointly significant. ``per_state_top_k`` should be
    set comfortably above the eventual ``keep_top_n``.

    The returned ``AttributionGraph`` carries:
        - ``edges_src`` / ``edges_dst`` / ``weights`` (sorted by |weight|, desc)
        - ``d_sae_src`` / ``d_sae_dst`` (inferred from the first attribution_fn call)
    """
    # cumulative weight per (src, dst); a Python dict is fine at <1e6 keys.
    acc: dict[tuple[int, int], float] = {}
    n_states = 0
    d_sae_src: int | None = None
    d_sae_dst: int | None = None

    for state in states_iter:
        a = attribution_fn(state, src_layer, dst_layer)
        a = np.asarray(a)
        if a.ndim != 2:
            raise ValueError(f"attribution_fn must return 2-D array, got {a.shape}")
        if d_sae_src is None:
            d_sae_src, d_sae_dst = int(a.shape[0]), int(a.shape[1])
        elif a.shape != (d_sae_src, d_sae_dst):
            raise ValueError(
                f"attribution_fn shape changed: expected {(d_sae_src, d_sae_dst)} got {a.shape}"
            )

        # Top-k by |weight|, flat indexing
        flat = a.reshape(-1)
        if flat.size == 0:
            n_states += 1
            continue
        k = min(per_state_top_k, flat.size)
        # argpartition gives unsorted top-k; we don't need them sorted yet.
        idx = np.argpartition(np.abs(flat), -k)[-k:]
        # Map flat index → (row, col)
        rows = idx // d_sae_dst
        cols = idx % d_sae_dst
        vals = flat[idx]
        for r, c, v in zip(rows.tolist(), cols.tolist(), vals.tolist()):
            key = (int(r), int(c))
            acc[key] = acc.get(key, 0.0) + float(v)
        n_states += 1

    if not acc:
        # Empty stream → return empty graph with shape inferred or zero.
        d_sae_src = d_sae_src or 0
        d_sae_dst = d_sae_dst or 0
        return AttributionGraph(
            src_layer=src_layer, dst_layer=dst_layer,
            edges_src=np.empty(0, dtype=np.int64),
            edges_dst=np.empty(0, dtype=np.int64),
            weights=np.empty(0, dtype=np.float64),
            d_sae_src=d_sae_src, d_sae_dst=d_sae_dst,
        )

    # Mean over states (so weights are comparable across runs of different sizes).
    keys = list(acc.keys())
    weights = np.array([acc[k] / max(n_states, 1) for k in keys], dtype=np.float64)
    src = np.array([k[0] for k in keys], dtype=np.int64)
    dst = np.array([k[1] for k in keys], dtype=np.int64)

    # Keep top keep_top_n by |weight|, desc.
    order = np.argsort(-np.abs(weights))
    if keep_top_n is not None and keep_top_n > 0:
        order = order[:keep_top_n]
    return AttributionGraph(
        src_layer=src_layer, dst_layer=dst_layer,
        edges_src=src[order],
        edges_dst=dst[order],
        weights=weights[order],
        d_sae_src=int(d_sae_src),
        d_sae_dst=int(d_sae_dst),
    )


def build_graph_from_sparse_fn(
    states_iter,
    attribution_fn_sparse: Callable[..., tuple],
    src_layer: int,
    dst_layer: int,
    keep_top_n: int = 10_000,
    per_state_top_k: int | None = None,
) -> AttributionGraph:
    """Sparse counterpart of `build_graph`.

    `attribution_fn_sparse(state, src_layer, dst_layer)` must return
    `(rows, cols, vals, d_sae_src, d_sae_dst)` — see
    `attribution_fn.linear_probe_attribution_sparse`.

    Memory is O(sum_states E_state) instead of O(d_sae_src × d_sae_dst);
    for L0 ~ 22 SAEs at d_sae=32768 this is ~500 floats/state vs 1B.

    `per_state_top_k` is optional (default = no per-state filter); set it
    when individual states emit many edges and you want to drop small ones
    before the cross-state aggregation.
    """
    acc: dict[tuple[int, int], float] = {}
    n_states = 0
    d_sae_src: int | None = None
    d_sae_dst: int | None = None

    for state in states_iter:
        rows, cols, vals, dss, dsd = attribution_fn_sparse(state, src_layer, dst_layer)
        if d_sae_src is None:
            d_sae_src, d_sae_dst = int(dss), int(dsd)
        elif (int(dss), int(dsd)) != (d_sae_src, d_sae_dst):
            raise ValueError(
                f"attribution_fn_sparse d_sae changed: expected "
                f"{(d_sae_src, d_sae_dst)} got {(dss, dsd)}"
            )
        n_states += 1
        if rows.size == 0:
            continue
        if per_state_top_k is not None and rows.size > per_state_top_k:
            idx = np.argpartition(np.abs(vals), -per_state_top_k)[-per_state_top_k:]
            rows, cols, vals = rows[idx], cols[idx], vals[idx]
        for r, c, v in zip(rows.tolist(), cols.tolist(), vals.tolist()):
            key = (int(r), int(c))
            acc[key] = acc.get(key, 0.0) + float(v)

    if not acc:
        d_sae_src = d_sae_src or 0
        d_sae_dst = d_sae_dst or 0
        return AttributionGraph(
            src_layer=src_layer, dst_layer=dst_layer,
            edges_src=np.empty(0, dtype=np.int64),
            edges_dst=np.empty(0, dtype=np.int64),
            weights=np.empty(0, dtype=np.float64),
            d_sae_src=d_sae_src, d_sae_dst=d_sae_dst,
        )

    keys = list(acc.keys())
    weights = np.array([acc[k] / max(n_states, 1) for k in keys], dtype=np.float64)
    src = np.array([k[0] for k in keys], dtype=np.int64)
    dst = np.array([k[1] for k in keys], dtype=np.int64)

    order = np.argsort(-np.abs(weights))
    if keep_top_n is not None and keep_top_n > 0:
        order = order[:keep_top_n]
    return AttributionGraph(
        src_layer=src_layer, dst_layer=dst_layer,
        edges_src=src[order],
        edges_dst=dst[order],
        weights=weights[order],
        d_sae_src=int(d_sae_src),
        d_sae_dst=int(d_sae_dst),
    )
