"""Convert BFCL v4 parquet (verl GRPO format) into SFT chat-completion data.

For each row in `bfcl_train.parquet`, we use the canonical ground_truth call
as the model's expected response. Output is a parquet of `messages` lists
ready for `trl.SFTTrainer` / `verl.trainer.sft_trainer` style consumption.

Schema (single output parquet):
    messages: list[dict[str, str]]    # [{"role":"system",...}, {"role":"user",...}, {"role":"assistant",...}]
    data_source: str
    ability: str
    extra_info: dict

For BFCL ground_truth (a list of acceptable calls per case), we pick the
*first* call by stable iteration order — that becomes the SFT target. The
GRPO downstream will see all alternates via `bfcl_reward.compute_score`.

Usage:
    python scripts/prepare_sft_from_bfcl.py \
        --in-parquet $WORK/bfcl_train.parquet \
        --out-parquet $WORK/bfcl_sft_train.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def first_call_from_ground_truth(gt_json: str) -> str | None:
    """BFCL ground_truth is a JSON string. The list-of-dicts shape is:
        [{"func_name": {"arg": [accepts]}}, ...]
    Pick the first call entry, render it back into a Python-call string
    using the FIRST accepted value per kwarg.
    """
    try:
        gt_list = json.loads(gt_json)
    except json.JSONDecodeError:
        return None
    if isinstance(gt_list, dict):
        gt_list = [gt_list]
    if not gt_list:
        return None
    first = gt_list[0]
    if not isinstance(first, dict) or not first:
        return None
    func_name, kwargs_dict = next(iter(first.items()))
    if not isinstance(kwargs_dict, dict):
        return None
    parts = []
    for k, accepts in kwargs_dict.items():
        if not isinstance(accepts, list):
            accepts = [accepts]
        if not accepts:
            continue
        v = accepts[0]
        parts.append(f"{k}={repr(v)}")
    return f"{func_name}({', '.join(parts)})"


def to_sft_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a verl-style GRPO row into an SFT row by appending an
    `assistant` message containing the canonical call. Returns None if the
    ground_truth can't be rendered (those rows skipped — typically <1%)."""
    prompt = row["prompt"]
    if not isinstance(prompt, list):
        return None
    rm = row.get("reward_model", {})
    if isinstance(rm, dict):
        gt_json = rm.get("ground_truth")
    else:
        gt_json = None
    if not gt_json:
        return None
    call = first_call_from_ground_truth(gt_json)
    if call is None:
        return None
    messages = [
        # original prompt is already [system, user]
        {"role": m["role"], "content": m["content"]} for m in prompt
    ]
    messages.append({"role": "assistant", "content": call})
    return {
        "messages": messages,
        "data_source": row.get("data_source", "bfcl_v4_unknown"),
        "ability": row.get("ability", "unknown"),
        "extra_info": row.get("extra_info", {}) or {},
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-parquet", type=Path, required=True)
    ap.add_argument("--out-parquet", type=Path, required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    table = pq.read_table(args.in_parquet)
    rows = table.to_pylist()
    out_rows: list[dict[str, Any]] = []
    n_skip = 0
    for r in rows:
        sft = to_sft_row(r)
        if sft is None:
            n_skip += 1
            continue
        out_rows.append(sft)
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(out_rows), args.out_parquet, compression="zstd")
    print(f"converted {len(out_rows)} rows / skipped {n_skip} → {args.out_parquet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
