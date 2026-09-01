#!/usr/bin/env python3
"""Patch Qwen3.5-VL get_rope_index to accept a 4D attention_mask.

A 4D (batch, 1, q, k) custom attention mask is used by the tree-mask parallel
verify. get_rope_index only needs a 2D (batch, seq) "valid token" signal to
compute 3D M-RoPE positions; the actual 4D masking is handled by the attention
layers (create_causal_mask passes 4D masks through unchanged). So we reduce 4D
-> 2D by OR-ing over the query axis: a key position is "valid" iff at least one
query attends to it.

Idempotent: re-running on an already-patched file is a no-op.
"""
import sys, pathlib

P = pathlib.Path("/root/autodl-tmp/conda/envs/torchspec/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py")
MARKER = "# --- 4D attention_mask support (tree-mask parallel verify) ---"
src = P.read_text()

if MARKER in src:
    print("ALREADY PATCHED: get_rope_index 4D support present.")
    sys.exit(0)

ANCHOR = "        # Separate video grid thw into multiple grids because timestamps are used to separate videos."
assert ANCHOR in src, "anchor not found"

PATCH = '''        # --- 4D attention_mask support (tree-mask parallel verify) ---
        # get_rope_index expects a 2D (batch, seq) mask to flag valid (non-padding)
        # tokens so it can compute 3D M-RoPE positions. When a 4D (batch, 1, q, k)
        # custom attention mask is passed (e.g. tree-mask parallel verify), reduce
        # it to 2D by OR-ing over the query axis: a key position is "valid" iff at
        # least one query attends to it. The actual 4D masking is left to the
        # attention layers -- create_causal_mask returns a 4D mask unchanged -- so
        # only the position computation needs this 2D reduction. The 2D path is
        # untouched.
        if attention_mask is not None and attention_mask.ndim == 4:
            _am = attention_mask[:, 0]  # (batch, q, k)
            if _am.dtype == torch.bool:
                attention_mask = _am.any(dim=1)
            else:
                attention_mask = torch.isfinite(_am).any(dim=1)
            attention_mask = attention_mask.to(dtype=torch.bool, device=attention_mask.device)

'''

new_src = src.replace(ANCHOR, PATCH + ANCHOR, 1)
assert new_src != src, "replace failed"
P.write_text(new_src)
print("PATCHED: get_rope_index now handles 4D attention_mask.")
