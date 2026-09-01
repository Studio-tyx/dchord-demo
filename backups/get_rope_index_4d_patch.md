# get_rope_index 4D attention_mask patch

## Problem

`Qwen3_5Model.forward` calls `compute_3d_position_ids` -> `get_rope_index`
whenever multimodal data is present (the DChord case: image + `mm_token_type_ids`).
`get_rope_index` uses `attention_mask[batch_idx].bool()` to index a **1D** tensor
(`current_input_ids`, shape `(seq_len,)`) in three places:

1. `current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]`
2. `input_token_type = input_token_type[attention_mask[batch_idx].bool()]`
3. `position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions`

For a 2D `(batch, seq)` mask this is fine. For a **4D** `(batch, 1, q, k)` tree-mask,
`attention_mask[batch_idx]` is 3D `(1, q, k)` and `.bool()` is a 3D bool tensor, so
boolean-indexing a 1D tensor raises `IndexError` (shape mismatch).

## Fix

Insert a 4D -> 2D reduction at the top of `get_rope_index` (before the existing
`# Separate video grid thw` block):

```python
if attention_mask is not None and attention_mask.ndim == 4:
    _am = attention_mask[:, 0]                      # (batch, q, k)
    if _am.dtype == torch.bool:
        attention_mask = _am.any(dim=1)             # (batch, k)  -- bool
    else:
        attention_mask = torch.isfinite(_am).any(dim=1)
    attention_mask = attention_mask.to(dtype=torch.bool, device=attention_mask.device)
```

### Why this is safe (does not break the 2D path)

- The branch is guarded by `attention_mask.ndim == 4`; the 2D path is untouched.
- `get_rope_index` only needs a 2D "valid token" signal to compute 3D M-RoPE
  positions. Reducing 4D -> 2D by OR-ing over the query axis marks a key position
  as valid iff at least one query attends to it. For a no-padding tree-mask this
  is all-True, so all tokens get positions exactly as in the 2D case.
- The **actual 4D masking** is left to the attention layers: `create_causal_mask`
  returns a 4D mask unchanged (`masking_utils.py:810-812`:
  *"If the mask is already 4D, simply return as-is"*), and the SDPA attention
  function consumes it directly. `create_recurrent_attention_mask` returns `None`
  for `ndim != 2`, so linear-attention layers keep their recurrent-causal behaviour
  (no padding to zero out in the single-sequence tree-mask case).
- Reassigning the **local** `attention_mask` parameter does not affect the caller:
  `compute_3d_position_ids` and `Qwen3_5Model.forward` still pass the original 4D
  mask to `language_model(...)` -> `create_causal_mask`.

## File patched

`/root/autodl-tmp/conda/envs/torchspec/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py`

Backup of the original: `modeling_qwen3_5.py.orig` (this directory).

## Reproduce

```bash
python /root/autodl-tmp/0829/backups/patch_get_rope_index.py   # idempotent
```
