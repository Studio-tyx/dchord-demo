#!/usr/bin/env python3
"""Validate transactional rollback for Qwen3.5 hybrid HF caches."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.dchord.repro.common import (  # noqa: E402
    PROMPT, TARGET, TEST, load_image, processor_inputs, read_jsonl,
)

OUT = Path(os.environ.get(
    "DCHORD_REPRO_CACHE_OUTPUT",
    "/root/autodl-tmp/reports/dchord_torch_repro/cache_transaction",
))


def clone_optional(value):
    return None if value is None else value.clone()


def snapshot_cache(cache):
    layers = []
    for layer in cache.layers:
        if hasattr(layer, "recurrent_states"):
            layers.append({
                "kind": "linear",
                "conv": {i: clone_optional(v) for i, v in layer.conv_states.items()},
                "recurrent": {
                    i: clone_optional(v) for i, v in layer.recurrent_states.items()
                },
                "has_previous": dict(layer.has_previous_state),
                "conv_initialized": dict(layer.is_conv_states_initialized),
                "recurrent_initialized": dict(layer.is_recurrent_states_initialized),
            })
        else:
            layers.append({
                "kind": "kv",
                "length": layer.get_seq_length(),
            })
    return layers


def restore_cache(cache, snapshot):
    for layer, saved in zip(cache.layers, snapshot, strict=True):
        if saved["kind"] == "kv":
            length = saved["length"]
            if getattr(layer, "is_initialized", False):
                layer.keys = layer.keys[..., :length, :]
                layer.values = layer.values[..., :length, :]
        else:
            layer.has_previous_state = dict(saved["has_previous"])
            layer.is_conv_states_initialized = dict(saved["conv_initialized"])
            layer.is_recurrent_states_initialized = dict(saved["recurrent_initialized"])
            for index, value in saved["conv"].items():
                if value is None:
                    layer.conv_states[index] = None
                elif layer.conv_states[index] is None or layer.conv_states[index].shape != value.shape:
                    layer.conv_states[index] = value.clone()
                else:
                    layer.conv_states[index].copy_(value)
            for index, value in saved["recurrent"].items():
                if value is None:
                    layer.recurrent_states[index] = None
                elif layer.recurrent_states[index] is None or layer.recurrent_states[index].shape != value.shape:
                    layer.recurrent_states[index] = value.clone()
                else:
                    layer.recurrent_states[index].copy_(value)


def state_fingerprint(cache):
    values = []
    for layer in cache.layers:
        if hasattr(layer, "recurrent_states"):
            for tensor in list(layer.conv_states.values()) + list(layer.recurrent_states.values()):
                if tensor is not None:
                    values.append((tuple(tensor.shape), float(tensor.float().sum().item())))
        elif getattr(layer, "is_initialized", False):
            values.append((tuple(layer.keys.shape), float(layer.keys.float().sum().item())))
            values.append((tuple(layer.values.shape), float(layer.values.float().sum().item())))
    return values


@torch.inference_mode()
def prefill(model, encoded, device):
    kwargs = {k: v.to(device) for k, v in encoded.items() if torch.is_tensor(v)}
    out = model(
        **kwargs, use_cache=True, output_hidden_states=True,
        logits_to_keep=1, return_dict=True,
    )
    return out.past_key_values, out.logits[0, -1].float(), int(encoded["input_ids"].shape[1])


@torch.inference_mode()
def advance(model, cache, tokens, total_before, device):
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    total_after = total_before + len(tokens)
    attention_mask = torch.ones((1, total_after), dtype=torch.long, device=device)
    cache_position = torch.arange(total_before, total_after, device=device)
    rope_delta = model.model.rope_deltas.to(device=device, dtype=torch.long)
    position_ids = cache_position.view(1, 1, -1).expand(3, 1, -1)
    position_ids = position_ids + rope_delta.view(1, 1, 1)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        cache_position=cache_position,
        position_ids=position_ids,
        use_cache=True,
        output_hidden_states=True,
        logits_to_keep=len(tokens),
        return_dict=True,
    )
    return out.logits[0].float(), out.hidden_states


@torch.inference_mode()
def full_prefix_top1(model, encoded, completion, device):
    prompt_ids = encoded["input_ids"]
    extra = torch.tensor([completion], dtype=torch.long)
    kwargs = {}
    for key, value in encoded.items():
        if not torch.is_tensor(value):
            continue
        if key == "input_ids":
            kwargs[key] = torch.cat([value, extra], dim=1).to(device)
        elif key in {"attention_mask", "mm_token_type_ids"}:
            fill = torch.ones_like(extra) if key == "attention_mask" else torch.zeros_like(extra)
            kwargs[key] = torch.cat([value, fill.to(value.dtype)], dim=1).to(device)
        else:
            kwargs[key] = value.to(device)
    keep = torch.arange(
        int(prompt_ids.shape[1]) - 1,
        int(prompt_ids.shape[1]) + len(completion) - 1,
        device=device,
    )
    out = model(
        **kwargs, use_cache=False, output_hidden_states=False,
        logits_to_keep=keep, return_dict=True,
    )
    return [int(x) for x in out.logits[0].argmax(-1)]


@torch.inference_mode()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(TARGET, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        TARGET, dtype=torch.bfloat16, local_files_only=True,
        low_cpu_mem_usage=True, attn_implementation="sdpa",
    ).to(device).eval()
    sample = read_jsonl(TEST)[0]
    encoded = processor_inputs(processor, load_image(sample), PROMPT.read_text())
    torch.cuda.synchronize(); started = time.perf_counter()
    cache, next_logits, prompt_len = prefill(model, encoded, device)
    base = snapshot_cache(cache)
    base_fp = state_fingerprint(cache)

    # Produce an 8-token direct cached greedy reference.
    reference = []
    direct_logits = next_logits
    for offset in range(8):
        token = int(direct_logits.argmax())
        reference.append(token)
        logits, _ = advance(model, cache, [token], prompt_len + offset, device)
        direct_logits = logits[-1]
    tokenwise_next = int(direct_logits.argmax())
    tokenwise_fp = state_fingerprint(cache)

    # Same 8 tokens in one block: this is the matched transaction reference.
    restore_cache(cache, base)
    block_logits, _ = advance(model, cache, reference, prompt_len, device)
    direct_next = int(block_logits[-1].argmax())
    direct_fp = state_fingerprint(cache)

    # Restore prompt state, contaminate it with a wrong block, then rollback.
    restore_cache(cache, base)
    restored_base_fp = state_fingerprint(cache)
    wrong = [0, 1, 2, 3]
    advance(model, cache, wrong, prompt_len, device)
    restore_cache(cache, base)
    restored_after_wrong_fp = state_fingerprint(cache)

    # Replay the same correct block after contamination and rollback.
    recovered_logits, _ = advance(model, cache, reference, prompt_len, device)
    recovered_next = int(recovered_logits[-1].argmax())
    recovered_fp = state_fingerprint(cache)
    full_top1 = full_prefix_top1(model, encoded, reference, device)
    elapsed = time.perf_counter() - started

    summary = {
        "version": "q35_hybrid_cache_transaction_probe_v1",
        "sample_id": sample["sample_id"],
        "reference_tokens": reference,
        "full_prefix_top1": full_top1,
        "cached_reference_matches_full_prefix": reference == full_top1,
        "base_restore_exact": base_fp == restored_base_fp,
        "wrong_candidate_rollback_exact": base_fp == restored_after_wrong_fp,
        "recovered_state_fingerprint_exact": direct_fp == recovered_fp,
        "direct_next_token": direct_next,
        "recovered_next_token": recovered_next,
        "recovered_next_token_exact": direct_next == recovered_next,
        "tokenwise_next_token": tokenwise_next,
        "tokenwise_vs_block_next_exact": tokenwise_next == direct_next,
        "tokenwise_vs_block_state_fingerprint_exact": tokenwise_fp == direct_fp,
        "linear_layers": sum(hasattr(layer, "recurrent_states") for layer in cache.layers),
        "kv_layers": sum(not hasattr(layer, "recurrent_states") for layer in cache.layers),
        "elapsed_seconds": elapsed,
    }
    summary["pass"] = all([
        summary["cached_reference_matches_full_prefix"],
        summary["base_restore_exact"],
        summary["wrong_candidate_rollback_exact"],
        summary["recovered_state_fingerprint_exact"],
        summary["recovered_next_token_exact"],
    ])
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
