#!/usr/bin/env python3
"""P1.0 compatibility gate for Qwen3.5-4B-DFlash and TorchSpec."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

from torchspec.models.draft.dflash import DFlashConfig, DFlashDraftModel
from torchspec.models.draft.keymap import to_internal_keys


ROOT = Path(os.environ.get("DCHORD_REPRO_ROOT", "/root/autodl-tmp"))
TARGET = Path(os.environ.get("DCHORD_TARGET", str(ROOT / "models/Qwen3.5-4B")))
OFFICIAL = Path(os.environ.get(
    "DCHORD_OFFICIAL_DFLASH", str(ROOT / "models/Qwen3.5-4B-DFlash")
))
REPORT = Path(os.environ.get(
    "DCHORD_COMPAT_OUTPUT",
    str(ROOT / "reports/dchord_torch_repro/training/compat"),
))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def derived_config() -> dict:
    raw = json.loads((OFFICIAL / "config.json").read_text())
    dflash = raw["dflash_config"]
    rope = raw.get("rope_parameters", {})
    return {
        "architectures": ["DFlashDraftModel"],
        "model_type": "dflash",
        "hidden_size": raw["hidden_size"],
        "intermediate_size": raw["intermediate_size"],
        "num_hidden_layers": raw["num_hidden_layers"],
        "num_attention_heads": raw["num_attention_heads"],
        "num_key_value_heads": raw["num_key_value_heads"],
        "head_dim": raw["head_dim"],
        "vocab_size": raw["vocab_size"],
        "rms_norm_eps": raw["rms_norm_eps"],
        "max_position_embeddings": raw["max_position_embeddings"],
        "rope_theta": rope.get("rope_theta", 10_000_000),
        "rope_parameters": rope,
        "num_target_layers": len(dflash["target_layer_ids"]),
        "target_hidden_size": raw["hidden_size"],
        "target_num_hidden_layers": raw["num_target_layers"],
        "target_layer_ids": dflash["target_layer_ids"],
        "fc_norm": False,
        "mask_token_id": dflash["mask_token_id"],
        "tie_word_embeddings": True,
        "source_block_size": dflash["block_size"],
        "dchord_k": 3,
        "embedding_key": "model.language_model.embed_tokens.weight",
        "lm_head_key": "model.language_model.embed_tokens.weight",
        "norm_key": "model.language_model.norm.weight",
    }


def load_target_tensor(key: str) -> torch.Tensor:
    index = json.loads((TARGET / "model.safetensors.index.json").read_text())
    filename = index["weight_map"][key]
    with safe_open(TARGET / filename, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def run(device: str) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    config_dict = derived_config()
    write_json(REPORT / "dflash_qwen35_torchspec_config.json", config_dict)
    config = DFlashConfig(**config_dict)
    model = DFlashDraftModel(config)

    with safe_open(OFFICIAL / "model.safetensors", framework="pt", device="cpu") as handle:
        official = {key: handle.get_tensor(key) for key in handle.keys()}
    mapped = to_internal_keys(official, model.state_dict().keys())
    load_result = model.load_state_dict(mapped, strict=False)
    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)
    allowed_missing = ["embed_tokens.weight"]
    strict_non_embedding = missing == allowed_missing and not unexpected
    if not strict_non_embedding:
        raise RuntimeError(f"checkpoint mapping mismatch: missing={missing}, unexpected={unexpected}")

    embedding = load_target_tensor(config_dict["embedding_key"])
    if tuple(embedding.shape) != tuple(model.embed_tokens.weight.shape):
        raise RuntimeError(
            f"target embedding shape mismatch: {tuple(embedding.shape)} vs "
            f"{tuple(model.embed_tokens.weight.shape)}"
        )
    model.embed_tokens.weight.data.copy_(embedding)
    model.freeze_embedding()

    checks = []
    state = model.state_dict()
    for export_key, tensor in official.items():
        internal = next(key for key, value in mapped.items() if value.data_ptr() == tensor.data_ptr())
        checks.append(torch.equal(state[internal].cpu(), tensor.cpu()))

    target_device = torch.device(device)
    model = model.to(device=target_device, dtype=torch.bfloat16).eval()
    generator = torch.Generator(device=target_device).manual_seed(20260824)
    hidden = [
        torch.randn(1, 12, config.target_hidden_size, generator=generator,
                    device=target_device, dtype=torch.bfloat16)
        for _ in range(config.num_target_layers)
    ]
    context = model.extract_context_feature(hidden)
    draft_ids = torch.full((1, 3), config.mask_token_id, device=target_device, dtype=torch.long)
    with torch.inference_mode():
        output = model(
            draft_input_ids=draft_ids,
            context_feature=context,
            context_position_ids=torch.arange(12, device=target_device).unsqueeze(0),
            draft_position_ids=torch.tensor([[12, 13, 14]], device=target_device),
            block_mask=None,
        )
        logits = torch.nn.functional.linear(output, model.embed_tokens.weight)
    finite = bool(torch.isfinite(output).all() and torch.isfinite(logits).all())
    summary = {
        "version": "q35_p1_compat_v1",
        "official_config_sha256": sha256(OFFICIAL / "config.json"),
        "official_weights_sha256": sha256(OFFICIAL / "model.safetensors"),
        "target_config_sha256": sha256(TARGET / "config.json"),
        "derived_config": config_dict,
        "official_tensor_count": len(official),
        "mapped_tensor_count": len(mapped),
        "exact_loaded_tensors": sum(checks),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "target_embedding_shape": list(embedding.shape),
        "trainable_parameters_before_dchord_adapter": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "frozen_parameters": sum(
            parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
        ),
        "forward": {
            "context_shape": list(context.shape),
            "output_shape": list(output.shape),
            "logits_shape": list(logits.shape),
            "finite": finite,
        },
        "pass": strict_non_embedding and all(checks) and finite,
    }
    write_json(REPORT / "compat_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary["pass"]:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args().device)


if __name__ == "__main__":
    main()
