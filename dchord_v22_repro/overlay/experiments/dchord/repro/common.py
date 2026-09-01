#!/usr/bin/env python3
"""Sequential, no-future-context DChord-K3 inference correctness probe.

This is the bridge between teacher-forced draft evaluation and an integrated
serving runtime.  It performs real proposal/verification/recovery rounds, but
uses full-prefix Hugging Face recomputation instead of production KV staging;
its acceptance trace is valid while its wall time is only a decomposed harness
measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from torchspec.models.dchord import DChordModel
from torchspec.models.draft.dflash import DFlashConfig, DFlashDraftModel


HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("DCHORD_REPRO_ROOT", "/root/autodl-tmp"))
ASSETS = Path(os.environ.get("DCHORD_REPRO_ASSETS", str(HERE / "assets")))
TARGET = Path(os.environ.get("DCHORD_TARGET", str(ROOT / "models/Qwen3.5-4B")))
CONFIG = ASSETS / "dflash_qwen35_torchspec_config.json"
PCS = ASSETS / "profiled_surface_spec_v1.json"
PROMPT = ASSETS / "prompt_schema_oneshot_qwen35_v1.txt"
TEST = ASSETS / "p1_test_512.jsonl"
PARQUET = Path(os.environ.get(
    "DCHORD_CELEBA_PARQUET",
    str(ROOT / "datasets/celeba/img_align+identity+attr"),
))
EXPORT = Path(os.environ.get(
    "DCHORD_K3_CHECKPOINT",
    str(ROOT / "models/DChord-K3-CelebA/pytorch_model.bin"),
))
OUT = Path(os.environ.get(
    "DCHORD_REPRO_OUTPUT",
    str(ROOT / "reports/dchord_torch_repro"),
))
LAYERS = [1, 5, 9, 13, 17, 21, 25, 29]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_image(row: dict[str, Any]) -> Image.Image:
    parquet = pq.ParquetFile(PARQUET / row["shard"])
    target = int(row["row_index"])
    cursor = 0
    for group_index in range(parquet.metadata.num_row_groups):
        count = parquet.metadata.row_group(group_index).num_rows
        if target < cursor + count:
            raw = parquet.read_row_group(group_index, columns=["image"]).slice(
                target - cursor, 1
            ).to_pylist()[0]["image"]["bytes"]
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                return image.convert("RGB")
        cursor += count
    raise IndexError(row["sample_id"])


def render(segments: list[str], attributes: list[str], branches: list[int]) -> str:
    pieces: list[str] = []
    for index, segment in enumerate(segments):
        pieces.append(segment)
        if index < len(attributes):
            pieces.append("true" if branches[index] else "false")
    return "".join(pieces)


def build_contract(tokenizer, segments: list[str], attributes: list[str]):
    false_text = render(segments, attributes, [0] * len(attributes))
    true_text = render(segments, attributes, [1] * len(attributes))
    false_ids = tokenizer.encode(false_text, add_special_tokens=False)
    true_ids = tokenizer.encode(true_text, add_special_tokens=False)
    if len(false_ids) != len(true_ids):
        raise ValueError("boolean alternatives change compiled token length")
    positions = [index for index, pair in enumerate(zip(false_ids, true_ids)) if pair[0] != pair[1]]
    if len(positions) != len(attributes):
        raise ValueError(f"expected {len(attributes)} branch positions, got {len(positions)}")
    pairs = [[int(false_ids[index]), int(true_ids[index])] for index in positions]
    return [int(value) for value in false_ids], positions, pairs


def processor_inputs(processor, image: Image.Image, prompt: str):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt}
    ]}]
    encoded = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    return {key: value for key, value in encoded.items() if torch.is_tensor(value)}


def model_inputs(encoded, completion_ids: list[int], device: torch.device):
    completion = torch.tensor([completion_ids], dtype=torch.long)
    input_ids = torch.cat([encoded["input_ids"], completion], dim=1).to(device)
    attention_mask = torch.cat([
        encoded["attention_mask"],
        torch.ones_like(completion, dtype=encoded["attention_mask"].dtype),
    ], dim=1).to(device)
    kwargs = {}
    for key, value in encoded.items():
        if key in {"input_ids", "attention_mask", "token_type_ids"}:
            continue
        if key == "mm_token_type_ids":
            value = torch.cat([value, torch.zeros_like(completion, dtype=value.dtype)], dim=1)
        kwargs[key] = value.to(device)
    return input_ids, attention_mask, kwargs


@torch.inference_mode()
def target_context(model, encoded, completion_ids, device):
    input_ids, attention_mask, kwargs = model_inputs(encoded, completion_ids, device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        logits_to_keep=1,
        return_dict=True,
        **kwargs,
    )
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000
    selected = [outputs.hidden_states[layer + 1] for layer in LAYERS]
    return input_ids, selected, elapsed_ms


@torch.inference_mode()
def target_choices(
    model,
    encoded,
    completion_ids: list[int],
    completion_prediction_positions: list[int],
    field_ids: list[int],
    branch_pairs: list[list[int]],
    device: torch.device,
):
    input_ids, attention_mask, kwargs = model_inputs(encoded, completion_ids, device)
    prompt_length = int(encoded["input_ids"].shape[1])
    keep = torch.tensor(
        [prompt_length + position - 1 for position in completion_prediction_positions],
        dtype=torch.long,
        device=device,
    )
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
        use_cache=False,
        logits_to_keep=keep,
        return_dict=True,
        **kwargs,
    )
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000
    choices, full_top1_in_branch = [], []
    for logit_index, field_index in enumerate(field_ids):
        pair = branch_pairs[field_index]
        pair_tensor = torch.tensor(pair, dtype=torch.long, device=device)
        branch = int(outputs.logits[0, logit_index, pair_tensor].argmax())
        choices.append(pair[branch])
        full_top1_in_branch.append(int(outputs.logits[0, logit_index].argmax()) in pair)
    return choices, full_top1_in_branch, elapsed_ms


def build_dchord(tokenizer, attributes, pairs, device, export_path: Path):
    config = DFlashConfig(**json.loads(CONFIG.read_text()))
    draft = DFlashDraftModel(config)
    schema_ids = [tokenizer.encode(f'"{field}"', add_special_tokens=False) for field in attributes]
    wrapper = DChordModel(draft, schema_ids, pairs, k=3, dpace_alpha=0.5)
    state = torch.load(export_path, map_location="cpu", weights_only=True)
    result = draft.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(result)
    del state
    return wrapper.to(device=device, dtype=torch.bfloat16).eval()


@torch.inference_mode()
def draft_choices(dchord, hidden_states, field_start, absolute_positions, pairs, device):
    positions = torch.tensor([absolute_positions], dtype=torch.long, device=device)
    transferred = [value.to(device) for value in hidden_states]
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    logits = dchord.propose_block(
        transferred,
        field_start=field_start,
        draft_position_ids=positions,
        lm_head_weight=dchord.draft_model.embed_tokens.weight,
    )
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000
    choices, full_top1_in_branch = [], []
    for offset in range(logits.shape[1]):
        pair = pairs[field_start + offset]
        pair_tensor = torch.tensor(pair, dtype=torch.long, device=device)
        branch = int(logits[0, offset, pair_tensor].argmax())
        choices.append(pair[branch])
        full_top1_in_branch.append(int(logits[0, offset].argmax()) in pair)
    return choices, full_top1_in_branch, elapsed_ms


def candidate_span(base_ids, positions, pairs, field_start, choices):
    field_end = field_start + len(choices)
    span_end = positions[field_end] if field_end < len(positions) else len(base_ids)
    span = list(base_ids[positions[field_start]:span_end])
    for offset, choice in enumerate(choices):
        absolute = positions[field_start + offset]
        span[absolute - positions[field_start]] = int(choice)
    return span, span_end


def sequential_reference(model, encoded, base_ids, positions, pairs, device):
    completion = list(base_ids[:positions[0]])
    branches, elapsed = [], 0.0
    for field in range(len(positions)):
        if len(completion) != positions[field]:
            raise RuntimeError("reference compiler cursor drift")
        choices, _, duration = target_choices(
            model, encoded, completion, [positions[field]], [field], pairs, device
        )
        elapsed += duration
        choice = choices[0]
        branches.append(pairs[field].index(choice))
        completion.append(choice)
        end = positions[field + 1] if field + 1 < len(positions) else len(base_ids)
        completion.extend(base_ids[positions[field] + 1:end])
    return completion, branches, elapsed


def infer_one(
    target,
    dchord,
    encoded,
    base_ids,
    positions,
    pairs,
    prompt_length,
    target_device,
    draft_device,
    bonus_mode,
):
    completion = list(base_ids[:positions[0]])
    branches = [-1] * len(positions)
    rounds = []
    field = 0
    while field < len(positions):
        if len(completion) != positions[field]:
            raise RuntimeError(f"compiler cursor drift at field {field}")
        before = len(completion)
        block_size = min(3, len(positions) - field)
        _, hidden_states, context_ms = target_context(
            target, encoded, completion, target_device
        )
        absolute_positions = [prompt_length + positions[index] for index in range(field, field + block_size)]
        proposals, draft_top1_ok, draft_ms = draft_choices(
            dchord, hidden_states, field, absolute_positions, pairs, draft_device
        )
        span, span_end = candidate_span(base_ids, positions, pairs, field, proposals)
        candidate_completion = completion + span

        verify_fields = list(range(field, field + block_size))
        verify_positions = [positions[index] for index in verify_fields]
        if bonus_mode == "standard" and field + block_size < len(positions):
            verify_fields.append(field + block_size)
            verify_positions.append(positions[field + block_size])
        verifier, target_top1_ok, verify_ms = target_choices(
            target,
            encoded,
            candidate_completion,
            verify_positions,
            verify_fields,
            pairs,
            target_device,
        )
        mismatch = next(
            (offset for offset in range(block_size) if proposals[offset] != verifier[offset]),
            None,
        )
        target_bonus = 0
        if mismatch is not None:
            accepted = mismatch
            rejected_field = field + mismatch
            rejected_position = positions[rejected_field]
            completion.extend(span[:rejected_position - positions[field]])
            correction = verifier[mismatch]
            completion.append(correction)
            branches[rejected_field] = pairs[rejected_field].index(correction)
            next_field = rejected_field + 1
            fixed_end = positions[next_field] if next_field < len(positions) else len(base_ids)
            completion.extend(base_ids[rejected_position + 1:fixed_end])
            for offset in range(mismatch):
                accepted_field = field + offset
                branches[accepted_field] = pairs[accepted_field].index(proposals[offset])
            field = next_field
            accepted_expanded = rejected_position - positions[field - accepted - 1]
        else:
            accepted = block_size
            completion = candidate_completion
            for offset, proposal in enumerate(proposals):
                branches[field + offset] = pairs[field + offset].index(proposal)
            field += block_size
            accepted_expanded = len(span)
            if bonus_mode == "standard" and field < len(positions):
                bonus = verifier[block_size]
                completion.append(bonus)
                branches[field] = pairs[field].index(bonus)
                next_field = field + 1
                fixed_end = positions[next_field] if next_field < len(positions) else len(base_ids)
                completion.extend(base_ids[positions[field] + 1:fixed_end])
                field = next_field
                target_bonus = 1

        rounds.append({
            "round": len(rounds),
            "field_start": field - accepted - target_bonus - (1 if mismatch is not None else 0),
            "draft_decisions": block_size,
            "accepted_draft_decisions": accepted,
            "target_bonus_decisions": target_bonus + int(mismatch is not None),
            "accepted_expanded_target_tokens": accepted_expanded,
            "committed_completion_tokens": len(completion) - before,
            "first_rejection_offset": mismatch,
            "draft_full_top1_in_branch": draft_top1_ok,
            "target_full_top1_in_branch": target_top1_ok,
            "context_recompute_ms": context_ms,
            "draft_ms": draft_ms,
            "verify_recompute_ms": verify_ms,
        })
    if any(value < 0 for value in branches):
        raise RuntimeError("not all decisions were committed")
    return completion, branches, rounds


def run(
    limit: int,
    reference_limit: int,
    bonus_mode: str,
    export_path: Path,
    output_dir: Path,
):
    target_device, draft_device = torch.device("cuda:0"), torch.device("cuda:1")
    tokenizer = AutoTokenizer.from_pretrained(TARGET, local_files_only=True)
    processor = AutoProcessor.from_pretrained(TARGET, local_files_only=True)
    pcs = json.loads(PCS.read_text())
    attributes = list(pcs["schema"]["attributes"])
    segments = list(pcs["surface"]["segments"])
    base_ids, positions, pairs = build_contract(tokenizer, segments, attributes)
    target = AutoModelForImageTextToText.from_pretrained(
        TARGET,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(target_device).eval()
    dchord = build_dchord(tokenizer, attributes, pairs, draft_device, export_path)
    rows = read_jsonl(TEST)[:limit]
    prompt = PROMPT.read_text()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sequential_{bonus_mode}_{limit}"
    raw_path = output_dir / f"{stem}_rows.jsonl"
    raw_path.unlink(missing_ok=True)
    results = []
    for index, row in enumerate(rows):
        encoded = processor_inputs(processor, load_image(row), prompt)
        prompt_length = int(encoded["input_ids"].shape[1])
        completion, branches, rounds = infer_one(
            target,
            dchord,
            encoded,
            base_ids,
            positions,
            pairs,
            prompt_length,
            target_device,
            draft_device,
            bonus_mode,
        )
        text = tokenizer.decode(completion, skip_special_tokens=False)
        compiler_text = render(segments, attributes, branches)
        reference_exact = None
        reference_ms = None
        if index < reference_limit:
            reference, reference_branches, reference_ms = sequential_reference(
                target, encoded, base_ids, positions, pairs, target_device
            )
            reference_exact = completion == reference and branches == reference_branches
        prelude_tokens = positions[0]
        committed_round_tokens = sum(r["committed_completion_tokens"] for r in rounds)
        if prelude_tokens + committed_round_tokens != len(completion):
            raise RuntimeError("completion accounting drift")
        record = {
            "index": index,
            "sample_id": row["sample_id"],
            "bonus_mode": bonus_mode,
            "rounds": len(rounds),
            "completion_tokens": len(completion),
            "prelude_tokens": prelude_tokens,
            "committed_round_tokens": committed_round_tokens,
            "mean_committed_tokens_per_round": committed_round_tokens / len(rounds),
            "mean_accepted_draft_decisions": sum(r["accepted_draft_decisions"] for r in rounds) / len(rounds),
            "mean_accepted_expanded_target_tokens": sum(
                r["accepted_expanded_target_tokens"] for r in rounds
            ) / len(rounds),
            "rejection_rounds": sum(r["first_rejection_offset"] is not None for r in rounds),
            "target_bonus_decisions": sum(r["target_bonus_decisions"] for r in rounds),
            "full_block_accept_rate": sum(
                r["accepted_draft_decisions"] == r["draft_decisions"] for r in rounds
            ) / len(rounds),
            "surface_token_exact": text == compiler_text,
            "json_valid": isinstance(json.loads(text), dict),
            "reference_exact": reference_exact,
            "reference_recompute_ms": reference_ms,
            "context_recompute_ms": sum(r["context_recompute_ms"] for r in rounds),
            "draft_ms": sum(r["draft_ms"] for r in rounds),
            "verify_recompute_ms": sum(r["verify_recompute_ms"] for r in rounds),
            "draft_full_top1_out_of_branch": sum(
                not value for r in rounds for value in r["draft_full_top1_in_branch"]
            ),
            "target_full_top1_out_of_branch": sum(
                not value for r in rounds for value in r["target_full_top1_in_branch"]
            ),
            "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "branches": branches,
            "round_trace": rounds,
        }
        results.append(record)
        with raw_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({key: value for key, value in record.items() if key != "round_trace"}), flush=True)

    summary = {
        "version": "dchord_k3_sequential_inference_v2",
        "bonus_mode": bonus_mode,
        "export_path": str(export_path),
        "records": len(results),
        "reference_records": sum(row["reference_exact"] is not None for row in results),
        "reference_exact": sum(row["reference_exact"] is True for row in results),
        "surface_token_exact": sum(row["surface_token_exact"] for row in results),
        "json_valid": sum(row["json_valid"] for row in results),
        "mean_rounds": sum(row["rounds"] for row in results) / len(results),
        "mean_committed_tokens_per_round": sum(row["committed_round_tokens"] for row in results) / sum(row["rounds"] for row in results),
        "mean_accepted_draft_decisions": sum(
            r["accepted_draft_decisions"] for row in results for r in row["round_trace"]
        ) / sum(row["rounds"] for row in results),
        "mean_accepted_expanded_target_tokens": sum(
            r["accepted_expanded_target_tokens"] for row in results for r in row["round_trace"]
        ) / sum(row["rounds"] for row in results),
        "rejection_rounds": sum(row["rejection_rounds"] for row in results),
        "target_bonus_decisions": sum(row["target_bonus_decisions"] for row in results),
        "first_rejection_offset_counts": {
            str(offset): sum(
                r["first_rejection_offset"] == offset
                for row in results
                for r in row["round_trace"]
            )
            for offset in range(3)
        },
        "mean_full_block_accept_rate": sum(row["full_block_accept_rate"] for row in results) / len(results),
        "mean_context_recompute_ms": sum(row["context_recompute_ms"] for row in results) / len(results),
        "mean_draft_ms": sum(row["draft_ms"] for row in results) / len(results),
        "mean_verify_recompute_ms": sum(row["verify_recompute_ms"] for row in results) / len(results),
        "draft_full_top1_out_of_branch": sum(row["draft_full_top1_out_of_branch"] for row in results),
        "target_full_top1_out_of_branch": sum(row["target_full_top1_out_of_branch"] for row in results),
        "timing_scope": "full-prefix HF recomputation; valid for stage decomposition, not production speedup",
    }
    (output_dir / f"{stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--reference-limit", type=int, default=2)
    parser.add_argument("--bonus-mode", choices=["none", "standard"], default="none")
    parser.add_argument("--export", type=Path, default=EXPORT)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.limit, args.reference_limit, args.bonus_mode, args.export, args.output_dir)
