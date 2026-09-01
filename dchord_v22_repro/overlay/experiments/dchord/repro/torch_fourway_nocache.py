#!/usr/bin/env python3
"""Matched no-cache speed test: AR, schema target, DFlash-B16, DChord-K3."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.dchord.repro.common import (  # noqa: E402
    CONFIG, EXPORT, LAYERS, PCS, PROMPT, TARGET, TEST, build_contract,
    build_dchord, candidate_span, load_image, model_inputs, processor_inputs,
    read_jsonl,
)
from torchspec.models.draft.dflash import DFlashConfig, DFlashDraftModel  # noqa: E402
from torchspec.models.draft.keymap import to_internal_keys  # noqa: E402

ROOT = Path(os.environ.get("DCHORD_REPRO_ROOT", "/root/autodl-tmp"))
OFFICIAL_DFLASH = Path(os.environ.get(
    "DCHORD_OFFICIAL_DFLASH",
    str(ROOT / "models/Qwen3.5-4B-DFlash"),
))
DEFAULT_OUT = Path(os.environ.get(
    "DCHORD_REPRO_OUTPUT",
    str(ROOT / "reports/dchord_torch_repro/nocache"),
))
MODES = ("ar", "schema", "dflash_b16", "dchord_zeroft", "dchord_k3")


def sync(device):
    torch.cuda.synchronize(device)


def pct(values, q):
    values = sorted(values)
    at = (len(values) - 1) * q
    lo, hi = math.floor(at), math.ceil(at)
    return values[lo] if lo == hi else values[lo] * (hi - at) + values[hi] * (at - lo)


def parse_schema(text: str, attributes, labels):
    start = text.find("{")
    parsed, error = None, None
    if start < 0:
        error = "no JSON object start"
    else:
        try:
            parsed, _ = json.JSONDecoder().raw_decode(text[start:])
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    valid = (
        isinstance(parsed, dict) and list(parsed) == attributes
        and all(type(parsed[name]) is bool for name in attributes)
    )
    return {
        "json_object_found": isinstance(parsed, dict),
        "parse_error": error,
        "exact_40_bool": valid,
        "field_correct": sum(parsed[n] == labels[n] for n in attributes) if valid else None,
        "field_total": len(attributes) if valid else 0,
    }


@torch.inference_mode()
def target_forward(model, encoded, completion, device, hidden=False, keep=1):
    ids, mask, kwargs = model_inputs(encoded, completion, device)
    sync(device)
    started = time.perf_counter()
    out = model(
        input_ids=ids, attention_mask=mask, output_hidden_states=hidden,
        use_cache=False, logits_to_keep=keep, return_dict=True, **kwargs,
    )
    sync(device)
    return ids, out, (time.perf_counter() - started) * 1000


@torch.inference_mode()
def target_branches(model, encoded, completion, positions, fields, pairs, device):
    prompt_len = int(encoded["input_ids"].shape[1])
    keep = torch.tensor([prompt_len + p - 1 for p in positions], device=device)
    _, out, elapsed = target_forward(model, encoded, completion, device, keep=keep)
    choices, margins = [], []
    for row, field in enumerate(fields):
        pair = torch.tensor(pairs[field], device=device)
        scores = out.logits[0, row, pair].float()
        choices.append(int(pair[int(scores.argmax())]))
        margins.append(float((scores.max() - scores.min()).item()))
    return choices, margins, elapsed


def result(completion, rounds, target_ms, draft_ms=0.0, accepted=0, proposed=0,
           bonuses=0, branches=None, trace=None, min_margin=None):
    value = {
        "completion_ids": completion,
        "rounds": len(rounds),
        "target_calls": sum(r.get("target_calls", 1) for r in rounds),
        "draft_calls": sum(r.get("draft_calls", 0) for r in rounds),
        "target_ms": target_ms,
        "draft_ms": draft_ms,
        "first_step_ms": rounds[0]["elapsed_ms"],
        "generation_ms": target_ms + draft_ms,
        "accepted_draft_tokens": accepted,
        "proposed_draft_tokens": proposed,
        "bonus_or_correction_tokens": bonuses,
    }
    if branches is not None:
        value["branches"] = branches
    if trace is not None:
        value["round_trace"] = trace
    if min_margin is not None:
        value["min_target_margin"] = min_margin
    return value


def run_ar(target, tokenizer, encoded, device, max_tokens):
    completion, rounds = [], []
    eos = {int(tokenizer.eos_token_id)}
    for _ in range(max_tokens):
        _, out, ms = target_forward(target, encoded, completion, device)
        token = int(out.logits[0, -1].argmax())
        completion.append(token)
        rounds.append({"elapsed_ms": ms})
        if token in eos:
            break
    return result(completion, rounds, sum(r["elapsed_ms"] for r in rounds))


def run_schema(target, encoded, base, positions, pairs, device):
    completion = list(base[:positions[0]])
    branches, rounds, margins = [], [], []
    for field in range(len(positions)):
        choices, local_margins, ms = target_branches(
            target, encoded, completion, [positions[field]], [field], pairs, device
        )
        token = choices[0]
        branches.append(pairs[field].index(token))
        completion.append(token)
        end = positions[field + 1] if field + 1 < len(positions) else len(base)
        completion.extend(base[positions[field] + 1:end])
        rounds.append({"elapsed_ms": ms})
        margins.extend(local_margins)
    return result(
        completion, rounds, sum(r["elapsed_ms"] for r in rounds),
        bonuses=40, branches=branches, min_margin=min(margins),
    )


def load_dflash(target, device):
    draft = DFlashDraftModel(DFlashConfig(**json.loads(CONFIG.read_text())))
    with safe_open(OFFICIAL_DFLASH / "model.safetensors", framework="pt") as handle:
        official = {key: handle.get_tensor(key) for key in handle.keys()}
    mapped = to_internal_keys(official, draft.state_dict().keys())
    loaded = draft.load_state_dict(mapped, strict=False)
    if loaded.missing_keys != ["embed_tokens.weight"] or loaded.unexpected_keys:
        raise RuntimeError(loaded)
    del official, mapped
    draft.embed_tokens.weight.data.copy_(target.get_input_embeddings().weight.detach().cpu())
    draft.freeze_embedding()
    return draft.to(device=device, dtype=torch.bfloat16).eval()


@torch.inference_mode()
def dflash_propose(draft, ids, hidden, device):
    length, block = int(ids.shape[1]), 16
    context = draft.extract_context_feature(hidden)
    noise_ids = torch.full((1, block), draft.mask_token_id, device=device, dtype=torch.long)
    noise_ids[0, 0] = ids[0, -1]
    noise = draft.embed_tokens(noise_ids)
    context_pos = torch.arange(length, device=device).unsqueeze(0)
    draft_pos = (length - 1 + torch.arange(block, device=device)).unsqueeze(0)
    dense = torch.zeros(1, 1, block, length + block, device=device, dtype=torch.bool)
    dense[..., :length - 1] = True
    dense[..., length:] = True
    sync(device)
    started = time.perf_counter()
    value = draft(
        draft_input_ids=None, context_feature=context,
        draft_position_ids=draft_pos, context_position_ids=context_pos,
        block_mask=dense, noise_embedding=noise,
    )
    logits = F.linear(value[:, 1:], draft.embed_tokens.weight).float()[0]
    sync(device)
    return logits, (time.perf_counter() - started) * 1000


@torch.inference_mode()
def verify_tokens(target, encoded, prefix, proposals, device):
    prompt_len = int(encoded["input_ids"].shape[1])
    begin = prompt_len + len(prefix) - 1
    keep = torch.arange(begin, begin + len(proposals) + 1, device=device)
    _, out, ms = target_forward(target, encoded, prefix + proposals, device, keep=keep)
    return [int(x) for x in out.logits[0].argmax(-1)], ms


def run_dflash(target, draft, tokenizer, encoded, device, max_tokens):
    completion, trace = [], []
    eos = int(tokenizer.eos_token_id)
    while len(completion) < max_tokens:
        ids, out, context_ms = target_forward(target, encoded, completion, device, hidden=True)
        hidden = [out.hidden_states[layer + 1] for layer in LAYERS]
        draft_logits, draft_ms = dflash_propose(draft, ids, hidden, device)
        proposals = [int(x) for x in draft_logits.argmax(-1)]
        verifier, verify_ms = verify_tokens(target, encoded, completion, proposals, device)
        mismatch = next((i for i, x in enumerate(proposals) if x != verifier[i]), None)
        if mismatch is None:
            committed, accepted = proposals + [verifier[len(proposals)]], len(proposals)
        else:
            committed, accepted = proposals[:mismatch] + [verifier[mismatch]], mismatch
        committed = committed[:max_tokens - len(completion)]
        if eos in committed:
            committed = committed[:committed.index(eos) + 1]
        completion.extend(committed)
        trace.append({
            "context_ms": context_ms, "draft_ms": draft_ms, "verify_ms": verify_ms,
            "elapsed_ms": context_ms + draft_ms + verify_ms,
            "target_calls": 2, "draft_calls": 1, "proposed": 15,
            "accepted": min(accepted, len(committed)), "committed": len(committed),
            "first_rejection": mismatch,
        })
        if eos in committed:
            break
    target_ms = sum(r["context_ms"] + r["verify_ms"] for r in trace)
    draft_ms = sum(r["draft_ms"] for r in trace)
    return result(
        completion, trace, target_ms, draft_ms,
        accepted=sum(r["accepted"] for r in trace), proposed=15 * len(trace),
        bonuses=sum(r["committed"] - r["accepted"] for r in trace), trace=trace,
    )


def run_dchord_zeroft(target, draft, encoded, base, positions, pairs, device):
    """Use an unchanged DFlash-B16 head as DChord's value proposer."""
    completion = list(base[:positions[0]])
    branches, trace = [-1] * len(positions), []
    field = 0
    while field < len(positions):
        start = field
        before = len(completion)
        ids, out, context_ms = target_forward(
            target, encoded, completion, device, hidden=True
        )
        hidden = [out.hidden_states[layer + 1] for layer in LAYERS]
        draft_logits, draft_ms = dflash_propose(draft, ids, hidden, device)

        covered_fields, offsets, proposals = [], [], []
        for candidate_field in range(field, len(positions)):
            offset = positions[candidate_field] - positions[field]
            if offset >= draft_logits.shape[0]:
                break
            pair = torch.tensor(pairs[candidate_field], device=device)
            token = int(pair[int(draft_logits[offset, pair].argmax())])
            covered_fields.append(candidate_field)
            offsets.append(offset)
            proposals.append(token)
        if not proposals:
            raise RuntimeError("DFlash-B16 did not cover the current value position")

        block = len(proposals)
        span, _ = candidate_span(base, positions, pairs, field, proposals)
        candidate = completion + span
        verify_fields = list(covered_fields)
        verify_positions = [positions[index] for index in verify_fields]
        if field + block < len(positions):
            verify_fields.append(field + block)
            verify_positions.append(positions[field + block])
        verifier, margins, verify_ms = target_branches(
            target, encoded, candidate, verify_positions, verify_fields,
            pairs, device,
        )
        mismatch = next(
            (index for index in range(block) if proposals[index] != verifier[index]),
            None,
        )

        bonus = correction = 0
        if mismatch is not None:
            accepted_decisions = mismatch
            rejected = field + mismatch
            rejected_pos = positions[rejected]
            accepted_target_tokens = rejected_pos - positions[field]
            completion.extend(span[:accepted_target_tokens])
            token = verifier[mismatch]
            completion.append(token)
            for offset, proposal in enumerate(proposals[:mismatch]):
                branches[field + offset] = pairs[field + offset].index(proposal)
            branches[rejected] = pairs[rejected].index(token)
            field = rejected + 1
            fixed_end = positions[field] if field < len(positions) else len(base)
            completion.extend(base[rejected_pos + 1:fixed_end])
            correction = 1
        else:
            accepted_decisions = block
            accepted_target_tokens = len(span)
            completion = candidate
            for offset, proposal in enumerate(proposals):
                branches[field + offset] = pairs[field + offset].index(proposal)
            field += block
            if field < len(positions):
                token = verifier[block]
                completion.append(token)
                branches[field] = pairs[field].index(token)
                next_field = field + 1
                fixed_end = positions[next_field] if next_field < len(positions) else len(base)
                completion.extend(base[positions[field] + 1:fixed_end])
                field = next_field
                bonus = 1

        trace.append({
            "field_start": start,
            "covered_fields": covered_fields,
            "dflash_b16_value_offsets": offsets,
            "proposed_values": proposals,
            "context_ms": context_ms,
            "draft_ms": draft_ms,
            "verify_ms": verify_ms,
            "elapsed_ms": context_ms + draft_ms + verify_ms,
            "target_calls": 2,
            "draft_calls": 1,
            "proposed_decisions": block,
            "accepted_decisions": accepted_decisions,
            "candidate_target_tokens": len(span),
            "accepted_target_tokens": accepted_target_tokens,
            "output_advance_target_tokens": len(completion) - before,
            "bonus": bonus,
            "correction": correction,
            "first_rejection": mismatch,
            "min_target_margin": min(margins),
        })

    output = result(
        completion, trace,
        sum(row["context_ms"] + row["verify_ms"] for row in trace),
        sum(row["draft_ms"] for row in trace),
        accepted=sum(row["accepted_target_tokens"] for row in trace),
        proposed=sum(row["candidate_target_tokens"] for row in trace),
        bonuses=sum(row["bonus"] + row["correction"] for row in trace),
        branches=branches,
        trace=trace,
        min_margin=min(row["min_target_margin"] for row in trace),
    )
    output.update({
        "proposed_value_decisions": sum(row["proposed_decisions"] for row in trace),
        "accepted_value_decisions": sum(row["accepted_decisions"] for row in trace),
        "accepted_target_tokens": sum(row["accepted_target_tokens"] for row in trace),
        "output_advance_target_tokens": sum(
            row["output_advance_target_tokens"] for row in trace
        ),
    })
    return output


def run_dchord(target, draft, encoded, base, positions, pairs, device):
    completion = list(base[:positions[0]])
    branches, trace = [-1] * len(positions), []
    field, prompt_len = 0, int(encoded["input_ids"].shape[1])
    while field < len(positions):
        start, block = field, min(3, len(positions) - field)
        _, out, context_ms = target_forward(target, encoded, completion, device, hidden=True)
        hidden = [out.hidden_states[layer + 1] for layer in LAYERS]
        absolute = torch.tensor(
            [[prompt_len + positions[i] for i in range(field, field + block)]],
            device=device,
        )
        sync(device)
        started = time.perf_counter()
        logits = draft.propose_block(
            hidden, field_start=field, draft_position_ids=absolute,
            lm_head_weight=draft.draft_model.embed_tokens.weight,
        )
        proposals = []
        for offset in range(block):
            pair = torch.tensor(pairs[field + offset], device=device)
            proposals.append(int(pair[int(logits[0, offset, pair].argmax())]))
        sync(device)
        draft_ms = (time.perf_counter() - started) * 1000
        span, _ = candidate_span(base, positions, pairs, field, proposals)
        candidate = completion + span
        verify_fields = list(range(field, field + block))
        verify_positions = [positions[i] for i in verify_fields]
        if field + block < len(positions):
            verify_fields.append(field + block)
            verify_positions.append(positions[field + block])
        verifier, margins, verify_ms = target_branches(
            target, encoded, candidate, verify_positions, verify_fields, pairs, device
        )
        mismatch = next((i for i in range(block) if proposals[i] != verifier[i]), None)
        bonus = correction = 0
        if mismatch is not None:
            accepted = mismatch
            rejected = field + mismatch
            rejected_pos = positions[rejected]
            completion.extend(span[:rejected_pos - positions[field]])
            token = verifier[mismatch]
            completion.append(token)
            branches[rejected] = pairs[rejected].index(token)
            for offset in range(mismatch):
                branches[field + offset] = pairs[field + offset].index(proposals[offset])
            field = rejected + 1
            fixed_end = positions[field] if field < len(positions) else len(base)
            completion.extend(base[rejected_pos + 1:fixed_end])
            correction = 1
        else:
            accepted = block
            completion = candidate
            for offset, token in enumerate(proposals):
                branches[field + offset] = pairs[field + offset].index(token)
            field += block
            if field < len(positions):
                token = verifier[block]
                completion.append(token)
                branches[field] = pairs[field].index(token)
                next_field = field + 1
                fixed_end = positions[next_field] if next_field < len(positions) else len(base)
                completion.extend(base[positions[field] + 1:fixed_end])
                field = next_field
                bonus = 1
        trace.append({
            "field_start": start, "context_ms": context_ms, "draft_ms": draft_ms,
            "verify_ms": verify_ms, "elapsed_ms": context_ms + draft_ms + verify_ms,
            "target_calls": 2, "draft_calls": 1, "proposed": block,
            "accepted": accepted, "bonus": bonus, "correction": correction,
            "first_rejection": mismatch, "min_target_margin": min(margins),
        })
    target_ms = sum(r["context_ms"] + r["verify_ms"] for r in trace)
    draft_ms = sum(r["draft_ms"] for r in trace)
    return result(
        completion, trace, target_ms, draft_ms,
        accepted=sum(r["accepted"] for r in trace),
        proposed=sum(r["proposed"] for r in trace),
        bonuses=sum(r["bonus"] + r["correction"] for r in trace),
        branches=branches, trace=trace,
        min_margin=min(r["min_target_margin"] for r in trace),
    )


def summarize(mode, records, wall):
    times = [r["generation_ms"] for r in records]
    total_ms = sum(times)
    fields = sum(r["field_total"] for r in records)
    rounds = sum(r["rounds"] for r in records)
    proposed = sum(r["proposed_draft_tokens"] for r in records)
    accepted = sum(r["accepted_draft_tokens"] for r in records)
    summary = {
        "version": "q35_torch_fourway_nocache_v1", "mode": mode,
        "protocol": {
            "framework": "PyTorch/HF Transformers", "dtype": "bfloat16",
            "attention": "sdpa", "use_cache": False, "batch_size": 1,
            "greedy": True, "thinking": False,
            "timing_excludes": ["model_load", "image_load", "processor", "json_parse"],
            "timing_includes": ["target", "draft", "verification", "recovery", "injection"],
        },
        "records": len(records),
        "mean_generation_ms": statistics.mean(times),
        "p50_generation_ms": pct(times, .5), "p95_generation_ms": pct(times, .95),
        "requests_per_second": len(records) * 1000 / total_ms,
        "output_tokens_per_second": sum(r["output_tokens"] for r in records) * 1000 / total_ms,
        "schema_fields_per_second": fields * 1000 / total_ms if fields else None,
        "mean_output_tokens": statistics.mean(r["output_tokens"] for r in records),
        "mean_rounds": rounds / len(records),
        "mean_target_calls": statistics.mean(r["target_calls"] for r in records),
        "mean_draft_calls": statistics.mean(r["draft_calls"] for r in records),
        "mean_target_ms": statistics.mean(r["target_ms"] for r in records),
        "mean_draft_ms": statistics.mean(r["draft_ms"] for r in records),
        "mean_first_step_ms": statistics.mean(r["first_step_ms"] for r in records),
        "exact_40_bool_records": sum(r["exact_40_bool"] for r in records),
        "label_accuracy": sum((r["field_correct"] or 0) for r in records) / fields if fields else None,
        "mean_accepted_draft_tokens_per_round": accepted / rounds if proposed else None,
        "draft_accept_rate": accepted / proposed if proposed else None,
        "wall_seconds_including_preprocessing": wall,
        "claim_scope": "matched no-cache algorithm puncture; not production serving throughput",
    }
    if mode == "dchord_zeroft":
        proposed_decisions = sum(r["proposed_value_decisions"] for r in records)
        accepted_decisions = sum(r["accepted_value_decisions"] for r in records)
        accepted_target_tokens = sum(r["accepted_target_tokens"] for r in records)
        output_advance = sum(r["output_tokens"] for r in records)
        summary.update({
            "definition": (
                "unchanged open-source DFlash-B16 proposer; select only schema value "
                "decisions within its 15-token horizon; compiler injects fixed structure"
            ),
            "schema_reference_exact_records": sum(
                r["schema_reference_branches_match"] for r in records
            ),
            "value_decision_accept_rate": accepted_decisions / proposed_decisions,
            "mean_proposed_value_decisions_per_round": proposed_decisions / rounds,
            "mean_accepted_value_decisions_per_round": accepted_decisions / rounds,
            "mean_accepted_target_tokens_per_round": accepted_target_tokens / rounds,
            "mean_output_advance_target_tokens_per_round": output_advance / rounds,
        })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--export", type=Path, default=EXPORT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard = f"{args.offset:03d}_{args.offset + args.limit:03d}"
    rows_path = args.output_dir / f"{args.mode}_{shard}_rows.jsonl"
    summary_path = args.output_dir / f"{args.mode}_{shard}_summary.json"
    rows_path.unlink(missing_ok=True)

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(TARGET, local_files_only=True)
    processor = AutoProcessor.from_pretrained(TARGET, local_files_only=True)
    pcs = json.loads(PCS.read_text())
    attributes = list(pcs["schema"]["attributes"])
    segments = list(pcs["surface"]["segments"])
    base, positions, pairs = build_contract(tokenizer, segments, attributes)
    samples = read_jsonl(TEST)[args.offset:args.offset + args.limit]
    prompt = PROMPT.read_text()
    target = AutoModelForImageTextToText.from_pretrained(
        TARGET, dtype=torch.bfloat16, local_files_only=True,
        low_cpu_mem_usage=True, attn_implementation="sdpa",
    ).to(device).eval()
    draft = None
    if args.mode in {"dflash_b16", "dchord_zeroft"}:
        draft = load_dflash(target, device)
    elif args.mode == "dchord_k3":
        draft = build_dchord(tokenizer, attributes, pairs, device, args.export)
    gc.collect(); torch.cuda.empty_cache()

    def infer(sample):
        encoded = processor_inputs(processor, load_image(sample), prompt)
        if args.mode == "ar":
            return run_ar(target, tokenizer, encoded, device, args.max_new_tokens)
        if args.mode == "schema":
            return run_schema(target, encoded, base, positions, pairs, device)
        if args.mode == "dflash_b16":
            return run_dflash(target, draft, tokenizer, encoded, device, args.max_new_tokens)
        if args.mode == "dchord_zeroft":
            output = run_dchord_zeroft(
                target, draft, encoded, base, positions, pairs, device
            )
            reference = run_schema(target, encoded, base, positions, pairs, device)
            output["schema_reference_branches_match"] = (
                output["branches"] == reference["branches"]
            )
            output["schema_reference_value_mismatches"] = [
                index for index, (actual, expected) in enumerate(
                    zip(output["branches"], reference["branches"], strict=True)
                ) if actual != expected
            ]
            return output
        return run_dchord(target, draft, encoded, base, positions, pairs, device)

    for index in range(min(args.warmup, len(samples))):
        print(json.dumps({"phase": "warmup", "index": index}), flush=True)
        infer(samples[index])
    records, wall_start = [], time.perf_counter()
    for index, sample in enumerate(samples):
        output = infer(sample)
        text = tokenizer.decode(output["completion_ids"], skip_special_tokens=True)
        record = {
            "index": index, "sample_id": sample["sample_id"], "mode": args.mode,
            "output_tokens": len(output["completion_ids"]), "output_text": text,
            **parse_schema(text, attributes, sample["labels"]),
            **{k: v for k, v in output.items() if k != "completion_ids"},
        }
        records.append(record)
        with rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({
            "progress": f"{index + 1}/{len(samples)}", "mode": args.mode,
            "generation_ms": round(record["generation_ms"], 3),
            "rounds": record["rounds"], "output_tokens": record["output_tokens"],
            "exact_40_bool": record["exact_40_bool"], "field_correct": record["field_correct"],
        }, ensure_ascii=False), flush=True)
    summary = summarize(args.mode, records, time.perf_counter() - wall_start)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
