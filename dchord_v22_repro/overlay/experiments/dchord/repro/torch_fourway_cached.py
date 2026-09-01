#!/usr/bin/env python3
"""Matched cached PyTorch speed test for AR, schema, DFlash-B16 and DChord-K3."""
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

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.dchord.repro.cache_transaction import (  # noqa: E402
    restore_cache, snapshot_cache,
)
from experiments.dchord.repro.common import (  # noqa: E402
    EXPORT, LAYERS, PCS, PROMPT, TARGET, TEST, build_contract, build_dchord,
    candidate_span, load_image, model_inputs, processor_inputs, read_jsonl,
)
from experiments.dchord.repro.torch_fourway_nocache import (  # noqa: E402
    load_dflash, parse_schema,
)

ROOT = Path(os.environ.get("DCHORD_REPRO_ROOT", "/root/autodl-tmp"))
DEFAULT_OUT = Path(os.environ.get(
    "DCHORD_REPRO_OUTPUT",
    str(ROOT / "reports/dchord_torch_repro/cached"),
))
MODES = ("ar", "schema", "dflash_b16", "dchord_k3")


def sync(device):
    torch.cuda.synchronize(device)


def pct(values, q):
    values = sorted(values)
    at = (len(values) - 1) * q
    lo, hi = math.floor(at), math.ceil(at)
    return values[lo] if lo == hi else values[lo] * (hi - at) + values[hi] * (at - lo)


class CachedTarget:
    def __init__(self, model, encoded, initial_completion, device, keep_hidden):
        self.model = model
        self.device = device
        self.keep_hidden = keep_hidden
        ids, mask, kwargs = model_inputs(encoded, initial_completion, device)
        sync(device)
        started = time.perf_counter()
        with torch.inference_mode():
            out = model(
                input_ids=ids, attention_mask=mask, output_hidden_states=keep_hidden,
                use_cache=True, logits_to_keep=1, return_dict=True, **kwargs,
            )
        sync(device)
        self.prefill_ms = (time.perf_counter() - started) * 1000
        self.cache = out.past_key_values
        self.total = int(ids.shape[1])
        self.next_logits = out.logits[0, -1].float()
        self.rope_delta = model.model.rope_deltas.detach().to(device=device, dtype=torch.long)
        self.history = (
            [out.hidden_states[layer + 1] for layer in LAYERS] if keep_hidden else None
        )
        self.target_ms = self.prefill_ms
        self.target_calls = 1
        self.snapshot_ms = 0.0
        self.restore_ms = 0.0

    @torch.inference_mode()
    def advance(self, tokens, hidden=None):
        if not tokens:
            raise ValueError("advance requires at least one token")
        hidden = self.keep_hidden if hidden is None else hidden
        before, after = self.total, self.total + len(tokens)
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        mask = torch.ones((1, after), dtype=torch.long, device=self.device)
        cache_position = torch.arange(before, after, device=self.device)
        position_ids = cache_position.view(1, 1, -1).expand(3, 1, -1)
        position_ids = position_ids + self.rope_delta.view(1, 1, 1)
        sync(self.device)
        started = time.perf_counter()
        out = self.model(
            input_ids=ids, attention_mask=mask, position_ids=position_ids,
            past_key_values=self.cache, cache_position=cache_position,
            output_hidden_states=hidden, use_cache=True,
            logits_to_keep=len(tokens), return_dict=True,
        )
        sync(self.device)
        elapsed = (time.perf_counter() - started) * 1000
        self.total = after
        self.target_ms += elapsed
        self.target_calls += 1
        return out, elapsed

    def append_hidden(self, out):
        if self.history is None:
            return
        self.history = [
            torch.cat([old, out.hidden_states[layer + 1]], dim=1)
            for old, layer in zip(self.history, LAYERS, strict=True)
        ]

    @torch.inference_mode()
    def checkpoint(self):
        sync(self.device)
        started = time.perf_counter()
        saved = snapshot_cache(self.cache)
        sync(self.device)
        elapsed = (time.perf_counter() - started) * 1000
        self.snapshot_ms += elapsed
        return saved, self.total, self.next_logits

    @torch.inference_mode()
    def rollback(self, checkpoint):
        saved, total, next_logits = checkpoint
        sync(self.device)
        started = time.perf_counter()
        restore_cache(self.cache, saved)
        sync(self.device)
        elapsed = (time.perf_counter() - started) * 1000
        self.restore_ms += elapsed
        self.total = total
        self.next_logits = next_logits


def base_result(session, completion, decode_ms, rounds, draft_ms=0.0, **extra):
    return {
        "completion_ids": completion,
        "prefill_ms": session.prefill_ms,
        "decode_ms": decode_ms,
        "generation_ms": session.prefill_ms + decode_ms,
        "target_ms": session.target_ms,
        "draft_ms": draft_ms,
        "snapshot_ms": session.snapshot_ms,
        "restore_ms": session.restore_ms,
        "target_calls": session.target_calls,
        "rounds": rounds,
        **extra,
    }


@torch.inference_mode()
def run_ar(model, tokenizer, encoded, device, max_tokens):
    session = CachedTarget(model, encoded, [], device, keep_hidden=False)
    completion = [int(session.next_logits.argmax())]
    eos = int(tokenizer.eos_token_id)
    sync(device); started = time.perf_counter()
    while completion[-1] != eos and len(completion) < max_tokens:
        out, _ = session.advance([completion[-1]], hidden=False)
        session.next_logits = out.logits[0, -1].float()
        completion.append(int(session.next_logits.argmax()))
    sync(device); decode_ms = (time.perf_counter() - started) * 1000
    return base_result(
        session, completion, decode_ms, len(completion), draft_calls=0,
        proposed_draft_tokens=0, accepted_draft_tokens=0,
        bonus_or_correction_tokens=0,
    )


def branch(logits, pair, device):
    pair_tensor = torch.tensor(pair, dtype=torch.long, device=device)
    scores = logits[pair_tensor].float()
    choice = int(pair_tensor[int(scores.argmax())])
    return choice, float((scores.max() - scores.min()).item())


@torch.inference_mode()
def run_schema(model, encoded, base, positions, pairs, device):
    prelude = list(base[:positions[0]])
    session = CachedTarget(model, encoded, prelude, device, keep_hidden=False)
    completion, branches, margins = list(prelude), [], []
    sync(device); started = time.perf_counter()
    for field in range(len(positions)):
        token, margin = branch(session.next_logits, pairs[field], device)
        branches.append(pairs[field].index(token)); margins.append(margin)
        completion.append(token)
        if field + 1 == len(positions):
            completion.extend(base[positions[field] + 1:])
            break
        fixed = list(base[positions[field] + 1:positions[field + 1]])
        completion.extend(fixed)
        out, _ = session.advance([token] + fixed, hidden=False)
        session.next_logits = out.logits[0, -1].float()
    sync(device); decode_ms = (time.perf_counter() - started) * 1000
    return base_result(
        session, completion, decode_ms, len(positions), branches=branches,
        min_target_margin=min(margins), draft_calls=0,
        proposed_draft_tokens=0, accepted_draft_tokens=0,
        bonus_or_correction_tokens=len(branches),
    )


@torch.inference_mode()
def cached_dflash_propose(draft, session, last_token, device):
    length, block = session.total, 16
    sync(device); started = time.perf_counter()
    context = draft.extract_context_feature(session.history)
    noise_ids = torch.full((1, block), draft.mask_token_id, device=device, dtype=torch.long)
    noise_ids[0, 0] = last_token
    noise = draft.embed_tokens(noise_ids)
    context_pos = torch.arange(length, device=device).unsqueeze(0)
    draft_pos = (length - 1 + torch.arange(block, device=device)).unsqueeze(0)
    dense = torch.zeros(1, 1, block, length + block, device=device, dtype=torch.bool)
    dense[..., :length - 1] = True
    dense[..., length:] = True
    value = draft(
        draft_input_ids=None, context_feature=context,
        draft_position_ids=draft_pos, context_position_ids=context_pos,
        block_mask=dense, noise_embedding=noise,
    )
    proposals = F.linear(value[:, 1:], draft.embed_tokens.weight).float().argmax(-1)[0]
    sync(device)
    return [int(x) for x in proposals], (time.perf_counter() - started) * 1000


@torch.inference_mode()
def run_dflash(model, draft, tokenizer, encoded, device, max_tokens):
    session = CachedTarget(model, encoded, [], device, keep_hidden=True)
    completion, trace, draft_ms = [], [], 0.0
    eos = int(tokenizer.eos_token_id)
    sync(device); started = time.perf_counter()
    while len(completion) < max_tokens:
        last_token = int(encoded["input_ids"][0, -1]) if not completion else completion[-1]
        proposals, local_draft_ms = cached_dflash_propose(
            draft, session, last_token, device
        )
        draft_ms += local_draft_ms
        checkpoint = session.checkpoint()
        base_next = session.next_logits
        verify_out, verify_ms = session.advance(proposals, hidden=True)
        verifier = [int(base_next.argmax())]
        verifier.extend(int(x) for x in verify_out.logits[0, :-1].argmax(-1))
        mismatch = next((i for i, x in enumerate(proposals) if x != verifier[i]), None)
        if mismatch is None:
            accepted = len(proposals)
            committed = list(proposals)
            if eos in committed:
                committed = committed[:committed.index(eos) + 1]
                completion.extend(committed)
                bonus = 0
            else:
                session.append_hidden(verify_out)
                session.next_logits = verify_out.logits[0, -1].float()
                bonus_token = int(session.next_logits.argmax())
                completion.extend(committed)
                completion.append(bonus_token)
                bonus_out, bonus_ms = session.advance([bonus_token], hidden=True)
                session.append_hidden(bonus_out)
                session.next_logits = bonus_out.logits[0, -1].float()
                bonus = 1
        else:
            accepted = mismatch
            correction = verifier[mismatch]
            committed = proposals[:mismatch] + [correction]
            if eos in committed:
                committed = committed[:committed.index(eos) + 1]
            session.rollback(checkpoint)
            replay_out, replay_ms = session.advance(committed, hidden=True)
            session.append_hidden(replay_out)
            session.next_logits = replay_out.logits[0, -1].float()
            completion.extend(committed)
            bonus = 1
        trace.append({
            "proposed": 15, "accepted": min(accepted, len(committed)),
            "committed": len(committed) + (bonus if mismatch is None and eos not in committed else 0),
            "first_rejection": mismatch, "verify_ms": verify_ms,
            "draft_ms": local_draft_ms,
        })
        if eos in completion or len(completion) >= max_tokens:
            completion = completion[:max_tokens]
            break
    sync(device); decode_ms = (time.perf_counter() - started) * 1000
    return base_result(
        session, completion, decode_ms, len(trace), draft_ms=draft_ms,
        draft_calls=len(trace), proposed_draft_tokens=15 * len(trace),
        accepted_draft_tokens=sum(r["accepted"] for r in trace),
        bonus_or_correction_tokens=sum(r["committed"] - r["accepted"] for r in trace),
        round_trace=trace,
    )


def verifier_branches(base_next, verify_out, span_start, positions, fields, pairs, device):
    choices, margins = [], []
    for field in fields:
        relative = positions[field] - span_start
        logits = base_next if relative == 0 else verify_out.logits[0, relative - 1].float()
        token, margin = branch(logits, pairs[field], device)
        choices.append(token); margins.append(margin)
    return choices, margins


@torch.inference_mode()
def run_dchord(model, draft, encoded, base, positions, pairs, device, guard_margin):
    prelude = list(base[:positions[0]])
    session = CachedTarget(model, encoded, prelude, device, keep_hidden=True)
    completion, branches, trace = list(prelude), [-1] * len(positions), []
    field, draft_ms = 0, 0.0
    sync(device); started = time.perf_counter()
    while field < len(positions):
        field_start, block = field, min(3, len(positions) - field)
        absolute = torch.tensor(
            [[session.total + positions[i] - positions[field] for i in range(field, field + block)]],
            device=device,
        )
        sync(device); draft_started = time.perf_counter()
        logits = draft.propose_block(
            session.history, field_start=field, draft_position_ids=absolute,
            lm_head_weight=draft.draft_model.embed_tokens.weight,
        )
        proposals = []
        for offset in range(block):
            token, _ = branch(logits[0, offset], pairs[field + offset], device)
            proposals.append(token)
        sync(device); local_draft_ms = (time.perf_counter() - draft_started) * 1000
        draft_ms += local_draft_ms
        span, _ = candidate_span(base, positions, pairs, field, proposals)
        checkpoint = session.checkpoint()
        base_next = session.next_logits
        verify_out, verify_ms = session.advance(span, hidden=True)
        verify_fields = list(range(field, field + block))
        if field + block < len(positions):
            verify_fields.append(field + block)
        verifier, margins = verifier_branches(
            base_next, verify_out, positions[field], positions, verify_fields,
            pairs, device,
        )
        draft_mismatch = next((i for i in range(block) if proposals[i] != verifier[i]), None)
        mismatch = draft_mismatch
        guarded = mismatch is not None and margins[mismatch] <= guard_margin
        if mismatch is None:
            accepted = block
            completion.extend(span)
            session.append_hidden(verify_out)
            session.next_logits = verify_out.logits[0, -1].float()
            for offset, token in enumerate(proposals):
                branches[field + offset] = pairs[field + offset].index(token)
            field += block
            bonus = 0
            if field < len(positions):
                token = verifier[block]
                branches[field] = pairs[field].index(token)
                next_field = field + 1
                fixed_end = positions[next_field] if next_field < len(positions) else len(base)
                commit = [token] + list(base[positions[field] + 1:fixed_end])
                completion.extend(commit)
                bonus_out, _ = session.advance(commit, hidden=True)
                session.append_hidden(bonus_out)
                session.next_logits = bonus_out.logits[0, -1].float()
                field = next_field
                bonus = 1
            correction = 0
        else:
            accepted = mismatch
            rejected = field + mismatch
            rejected_pos = positions[rejected]
            relative = rejected_pos - positions[field]
            correction_token = verifier[mismatch]
            next_field = rejected + 1
            fixed_end = positions[next_field] if next_field < len(positions) else len(base)
            session.rollback(checkpoint)
            prefix_commit = list(span[:relative])
            if guarded:
                if prefix_commit:
                    prefix_out, _ = session.advance(prefix_commit, hidden=True)
                    session.append_hidden(prefix_out)
                    session.next_logits = prefix_out.logits[0, -1].float()
                correction_token, _ = branch(
                    session.next_logits, pairs[rejected], device
                )
                suffix_commit = [correction_token]
                suffix_commit.extend(base[rejected_pos + 1:fixed_end])
                suffix_out, _ = session.advance(suffix_commit, hidden=True)
                session.append_hidden(suffix_out)
                session.next_logits = suffix_out.logits[0, -1].float()
                commit = prefix_commit + suffix_commit
            else:
                commit = prefix_commit + [correction_token]
                commit.extend(base[rejected_pos + 1:fixed_end])
                replay_out, _ = session.advance(commit, hidden=True)
                session.append_hidden(replay_out)
                session.next_logits = replay_out.logits[0, -1].float()
            completion.extend(commit)
            for offset in range(mismatch):
                branches[field + offset] = pairs[field + offset].index(proposals[offset])
            branches[rejected] = pairs[rejected].index(correction_token)
            field = next_field
            bonus, correction = 0, 1
        trace.append({
            "field_start": field_start, "proposed": block, "accepted": accepted,
            "bonus": bonus, "correction": correction,
            "first_rejection": mismatch, "draft_mismatch": draft_mismatch,
            "guarded_rejection": guarded, "draft_ms": local_draft_ms,
            "verify_ms": verify_ms, "min_target_margin": min(margins),
        })
    sync(device); decode_ms = (time.perf_counter() - started) * 1000
    return base_result(
        session, completion, decode_ms, len(trace), draft_ms=draft_ms,
        branches=branches, draft_calls=len(trace),
        proposed_draft_tokens=sum(r["proposed"] for r in trace),
        accepted_draft_tokens=sum(r["accepted"] for r in trace),
        bonus_or_correction_tokens=sum(r["bonus"] + r["correction"] for r in trace),
        guarded_rejections=sum(r["guarded_rejection"] for r in trace),
        min_target_margin=min(r["min_target_margin"] for r in trace),
        round_trace=trace,
    )


def summarize(mode, records, wall):
    generation = [r["generation_ms"] for r in records]
    decode = [r["decode_ms"] for r in records]
    total_generation, total_decode = sum(generation), sum(decode)
    fields = sum(r["field_total"] for r in records)
    rounds = sum(r["rounds"] for r in records)
    proposed = sum(r["proposed_draft_tokens"] for r in records)
    accepted = sum(r["accepted_draft_tokens"] for r in records)
    return {
        "version": "q35_torch_fourway_cached_v1", "mode": mode,
        "protocol": {
            "framework": "PyTorch/HF Transformers", "dtype": "bfloat16",
            "attention": "sdpa", "use_cache": True, "batch_size": 1,
            "greedy": True, "thinking": False,
            "hybrid_state_transaction": "checkpoint/commit with rollback and accepted-prefix replay",
        },
        "records": len(records),
        "mean_prefill_ms": statistics.mean(r["prefill_ms"] for r in records),
        "mean_decode_ms": statistics.mean(decode),
        "p50_decode_ms": pct(decode, .5), "p95_decode_ms": pct(decode, .95),
        "mean_generation_ms": statistics.mean(generation),
        "requests_per_second_decode": len(records) * 1000 / total_decode,
        "requests_per_second_generation": len(records) * 1000 / total_generation,
        "output_tokens_per_second_decode": sum(r["output_tokens"] for r in records) * 1000 / total_decode,
        "schema_fields_per_second_decode": fields * 1000 / total_decode if fields else None,
        "mean_output_tokens": statistics.mean(r["output_tokens"] for r in records),
        "mean_rounds": rounds / len(records),
        "mean_target_calls": statistics.mean(r["target_calls"] for r in records),
        "mean_target_ms": statistics.mean(r["target_ms"] for r in records),
        "mean_draft_ms": statistics.mean(r["draft_ms"] for r in records),
        "mean_snapshot_ms": statistics.mean(r["snapshot_ms"] for r in records),
        "mean_restore_ms": statistics.mean(r["restore_ms"] for r in records),
        "exact_40_bool_records": sum(r["exact_40_bool"] for r in records),
        "label_accuracy": sum((r["field_correct"] or 0) for r in records) / fields if fields else None,
        "mean_accepted_draft_tokens_per_round": accepted / rounds if proposed else None,
        "draft_accept_rate": accepted / proposed if proposed else None,
        "wall_seconds_including_preprocessing": wall,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--guard-margin", type=float, default=0.125)
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
    model = AutoModelForImageTextToText.from_pretrained(
        TARGET, dtype=torch.bfloat16, local_files_only=True,
        low_cpu_mem_usage=True, attn_implementation="sdpa",
    ).to(device).eval()
    draft = None
    if args.mode == "dflash_b16":
        draft = load_dflash(model, device)
    elif args.mode == "dchord_k3":
        draft = build_dchord(tokenizer, attributes, pairs, device, args.export)
    gc.collect(); torch.cuda.empty_cache()

    def infer(sample):
        encoded = processor_inputs(processor, load_image(sample), prompt)
        if args.mode == "ar":
            return run_ar(model, tokenizer, encoded, device, args.max_new_tokens)
        if args.mode == "schema":
            return run_schema(model, encoded, base, positions, pairs, device)
        if args.mode == "dflash_b16":
            return run_dflash(model, draft, tokenizer, encoded, device, args.max_new_tokens)
        return run_dchord(
            model, draft, encoded, base, positions, pairs, device,
            args.guard_margin,
        )

    for index in range(min(args.warmup, len(samples))):
        print(json.dumps({"phase": "warmup", "index": index}), flush=True)
        infer(samples[index])
    records, wall_started = [], time.perf_counter()
    for index, sample in enumerate(samples):
        output = infer(sample)
        text = tokenizer.decode(output["completion_ids"], skip_special_tokens=True)
        record = {
            "index": args.offset + index, "sample_id": sample["sample_id"],
            "mode": args.mode, "output_tokens": len(output["completion_ids"]),
            "output_text": text, **parse_schema(text, attributes, sample["labels"]),
            **{k: v for k, v in output.items() if k != "completion_ids"},
        }
        records.append(record)
        with rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({
            "progress": f"{index + 1}/{len(samples)}", "mode": args.mode,
            "prefill_ms": round(record["prefill_ms"], 3),
            "decode_ms": round(record["decode_ms"], 3),
            "rounds": record["rounds"], "output_tokens": record["output_tokens"],
            "exact_40_bool": record["exact_40_bool"],
        }, ensure_ascii=False), flush=True)
    summary = summarize(args.mode, records, time.perf_counter() - wall_started)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
