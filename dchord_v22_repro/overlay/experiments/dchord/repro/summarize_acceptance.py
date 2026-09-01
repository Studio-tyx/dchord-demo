#!/usr/bin/env python3
"""Summarize the matched 128-sample P1.H acceptance/component run."""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from pathlib import Path

from transformers import AutoTokenizer


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def overlap(span, regions):
    start, end = span
    return any(start < right and end > left for left, right in regions)


def rejection_surface_breakdown(rows, tokenizer):
    counts = {"value": 0, "key": 0, "separator_or_format": 0, "terminal": 0}
    mismatch_histogram = {str(index): 0 for index in range(15)}
    rejection_events = 0
    full_accept_rounds = 0
    for row in rows:
        text = row["output_text"]
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        value_regions = [match.span(1) for match in re.finditer(r":\s*(true|false)", text)]
        key_regions = [match.span(0) for match in re.finditer(r'"[^"\\]*(?:\\.[^"\\]*)*"\s*:', text)]
        cursor = 0
        for trace in row["round_trace"]:
            mismatch = trace["first_rejection"]
            if mismatch is None:
                full_accept_rounds += 1
            else:
                rejection_events += 1
                mismatch_histogram[str(mismatch)] += 1
                absolute = cursor + int(mismatch)
                if absolute >= len(offsets):
                    counts["terminal"] += 1
                elif overlap(offsets[absolute], value_regions):
                    counts["value"] += 1
                elif overlap(offsets[absolute], key_regions):
                    counts["key"] += 1
                else:
                    counts["separator_or_format"] += 1
            cursor += int(trace["committed"])
    return {
        "rejection_events": rejection_events,
        "full_accept_rounds": full_accept_rounds,
        "first_rejection_surface_counts": counts,
        "first_rejection_surface_rates": {
            key: value / rejection_events if rejection_events else None
            for key, value in counts.items()
        },
        "mismatch_index_histogram": mismatch_histogram,
    }


def component_summary(rows):
    decode = sum(row["decode_ms"] for row in rows)
    prefill = sum(row["prefill_ms"] for row in rows)
    target = sum(row["target_ms"] for row in rows)
    verify = sum(
        trace["verify_ms"] for row in rows for trace in row["round_trace"]
    )
    draft = sum(row["draft_ms"] for row in rows)
    snapshot = sum(row["snapshot_ms"] for row in rows)
    restore = sum(row["restore_ms"] for row in rows)
    nonverify_target = target - prefill - verify
    rounds = sum(row["rounds"] for row in rows)
    return {
        "mean_decode_ms_raw_safe_adapter": decode / len(rows),
        "mean_target_verify_ms": verify / len(rows),
        "mean_draft_ms": draft / len(rows),
        "mean_snapshot_ms": snapshot / len(rows),
        "mean_restore_ms": restore / len(rows),
        "mean_nonverify_target_advance_ms": nonverify_target / len(rows),
        "target_verify_ms_per_round": verify / rounds,
        "draft_ms_per_round": draft / rounds,
        "verify_plus_draft_ms_per_round": (verify + draft) / rounds,
        "note": (
            "nonverify_target_advance includes accepted-prefix replay and/or separate "
            "bonus/state-commit forwards; it is an optimization target, not all removable work"
        ),
    }


def quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def aggregate(rows):
    rounds = sum(row["rounds"] for row in rows)
    proposed = sum(row["proposed_draft_tokens"] for row in rows)
    accepted = sum(row["accepted_draft_tokens"] for row in rows)
    output = sum(row["output_tokens"] for row in rows)
    verify = sum(trace["verify_ms"] for row in rows for trace in row["round_trace"])
    draft = sum(row["draft_ms"] for row in rows)
    return {
        "accepted_per_round": accepted / rounds,
        "accept_rate": accepted / proposed,
        "output_advance_per_round": output / rounds,
        "verify_plus_draft_ms_per_round": (verify + draft) / rounds,
        "decode_ms": sum(row["decode_ms"] for row in rows),
    }


def paired_bootstrap(dflash, dchord, repetitions=10_000, seed=20260824):
    rng = random.Random(seed)
    metrics = {
        "dflash_accepted_per_round": [],
        "dchord_accepted_per_round": [],
        "complete_output_advance_ratio_dchord_over_dflash": [],
        "verify_plus_draft_cost_ratio_dchord_over_dflash": [],
        "raw_safe_adapter_speed_ratio_dchord_over_dflash": [],
    }
    for _ in range(repetitions):
        indices = [rng.randrange(len(dflash)) for _ in range(len(dflash))]
        left = aggregate([dflash[index] for index in indices])
        right = aggregate([dchord[index] for index in indices])
        metrics["dflash_accepted_per_round"].append(left["accepted_per_round"])
        metrics["dchord_accepted_per_round"].append(right["accepted_per_round"])
        metrics["complete_output_advance_ratio_dchord_over_dflash"].append(
            right["output_advance_per_round"] / left["output_advance_per_round"]
        )
        metrics["verify_plus_draft_cost_ratio_dchord_over_dflash"].append(
            right["verify_plus_draft_ms_per_round"]
            / left["verify_plus_draft_ms_per_round"]
        )
        metrics["raw_safe_adapter_speed_ratio_dchord_over_dflash"].append(
            left["decode_ms"] / right["decode_ms"]
        )
    return {
        key: {"low_95": quantile(values, .025), "high_95": quantile(values, .975)}
        for key, values in metrics.items()
    }


def method_summary(rows):
    rounds = sum(row["rounds"] for row in rows)
    proposed = sum(row["proposed_draft_tokens"] for row in rows)
    accepted = sum(row["accepted_draft_tokens"] for row in rows)
    output = sum(row["output_tokens"] for row in rows)
    return {
        "records": len(rows),
        "exact_40_bool_records": sum(bool(row["exact_40_bool"]) for row in rows),
        "mean_rounds": rounds / len(rows),
        "mean_proposed_units_per_round": proposed / rounds,
        "mean_accepted_units_per_round": accepted / rounds,
        "draft_accept_rate": accepted / proposed,
        "mean_complete_output_tokens_advanced_per_round": output / rounds,
        "mean_bonus_or_correction_units_per_round": sum(
            row["bonus_or_correction_tokens"] for row in rows
        ) / rounds,
        "label_accuracy": sum(row["field_correct"] for row in rows) / sum(
            row["field_total"] for row in rows
        ),
        "component_diagnostics": component_summary(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dflash = read_jsonl(args.result_dir / "dflash_b16_000_128_rows.jsonl")
    dchord = read_jsonl(args.result_dir / "dchord_k3_000_128_rows.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    dflash_summary = method_summary(dflash)
    dchord_summary = method_summary(dchord)
    advance_ratio = (
        dchord_summary["mean_complete_output_tokens_advanced_per_round"]
        / dflash_summary["mean_complete_output_tokens_advanced_per_round"]
    )
    result = {
        "version": "p1h_acceptance_component_128_v1",
        "comparison_scope": "same frozen 128 CelebA samples; same GPU; serial; batch 1",
        "dflash_b16": dflash_summary,
        "dchord_k3": dchord_summary,
        "dchord_over_dflash_complete_output_advance_per_verify_round": advance_ratio,
        "paired_bootstrap_10000": paired_bootstrap(dflash, dchord),
        "dflash_first_rejection_surface": rejection_surface_breakdown(dflash, tokenizer),
        "interpretation": (
            "accepted units differ: DFlash uses raw tokens and DChord uses value decisions. "
            "The comparable algorithmic quantity is complete output tokens advanced per target verify round. "
            "Raw Torch adapter latency is diagnostic and is not production serving speed."
        ),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
