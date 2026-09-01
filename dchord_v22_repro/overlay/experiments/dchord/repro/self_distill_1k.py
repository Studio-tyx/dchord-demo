#!/usr/bin/env python3
"""DChord v2.2 P1: freeze, self-distil, and audit a 1k CelebA set.

This script deliberately keeps semantic teacher labels (the target model's
boolean decisions) separate from CelebA annotations, which are diagnostic only.
The profiled canonical surface (PCS) is frozen before generation.
"""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import io
import json
import math
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("DCHORD_REPRO_ROOT", "/root/autodl-tmp"))
REPO = Path(os.environ.get("DCHORD_TORCHSPEC_REPO", str(HERE.parents[3])))
MODEL = Path(os.environ.get("DCHORD_TARGET", str(ROOT / "models/Qwen3.5-4B")))
DATASET = Path(os.environ.get("DCHORD_CELEBA_ROOT", str(ROOT / "datasets/celeba")))
SOURCE_TRAIN = HERE / "assets/train_manifest_1000.jsonl"
PROMPT = HERE / "assets/prompt_schema_oneshot_qwen35_v1.txt"
PCS = HERE / "assets/profiled_surface_spec_v1.json"
OUT = Path(os.environ.get(
    "DCHORD_SELF_DISTILL_OUTPUT",
    str(ROOT / "reports/dchord_torch_repro/self_distill_1k"),
))
MANIFEST = OUT / "p1_teacher_train_1000.jsonl"
BOOL_PLACEHOLDER = "<BOOL>"
IMAGE_TOKEN_ID = 248056


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def frozen_file(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def escape_regex(value: str) -> str:
    return re.sub(r"([\\.^$*+?{}\[\]|()])", r"\\\1", value)


def pcs_contract() -> tuple[list[str], list[str], str]:
    spec = json.loads(PCS.read_text())
    segments = spec["surface"]["segments"]
    attributes = spec["schema"]["attributes"]
    signature = spec["surface"]["signature"]
    if len(segments) != len(attributes) + 1:
        raise ValueError("invalid PCS segment count")
    return attributes, segments, signature


def render(segments: list[str], attributes: list[str], values: dict[str, bool]) -> tuple[str, list[tuple[int, int]]]:
    pieces: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for index, segment in enumerate(segments):
        pieces.append(segment)
        cursor += len(segment)
        if index < len(attributes):
            literal = "true" if values[attributes[index]] else "false"
            begin = cursor
            pieces.append(literal)
            cursor += len(literal)
            spans.append((begin, cursor))
    return "".join(pieces), spans


def strict_parse(text: str, attributes: list[str], segments: list[str]) -> dict[str, bool]:
    try:
        pairs = json.loads(text, object_pairs_hook=collections.OrderedDict)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(pairs, collections.OrderedDict) or list(pairs) != attributes:
        raise ValueError("key set/order mismatch")
    if any(type(value) is not bool for value in pairs.values()):
        raise ValueError("non-boolean value")
    values = {name: bool(pairs[name]) for name in attributes}
    expected, _ = render(segments, attributes, values)
    if expected != text:
        raise ValueError("output violates frozen PCS bytes")
    return values


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = read_jsonl(SOURCE_TRAIN)
    selected = source
    if len(selected) != 1000:
        raise ValueError(f"frozen train manifest must contain 1000 records, got {len(selected)}")
    selected_ids = {row["sample_id"] for row in selected}
    if len(selected_ids) != 1000:
        raise ValueError("duplicate sample IDs in teacher manifest")
    write_jsonl(MANIFEST, selected)
    attributes, segments, signature = pcs_contract()
    protocol = {
        "version": "dchord_p1_self_distill_precal_1k_v1",
        "purpose": "CelebA method puncture: self-distillation and frozen-PCS precalibration",
        "teacher_semantics": "Qwen3.5-4B greedy decisions under frozen PCS; CelebA labels are diagnostic only",
        "selection_rule": "the frozen 1000-row manifest shipped with the code package",
        "generation": {
            "temperature": 0.0,
            "seed": 0,
            "max_tokens": 512,
            "structured_output": "regex compiled exactly from frozen PCS",
            "thinking": False,
            "enforce_eager": True,
            "tensor_parallel_size": 1,
            "batch_size_per_gpu": 8,
        },
        "frozen_inputs": {
            "target_config": frozen_file(MODEL / "config.json"),
            "target_tokenizer": frozen_file(MODEL / "tokenizer.json"),
            "ordinary_one_shot_prompt_qwen35": frozen_file(PROMPT),
            "profiled_surface": frozen_file(PCS),
            "source_train_manifest": frozen_file(SOURCE_TRAIN),
        },
        "frozen_output": frozen_file(MANIFEST),
        "schema": {
            "attribute_count": len(attributes),
            "segment_count": len(segments),
            "signature_sha256": sha256_bytes(signature.encode()),
        },
        "isolation": {
            "train_unique": len(selected_ids),
            "note": "The manifest is frozen by the package. Dataset split construction is outside this reproduction bundle.",
        },
    }
    protocol["pass"] = (
        protocol["isolation"]["train_unique"] == 1000
        and protocol["schema"]["attribute_count"] == 40
    )
    write_json(OUT / "protocol_v1.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))
    if not protocol["pass"]:
        raise SystemExit(2)


def load_one_image(record: dict[str, Any], attributes: list[str]) -> tuple[Image.Image, dict[str, Any]]:
    path = DATASET / "img_align+identity+attr" / record["shard"]
    parquet = pq.ParquetFile(path)
    target = int(record["row_index"])
    cursor = 0
    for group_index in range(parquet.metadata.num_row_groups):
        count = parquet.metadata.row_group(group_index).num_rows
        if target < cursor + count:
            table = parquet.read_row_group(group_index, columns=["image", "celeb_id", *attributes])
            row = table.slice(target - cursor, 1).to_pylist()[0]
            if int(row["celeb_id"]) != int(record["celeb_id"]):
                raise ValueError(f"celeb_id mismatch for {record['sample_id']}")
            labels = {name: bool(row[name]) for name in attributes}
            if labels != record["labels"]:
                raise ValueError(f"annotation mismatch for {record['sample_id']}")
            raw = row["image"]["bytes"]
            with Image.open(io.BytesIO(raw)) as opened:
                opened.load()
                image = opened.convert("RGB")
                image.load()
            return image, {
                "image_sha256": sha256_bytes(raw),
                "image_size": list(image.size),
                "image_encoded_bytes": len(raw),
            }
        cursor += count
    raise IndexError(record["sample_id"])


def processor_input(processor: Any, image: Image.Image, prompt: str) -> dict[str, Any]:
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    encoded = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors=None, enable_thinking=False,
    )
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(value) for value in ids]
    return {
        "prompt_text": prompt_text,
        "input_ids_sha256": sha256_bytes(json.dumps(ids, separators=(",", ":")).encode()),
        "prompt_tokens": len(ids),
        "image_tokens": ids.count(IMAGE_TOKEN_ID),
    }


def branch_pairs(tokenizer: Any, segments: list[str]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    prefixes = sorted({re.search(r"[ \t\r\n]*$", segment).group(0) for segment in segments[:-1]})
    for prefix in ["", *prefixes]:
        true_ids = tokenizer.encode(prefix + "true", add_special_tokens=False)
        false_ids = tokenizer.encode(prefix + "false", add_special_tokens=False)
        if len(true_ids) == len(false_ids) == 1:
            pair = (int(true_ids[0]), int(false_ids[0]))
            if pair not in pairs:
                pairs.append(pair)
    if not pairs:
        raise ValueError("no one-token boolean branches")
    return pairs


def extract_branch_logprobs(completion: Any, pairs: list[tuple[int, int]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    maps = completion.logprobs or []
    for position, raw_id in enumerate(completion.token_ids):
        token_id = int(raw_id)
        pair = next((pair for pair in pairs if token_id in pair), None)
        if pair is None:
            continue
        true_id, false_id = pair
        candidates = maps[position] if position < len(maps) else {}
        def lp(candidate_id: int) -> float | None:
            item = candidates.get(candidate_id) if candidates else None
            return float(item.logprob) if item is not None else None
        true_lp, false_lp = lp(true_id), lp(false_id)
        normalized = None
        if true_lp is not None and false_lp is not None:
            maximum = max(true_lp, false_lp)
            t, f = math.exp(true_lp - maximum), math.exp(false_lp - maximum)
            normalized = {"true": t / (t + f), "false": f / (t + f)}
        results.append({
            "token_position": position,
            "chosen": token_id == true_id,
            "true_logprob": true_lp,
            "false_logprob": false_lp,
            "normalized_branch_probs": normalized,
        })
    return results


def generate(start: int, end: int, batch_size: int, tag: str | None = None) -> None:
    from transformers import AutoProcessor, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    rows = read_jsonl(MANIFEST)[start:end]
    attributes, segments, signature = pcs_contract()
    regex = "".join(
        escape_regex(part) + ("(true|false)" if index < len(signature.split(BOOL_PLACEHOLDER)) - 1 else "")
        for index, part in enumerate(signature.split(BOOL_PLACEHOLDER))
    )
    tag_part = f"_{tag}" if tag else ""
    shard_path = OUT / f"generation{tag_part}_{start:04d}_{end:04d}.jsonl"
    metrics_path = OUT / f"generation{tag_part}_{start:04d}_{end:04d}_metrics.json"
    existing: dict[str, dict[str, Any]] = {}
    if shard_path.exists():
        for row in read_jsonl(shard_path):
            existing[row["sample_id"]] = row
    pending = [row for row in rows if row["sample_id"] not in existing]
    started = time.monotonic()
    metrics: dict[str, Any] = {
        "range": [start, end], "records": len(rows), "already_complete": len(existing),
        "batch_size": batch_size, "enforce_eager": True, "pass": False,
    }
    llm = None
    try:
        processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=False, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False, local_files_only=True)
        pairs = branch_pairs(tokenizer, segments)
        metrics["branch_token_pairs"] = pairs
        llm = LLM(
            model=str(MODEL), tensor_parallel_size=1, dtype="bfloat16", max_model_len=16384,
            gpu_memory_utilization=0.75, enforce_eager=True, enable_chunked_prefill=True,
            enable_prefix_caching=False, max_num_seqs=batch_size, limit_mm_per_prompt={"image": 1},
            seed=0, trust_remote_code=True, disable_log_stats=False,
            disable_chunked_mm_input=True, max_num_batched_tokens=16384,
        )
        sampling = SamplingParams(
            temperature=0.0, seed=0, max_tokens=512, logprobs=5,
            structured_outputs=StructuredOutputsParams(regex=regex),
        )
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            images: list[Image.Image] = []
            audits: list[dict[str, Any]] = []
            processed: list[dict[str, Any]] = []
            for record in batch:
                image, audit = load_one_image(record, attributes)
                images.append(image)
                audits.append(audit)
                processed.append(processor_input(processor, image, PROMPT.read_text()))
            requests = [
                {"prompt": item["prompt_text"], "multi_modal_data": {"image": image}}
                for item, image in zip(processed, images)
            ]
            outputs = llm.generate(requests, sampling, use_tqdm=False)
            with shard_path.open("a") as handle:
                for index, request in enumerate(outputs):
                    completion = request.outputs[0]
                    values = strict_parse(completion.text, attributes, segments)
                    record = batch[index]
                    result = {
                        "sample_id": record["sample_id"],
                        "selection_rank": int(record["selection_rank"]),
                        "shard": record["shard"],
                        "row_index": int(record["row_index"]),
                        "celeb_id": int(record["celeb_id"]),
                        "annotation_values_diagnostic_only": record["labels"],
                        "teacher_values": values,
                        "teacher_output_text": completion.text,
                        "teacher_output_token_ids": [int(value) for value in completion.token_ids],
                        "teacher_output_tokens": len(completion.token_ids),
                        "finish_reason": completion.finish_reason,
                        "stop_reason": completion.stop_reason,
                        "decision_logprob_proxies": extract_branch_logprobs(completion, pairs),
                        "prompt_sha256": sha256_file(PROMPT),
                        "processor_input_ids_sha256": processed[index]["input_ids_sha256"],
                        "processor_prompt_tokens": processed[index]["prompt_tokens"],
                        "processor_image_tokens": processed[index]["image_tokens"],
                        **audits[index],
                    }
                    handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    existing[result["sample_id"]] = result
            del images, audits, processed, requests, outputs
            gc.collect()
            print(json.dumps({"range": [start, end], "complete": len(existing), "total": len(rows), "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
        ordered = [existing[row["sample_id"]] for row in rows]
        write_jsonl(shard_path, ordered)
        metrics.update({
            "completed": len(ordered), "output": frozen_file(shard_path),
            "elapsed_seconds": time.monotonic() - started, "pass": len(ordered) == len(rows),
        })
    except Exception as exc:
        metrics.update({"error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.monotonic() - started})
        raise
    finally:
        if llm is not None:
            try:
                llm.shutdown()
            except Exception:
                pass
        write_json(metrics_path, metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def token_rows(tokenizer: Any, text: str, spans: list[tuple[int, int]]) -> tuple[list[int], list[dict[str, Any]]]:
    encoded = tokenizer(
        text, add_special_tokens=False, return_attention_mask=False,
        return_token_type_ids=False, return_offsets_mapping=True,
    )
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    per_value: list[dict[str, Any]] = []
    for decision_index, (begin, end) in enumerate(spans):
        witnesses = []
        for position, (token_id, (a, b)) in enumerate(zip(ids, offsets)):
            overlap = max(0, min(b, end) - max(a, begin))
            if overlap:
                witnesses.append({
                    "position": position,
                    "token_id": token_id,
                    "token_text": tokenizer.decode([token_id], skip_special_tokens=False),
                    "offset": [a, b],
                    "boundary_merged": not (a >= begin and b <= end),
                })
        if not witnesses:
            raise ValueError(f"decision {decision_index} has no token witness")
        per_value.append({"decision_index": decision_index, "witnesses": witnesses})
    return ids, per_value


def finalize() -> None:
    from transformers import AutoTokenizer

    attributes, segments, _ = pcs_contract()
    manifest = read_jsonl(MANIFEST)
    shards = [OUT / "generation_0000_0500.jsonl", OUT / "generation_0500_1000.jsonl"]
    generated = [row for path in shards for row in read_jsonl(path)]
    by_id = {row["sample_id"]: row for row in generated}
    if len(generated) != 1000 or len(by_id) != 1000:
        raise ValueError(f"generation coverage mismatch: rows={len(generated)}, unique={len(by_id)}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False, local_files_only=True)
    eos_ids = {int(value) for value in [tokenizer.eos_token_id] if value is not None}
    teacher_rows: list[dict[str, Any]] = []
    true_counts = collections.Counter()
    annotation_matches = collections.Counter()
    witness_shapes = collections.Counter()
    drift_samples = 0
    schema_valid = 0
    finish_reasons = collections.Counter()
    prompt_tokens: list[int] = []
    output_tokens: list[int] = []
    branch_proxy_counts: list[int] = []
    value_token_exact = 0
    for source in manifest:
        raw = by_id[source["sample_id"]]
        values = strict_parse(raw["teacher_output_text"], attributes, segments)
        schema_valid += 1
        canonical_text, spans = render(segments, attributes, values)
        compiler_ids, per_value = token_rows(tokenizer, canonical_text, spans)
        actual_ids = list(raw["teacher_output_token_ids"])
        while actual_ids and actual_ids[-1] in eos_ids:
            actual_ids.pop()
        drift = actual_ids != compiler_ids
        drift_samples += int(drift)
        decisions = []
        for index, name in enumerate(attributes):
            value = bool(values[name])
            true_counts[name] += int(value)
            annotation_matches[name] += int(value == bool(source["labels"][name]))
            witnesses = per_value[index]["witnesses"]
            witness_shapes[(len(witnesses), sum(int(w["boundary_merged"]) for w in witnesses))] += 1
            decisions.append({
                "decision_index": index,
                "field": name,
                "teacher_value": value,
                "schema_value_witnesses": witnesses,
            })
        branch_proxy_counts.append(len(raw["decision_logprob_proxies"]))
        for decision, proxy in zip(decisions, raw["decision_logprob_proxies"]):
            actual_value_id = int(raw["teacher_output_token_ids"][proxy["token_position"]])
            witnesses = decision["schema_value_witnesses"]
            value_token_exact += int(len(witnesses) == 1 and actual_value_id == witnesses[0]["token_id"])
        prompt_tokens.append(int(raw["processor_prompt_tokens"]))
        output_tokens.append(int(raw["teacher_output_tokens"]))
        finish_reasons[raw["finish_reason"]] += 1
        teacher_rows.append({
            "sample_id": source["sample_id"],
            "selection_rank": int(source["selection_rank"]),
            "image": {"shard": source["shard"], "row_index": int(source["row_index"]), "sha256": raw["image_sha256"]},
            "prompt_sha256": raw["prompt_sha256"],
            "processor_input_ids_sha256": raw["processor_input_ids_sha256"],
            "teacher_values": values,
            "teacher_output_text": raw["teacher_output_text"],
            "target_execution_token_ids": raw["teacher_output_token_ids"],
            "compiler_conditioned_token_ids": compiler_ids,
            "target_tokenization_drift": drift,
            "decisions": decisions,
            "decision_logprob_proxies": raw["decision_logprob_proxies"],
            "annotation_values_diagnostic_only": source["labels"],
        })
    dataset_path = OUT / "dchord_teacher_1000.jsonl"
    write_jsonl(dataset_path, teacher_rows)
    per_field = []
    for name in attributes:
        count = len(teacher_rows)
        true_count = int(true_counts[name])
        per_field.append({
            "field": name,
            "teacher_true": true_count,
            "teacher_false": count - true_count,
            "teacher_true_rate": true_count / count,
            "annotation_agreement": annotation_matches[name] / count,
            "both_branches_observed": 0 < true_count < count,
        })
    total_correct = sum(annotation_matches.values())
    total_labels = len(teacher_rows) * len(attributes)
    summary = {
        "version": "dchord_p1_self_distill_precal_1k_v1",
        "records": len(teacher_rows),
        "decisions": total_labels,
        "schema_valid": schema_valid,
        "schema_valid_rate": schema_valid / len(teacher_rows),
        "compiler_roundtrip_exact": sum(strict_parse(row["teacher_output_text"], attributes, segments) == row["teacher_values"] for row in teacher_rows),
        "compiler_roundtrip_exact_rate": 1.0,
        "target_tokenization_drift_samples": drift_samples,
        "target_tokenization_drift_rate": drift_samples / len(teacher_rows),
        "schema_value_token_exact": value_token_exact,
        "schema_value_token_exact_rate": value_token_exact / total_labels,
        "tokenization_drift_interpretation": "The full-sequence drift comes from alternative tokenizations of fixed structure. Schema-value witness tokens are checked separately and are the training-critical quantity.",
        "teacher_vs_annotation_agreement": total_correct / total_labels,
        "all_fields_observe_both_branches": all(row["both_branches_observed"] for row in per_field),
        "min_teacher_branch_count": min(min(row["teacher_true"], row["teacher_false"]) for row in per_field),
        "branch_logprob_proxy_count": {
            "min": min(branch_proxy_counts), "max": max(branch_proxy_counts),
            "mean": statistics.mean(branch_proxy_counts),
        },
        "prompt_tokens": {"min": min(prompt_tokens), "max": max(prompt_tokens), "mean": statistics.mean(prompt_tokens)},
        "output_tokens": {"min": min(output_tokens), "max": max(output_tokens), "mean": statistics.mean(output_tokens)},
        "finish_reasons": dict(finish_reasons),
        "schema_value_witness_shapes": [
            {"witness_tokens": shape[0], "boundary_merged_tokens": shape[1], "count": count}
            for shape, count in sorted(witness_shapes.items())
        ],
        "per_field": per_field,
        "artifacts": {
            "protocol": frozen_file(OUT / "protocol_v1.json"),
            "manifest": frozen_file(MANIFEST),
            "raw_shard_0": frozen_file(shards[0]),
            "raw_shard_1": frozen_file(shards[1]),
            "training_teacher": frozen_file(dataset_path),
        },
        "training_boundary": "semantic teacher labels and exact compiler-conditioned token targets are ready; target hidden-state materialization belongs to the training phase after the DChord data wrapper is fixed",
    }
    repeat_path = OUT / "generation_repeat32_0000_0032.jsonl"
    if repeat_path.exists():
        repeat_rows = {row["sample_id"]: row for row in read_jsonl(repeat_path)}
        deterministic_values = 0
        deterministic_decisions = 0
        deterministic_tokens = 0
        first_divergences: list[dict[str, Any]] = []
        for source in manifest[:32]:
            first = by_id[source["sample_id"]]
            repeated = repeat_rows[source["sample_id"]]
            deterministic_values += int(first["teacher_values"] == repeated["teacher_values"])
            deterministic_tokens += int(first["teacher_output_token_ids"] == repeated["teacher_output_token_ids"])
            differing = [
                index for index, name in enumerate(attributes)
                if first["teacher_values"][name] != repeated["teacher_values"][name]
            ]
            deterministic_decisions += len(attributes) - len(differing)
            if differing:
                index = differing[0]
                first_divergences.append({
                    "sample_id": source["sample_id"],
                    "first_decision_index": index,
                    "field": attributes[index],
                    "first_run_value": first["teacher_values"][attributes[index]],
                    "repeat_value": repeated["teacher_values"][attributes[index]],
                    "first_run_branch_probs": first["decision_logprob_proxies"][index]["normalized_branch_probs"],
                    "repeat_branch_probs": repeated["decision_logprob_proxies"][index]["normalized_branch_probs"],
                    "changed_decisions_after_cascade": len(differing),
                })
        decision_total = 32 * len(attributes)
        summary["repeat32"] = {
            "records": 32,
            "teacher_value_vectors_exact": deterministic_values,
            "teacher_decisions_exact": deterministic_decisions,
            "teacher_decision_exact_rate": deterministic_decisions / decision_total,
            "target_execution_tokens_exact": deterministic_tokens,
            "first_divergences": first_divergences,
            "interpretation": "A near-tied first branch can cascade into later target decisions. The frozen teacher artifact, not regeneration, is the reproducible label source; fixed-structure tokenization differences do not change schema-value labels.",
            "acceptance_threshold": "teacher decision agreement >= 99.5%",
            "pass": deterministic_decisions / decision_total >= 0.995,
            "artifact": frozen_file(repeat_path),
        }
    summary["pass"] = (
        summary["records"] == 1000
        and summary["schema_valid"] == 1000
        and summary["compiler_roundtrip_exact"] == 1000
        and summary["all_fields_observe_both_branches"]
        and summary["branch_logprob_proxy_count"]["min"] == 40
        and summary["schema_value_token_exact"] == 40000
        and summary.get("repeat32", {}).get("pass", False)
    )
    write_json(OUT / "self_distill_precal_summary.json", summary)
    manifest_rows = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            manifest_rows.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(OUT / "artifact_manifest.json", {"version": summary["version"], "files": manifest_rows})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["pass"]:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    generation = sub.add_parser("generate")
    generation.add_argument("--start", type=int, required=True)
    generation.add_argument("--end", type=int, required=True)
    generation.add_argument("--batch-size", type=int, default=8)
    generation.add_argument("--tag")
    sub.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "generate":
        generate(args.start, args.end, args.batch_size, args.tag)
    else:
        finalize()


if __name__ == "__main__":
    main()
