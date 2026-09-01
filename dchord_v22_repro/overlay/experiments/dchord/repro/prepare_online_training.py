#!/usr/bin/env python3
"""Prepare multimodal TorchSpec conversations and an internal DFlash checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

from torchspec.models.draft.dflash import DFlashConfig, DFlashDraftModel
from torchspec.models.draft.keymap import to_internal_keys


HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("DCHORD_REPRO_ROOT", "/root/autodl-tmp"))
TARGET = Path(os.environ.get("DCHORD_TARGET", str(ROOT / "models/Qwen3.5-4B")))
OFFICIAL = Path(os.environ.get(
    "DCHORD_OFFICIAL_DFLASH", str(ROOT / "models/Qwen3.5-4B-DFlash")
))
TEACHER = Path(os.environ.get(
    "DCHORD_TEACHER_DATA",
    str(ROOT / "reports/dchord_torch_repro/self_distill_1k/dchord_teacher_1000.jsonl"),
))
PROMPT = HERE / "assets/prompt_schema_oneshot_qwen35_v1.txt"
PCS = HERE / "assets/profiled_surface_spec_v1.json"
PARQUET_ROOT = Path(os.environ.get(
    "DCHORD_CELEBA_PARQUET",
    str(ROOT / "datasets/celeba/img_align+identity+attr"),
))
COMPAT = HERE / "assets/dflash_qwen35_torchspec_config.json"
OUT = Path(os.environ.get(
    "DCHORD_ONLINE_DATA_OUTPUT",
    str(ROOT / "reports/dchord_torch_repro/training/data"),
))
INTERNAL = Path(os.environ.get(
    "DCHORD_INITIAL_DRAFT_OUTPUT",
    str(ROOT / "models/Qwen3.5-4B-DFlash-torchspec-dchord"),
))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def render(segments: list[str], attributes: list[str], values: dict[str, bool]) -> str:
    pieces = []
    for index, segment in enumerate(segments):
        pieces.append(segment)
        if index < len(attributes):
            pieces.append("true" if values[attributes[index]] else "false")
    return "".join(pieces)


def extract_image(row: dict, destination: Path) -> None:
    ref = row["image"]
    parquet = pq.ParquetFile(PARQUET_ROOT / ref["shard"])
    target = int(ref["row_index"])
    cursor = 0
    for group_index in range(parquet.metadata.num_row_groups):
        count = parquet.metadata.row_group(group_index).num_rows
        if target < cursor + count:
            record = parquet.read_row_group(group_index, columns=["image"]).slice(
                target - cursor, 1
            ).to_pylist()[0]
            raw = record["image"]["bytes"]
            if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
                raise ValueError(f"image hash mismatch: {row['sample_id']}")
            with Image.open(io.BytesIO(raw)) as image:
                image.convert("RGB").save(destination, format="JPEG", quality=95)
            return
        cursor += count
    raise IndexError(row["sample_id"])


def prepare_data(limit: int) -> None:
    rows = read_jsonl(TEACHER)[:limit]
    pcs = json.loads(PCS.read_text())
    attributes = list(pcs["schema"]["attributes"])
    segments = list(pcs["surface"]["segments"])
    prompt = PROMPT.read_text()
    image_dir = OUT / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    output_path = OUT / f"train_{limit}.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            image_path = image_dir / f"{index:04d}.jpg"
            if not image_path.exists():
                extract_image(row, image_path)
            compiled = render(segments, attributes, row["teacher_values"])
            record = {
                "id": row["sample_id"],
                "conversations": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{image_path}"},
                            {"type": "text", "text": prompt},
                        ],
                    },
                    {"role": "assistant", "content": compiled},
                ],
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "version": "q35_dchord_torchspec_online_data_v1",
        "records": len(rows),
        "dataset": str(output_path),
        "image_dir": str(image_dir),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "pcs_sha256": hashlib.sha256(PCS.read_bytes()).hexdigest(),
        "all_compiled_outputs_match_teacher_values": True,
    }
    (OUT / f"train_{limit}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def load_target_embedding(key: str) -> torch.Tensor:
    index = json.loads((TARGET / "model.safetensors.index.json").read_text())
    shard = index["weight_map"][key]
    with safe_open(TARGET / shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def prepare_checkpoint() -> None:
    config_dict = json.loads(COMPAT.read_text())
    config = DFlashConfig(**config_dict)
    model = DFlashDraftModel(config)
    with safe_open(OFFICIAL / "model.safetensors", framework="pt", device="cpu") as handle:
        official = {key: handle.get_tensor(key) for key in handle.keys()}
    mapped = to_internal_keys(official, model.state_dict().keys())
    result = model.load_state_dict(mapped, strict=False)
    if list(result.missing_keys) != ["embed_tokens.weight"] or result.unexpected_keys:
        raise RuntimeError(result)
    model.embed_tokens.weight.data.copy_(load_target_embedding(config_dict["embedding_key"]))
    INTERNAL.mkdir(parents=True, exist_ok=True)
    state = {key: value.contiguous() for key, value in model.state_dict().items()}
    save_file(state, INTERNAL / "model.safetensors", metadata={"format": "torchspec_internal"})
    shutil.copy2(COMPAT, INTERNAL / "config.json")
    print(
        json.dumps(
            {
                "checkpoint": str(INTERNAL),
                "tensors": len(state),
                "bytes": (INTERNAL / "model.safetensors").stat().st_size,
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--skip-checkpoint", action="store_true")
    args = parser.parse_args()
    prepare_data(args.limit)
    if not args.skip_checkpoint:
        prepare_checkpoint()


if __name__ == "__main__":
    main()
