#!/usr/bin/env python3
"""Rebuild synthetic CelebA test parquet shards for dchord load_image.

load_image (common.py) reads a row by absolute position == row_index, slicing
one row out of a row group.  Therefore each shard must hold dense rows
0..max(row_index) so every requested row_index resolves.  All rows carry the
same image bytes from celeba_test.jpg.  The image column mirrors the HF image
feature schema: struct<bytes: binary, path: string> so that
cell["image"]["bytes"] yields the JPEG bytes.
"""
import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ASSETS = Path("/root/autodl-tmp/dchord_v22_torch_repro_code_v2/overlay/experiments/dchord/repro/assets")
JSONL = ASSETS / "p1_test_512.jsonl"
IMG = Path("/root/autodl-tmp/celeba_test.jpg")
OUT = Path("/root/autodl-tmp/dchord_celeba_synth")
SHARDS = [
    "test-00000-of-00003.parquet",
    "test-00001-of-00003.parquet",
    "test-00002-of-00003.parquet",
]

IMAGE_TYPE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def main():
    img_bytes = IMG.read_bytes()
    print(f"image bytes: {len(img_bytes)} ({IMG})", flush=True)
    rows = [json.loads(line) for line in JSONL.read_text().splitlines() if line.strip()]
    print(f"jsonl rows: {len(rows)}", flush=True)

    by_shard = defaultdict(list)
    for r in rows:
        by_shard[r["shard"]].append(int(r["row_index"]))

    OUT.mkdir(parents=True, exist_ok=True)
    for shard in SHARDS:
        indices = by_shard.get(shard, [])
        if indices:
            n_rows = max(indices) + 1
            min_idx = min(indices)
            max_idx = max(indices)
            n_entries = len(indices)
        else:
            n_rows = 1
            min_idx = max_idx = -1
            n_entries = 0

        cells = [{"bytes": img_bytes, "path": None} for _ in range(n_rows)]
        table = pa.table({"image": pa.array(cells, type=IMAGE_TYPE)})
        out_path = OUT / shard
        pq.write_table(table, out_path, compression="snappy")

        pf = pq.ParquetFile(out_path)
        meta = pf.metadata
        probe_at = max_idx if n_entries else 0
        got = (pf.read_row_group(0, columns=["image"])
               .slice(probe_at, 1).to_pylist()[0]["image"]["bytes"])
        ok = got == img_bytes
        print(
            f"{shard}: rows={n_rows} row_groups={meta.num_row_groups} "
            f"row_index[min={min_idx},max={max_idx}] jsonl_entries={n_entries} "
            f"size={out_path.stat().st_size} verify_bytes_ok={ok}",
            flush=True,
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
