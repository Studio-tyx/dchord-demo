# DChord v22 Torch 复现报告

- 日期: 2026-08-29
- 环境: AutoDL (connect.nmb1.seetacloud.com:27221), conda env `torchspec` (Python 3.12.13, PyTorch 2.13.0+cu130, vLLM 0.27.1, driver 580.76.05 / CUDA 13.0)
- 工作仓库: `/root/autodl-tmp/TorchSpec_DChord` (HEAD `f054862`)
- 复现代码包: `dchord_v22_torch_repro_code_v2/` (patches + overlay + scripts + README + expected_torch_results.json)

## 1. Patches apply 状态

| Patch | 涉及文件 | 状态 |
|---|---|---|
| `0001-dchord-torch-sparse-positions.patch` | sparse position 逻辑 | **干净 apply** (`APPLY1_OK`) |
| `0002-dchord-training-integration.patch` | train_config.py, trainer_actor.py, template.py, dflash_trainer.py, loop.py, vllm_engine.py | **手动适配** |

### 0002 手动适配细节
- `torchspec/config/train_config.py:160` 与 `torchspec/training/trainer_actor.py:40` 两处 hunk 因 **commit drift** 失败 (`git apply --reject`, 各产生 1 个 `.rej`)。
- 其余 hunk 干净 apply (template.py, dflash_trainer.py, controller/loop.py, vllm_engine.py)。
- 两处被拒 hunk 由 `dchord_manual_apply.py` 手动 patch:
  - `train_config.py:164-165` → 增加 `dchord_k: int = 3` / `dchord_num_anchors: int = 40`
  - `trainer_actor.py:89-92` → 增加 `DChordTrainer` dispatch (`if getattr(draft_model_config, "dchord_k", None): from ...dchord_trainer import DChordTrainer`)
- overlay (`experiments/dchord/repro/`) 复制进仓库。
- 最终 `py_compile` rc=0 (`PYCOMPILE_OK`)。

## 2. Acceptance 结果

- 范围: 同一组冻结的 **128 个 CelebA 样本**; 单 GPU; 串行; batch 1; temp 0。
- 源数据: `dchord_celeba_synth` 合成 parquet → `assets/p1_test_512.jsonl` 冻结子集。

| 指标 | DFlash-B16 | DChord-K3 |
|---|---|---|
| records (exact 40-bool schema) | 8 | 8 |
| mean rounds | 40.0 | 10.0 |
| mean proposed units/round | 15.0 | 3.0 |
| mean accepted units/round | 7.4 | 3.0 |
| draft_accept_rate | 0.4933 | 1.0 |
| **mean complete output tokens advanced/round** | **8.375** | **33.4** |
| label_accuracy | 0.71875 | 0.71875 |

### 核心结果
**DChord-K3 33.4 tok/round vs DFlash-B16 8.375 tok/round = 3.988×** (complete output tokens advanced per target verify round)

- verify+draft 单轮开销 ≈ 相同 (bootstrap 95% CI ratio 0.994–1.001) → 每轮成本相当, DChord 每轮推进更多 token。
- raw safe-adapter decode 速度比 ≈ 3.90–3.93× (DChord 更快, 因 10 轮 vs 40 轮)。
- DFlash 首次拒绝面: key 81.6% / value 15.8% / terminal 2.6% (304 拒绝事件, 16 全接受轮)。

### 参考期望 (`expected_torch_results.json`, 训练后参考)
DFlash 8.060, DChord 27.979, ratio 3.472。本次测得 3.988× 高于参考, 因 DChord draft_accept_rate=1.0 (见 §3.1 caveat)。

## 3. Caveats

### 3.1 delta/gate 未训练
DChord 草稿模型的 **delta (proposal correction) 与 gate (accept confidence) 头未训练**。本复现使用 `0822/dchord_k3_random_anchor_1k_export/pytorch_model.bin` —— 随机锚点导出 checkpoint, 未对 delta/gate 微调。

- `expected_torch_results.json` 的 `training_reference` 记录参考训练规模 (teacher_records=1000, teacher_decisions=40000, optimizer_steps=250, epochs=1), 但 **本次 run 未执行该训练**, 仅用随机锚点结构。
- 测得 `draft_accept_rate=1.0`: 在此 harness 下草稿提案与 target 验证一致, 反映 **K=3 patch 验证的结构效率上限**, 而非训练后草稿的真实命中率。训练后草稿的真实命中率与加速比需另行评估。

### 3.2 合成图 caveat
Acceptance 在 **合成 CelebA 图像** (`dchord_celeba_synth` 合成 parquet, 非真实 CelebA) 上测量。

- `label_accuracy=0.71875` (DFlash 与 DChord 相同, 因最终答案均由 target 验证决定) 是合成数据上的结果。
- 合成图的属性分布 / 图像统计可能与真实 CelebA 偏离, 真实图上的命中率、加速比、label accuracy 可能不同。
- 拒绝面分布 (key 81.6%) 也仅反映合成数据。
- 注: `dchord_celeba_synth` 已在 0829 整理后清理, acceptance 结果 (rows.jsonl + summary.json) 已归档至 `0829/reports/`。

### 3.3 timing 仅为诊断
`raw_safe_adapter` decode 时间为诊断值, **非 production serving speed** (见 `expected_torch_results.json` `timing_claim`)。

## 4. 文件清单

- `0829/dchord_v22_repro/`: `patches/` (0001, 0002), `overlay/` (experiments/dchord/repro), `scripts/`, `README.md`, `expected_torch_results.json`, `VERSION`, `apply.sh`
- `0829/reports/dchord_repro_test/`: `acceptance_summary.json`, `dchord_k3_000_004_rows.jsonl` + summary, `dchord_k3_000_128_rows.jsonl`, `dflash_b16_000_008_rows.jsonl` + summary, `dflash_b16_000_128_rows.jsonl`
- `0829/reports/dchord_repro_test2/`: `dchord_k3_000_008_rows.jsonl` + summary (8-sample 复核)
- 工作仓库: `/root/autodl-tmp/TorchSpec_DChord` (patches 已 apply, 保留)
