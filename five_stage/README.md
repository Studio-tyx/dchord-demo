# DChord Parallel-verify + Parallel-draft 五阶段实现

## 环境要求
- conda env: torchspec
- 模型: Qwen3.5-4B + Qwen3.5-4B-DFlash (dchord_k3 checkpoint)
- transformers: get_rope_index 已改支持 4D（备份在 0829/backups/）
- xgrammar 0.2.3

## 文件说明
- dchord_5stage.py: 5 阶段实现（serial + parallel, 4D tree-mask, xgrammar bonus 判断, island-aware DFS/BFS 切分）
- dchord_5stage.py.bak_before_island_split: 孤岛切分前的版本
- test_dag_selection.py: DAG + DFS/BFS 选择算法单元测试（P22 例子）
- test_bonus.py: bonus 边界检测测试（多 token value）
- logs/: 运行日志

## 复现步骤

### 1. 环境
export PATH=/root/autodl-tmp/conda/envs/torchspec/bin:$PATH
export LD_LIBRARY_PATH=/root/autodl-tmp/conda/envs/torchspec/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/root/autodl-tmp/TorchSpec_DChord:/root/autodl-tmp/dchord_v22_torch_repro_code_v2/overlay/experiments/dchord/repro:$PYTHONPATH
export DCHORD_K3_CHECKPOINT=/root/autodl-tmp/0822/dchord_k3_random_anchor_1k_export/pytorch_model.bin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### 2. 跑 serial（baseline）
python dchord_5stage.py --limit 2 --mode serial --device cuda:0

### 3. 跑 parallel DFS（孤岛优先 + 连通子图深度优先）
python dchord_5stage.py --limit 2 --mode parallel --strategy dfs --device cuda:0

### 4. 跑 parallel BFS（孤岛优先 + 连通子图广度优先）
python dchord_5stage.py --limit 2 --mode parallel --strategy bfs --device cuda:0

### 5. 跑 DAG 选择算法单元测试
python test_dag_selection.py

## 结果对比（CelebA 2 samples, K=3）
| 指标 | serial | parallel/dfs | parallel/bfs |
|---|---|---|---|
| rounds | 14 | 14 | 14 |
| verify_forwards | 27 | 27 | 27 |
| correct | 32/40 | 32/40 | 32/40 |
| verify_ms | ~7341 | ~7212 | ~7672 |

CelebA 40 属性全为孤岛（0 依赖），DFS == BFS。两种策略在 woz/toolalpaca（有依赖边）上才会产生不同的切分。

## 关键设计

### 退化版 verify 切分（P22-23）
- DAG 只区分孤岛节点 vs 连通子图节点，不追踪子图内部的拓扑序
- 子图内部依赖由 4D tree-mask 的 causal chain 处理（自回归复用 KV）
- `is_ready` 只检查外部依赖（跨子图），忽略子图内依赖
- 孤岛最优先并行验证（独立 block），连通子图按 prompt 顺序给入 target forward

### DFS vs BFS
- **DFS**：尽量在一轮里推完一个连通子图（少存 KV）
- **BFS**：轮询每个子图各取一个就绪节点（多并行度，多存 KV）

### KV cache 管理（P21）
- 孤岛：独立 block，tree-mask 隔离，验证后提交到 prefix
- 连通子图：同一 group 的 block 形成 causal chain，自回归复用
- 当前实现：单线性 prefix（CachedTarget），speculative span 用 checkpoint/rollback 保护
- P21 的"当前连通子图结束可清除 kv cache"是未来优化（prefix 线性增长，未实现选择性清除）

## 关键改动
1. get_rope_index 4D 适配（0829/backups/modeling_qwen3_5.py.orig）
2. xgrammar SchemaStateMachine（替代 hardcoded STRUCTURAL）
3. 4D tree-mask verify（per-key 独立 attention + per-subgraph grouping）
4. bonus 边界检测（Case A/B）
5. island-aware DFS/BFS 选择算法（P23）
6. `external_preds` 忽略子图内依赖（退化核心）
7. `draft_choices_grouped` 支持非连续 field 选择
8. CachedTarget checkpoint/rollback（对齐 v22 风格）

## 相关文档
- /root/autodl-tmp/0829/reports/dchord_parallel_verify_report.md (完整 report)
- /root/autodl-tmp/0829/backups/get_rope_index_4d_patch.md (4D patch 说明)
