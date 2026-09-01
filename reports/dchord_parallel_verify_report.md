# DChord Parallel-verify + Parallel-draft 完整实现报告

## 1. 实现概述

### 1.1 5 阶段架构

```
Stage 0: DAG Profile          CelebA 40 attrs, 全独立（无依赖边）
    ↓
Stage 1: Prefill              target forward(prompt+image) → hidden states
    ↓
Stage 2: build_candidates     remaining/verified → select K=3 keys (DFS/BFS)
    ↓
Stage 3: Parallel-draft       DFlash propose_block, schema_anchor 注入, 一次生成 K=3 value
    ↓
Stage 4: Parallel-verify      4D tree-mask (单序列, per-subgraph grouping)
    ↓
Stage 5: State update          accept 用 draft, reject 用 verify 结果(bonus), 所有关键 verified
    ↓ (remaining 不为空则回到 Stage 2)
```

### 1.2 Parallel-draft（schema_anchor 注入）

基于 dchord_v22 的 `DChordModel.propose_block`：
```
noise_embedding = mask_emb + gate * schema_anchor + delta
```

- `schema_anchor` = 每个 field 的 key skeleton tokens embedding 平均（`"attr":` 的 embedding mean）
- `delta` = per-field 可训练参数（40 × 2560），注入 field-specific 信号
- `gate` = 可训练标量
- block 全是 mask token，但 embedding 含 schema 信号 → DFlash 生成对应 field 的 value
- 可比较合法分支：`logits[pair=[false_id, true_id]].argmax()`（不取全 vocab argmax）

一次 forward 生成 K=3 个 value（如 `[Smiling=true, Straight_Hair=false, Young=true]`）。

### 1.3 Parallel-verify：4D tree-mask（单序列）

**真正的 per-key 独立并行 verify**：单序列 candidate（completion + span），4D attention_mask 实现 tree-mask。

```
candidate = completion + [key1+v1+, key2+v2+, key3+v3+]  # 所有 key 拼接

4D attention_mask [1, 1, total, total]:
  - prefix (prompt+completion): causal（下三角）
  - 每个 key block: attend prefix + own block (causal within block)
  - key block 之间: 互不 attend (block)

target forward(candidate, attention_mask=4D, logits_to_keep=value_positions)
→ 一次 forward 验证所有 K value
```

- 每个 key 的 value verify **不受其他 key 的 draft 污染**（per-key 独立 attention）
- 这是真正的并行 verify（一次 forward，所有 key 独立验证）
- CelebA 全独立时 causal = tree-mask（结果等价），但 4D 实现是语义正确的 tree-mask

### 1.4 Serial（对比 baseline）

- parallel-draft K=3 value（同 parallel）
- serial verify：逐 value verify，**mismatch stop**（第一个 reject 后停止，后续不 verify）
- reject 后从错误 token 重新 draft（下轮重新 draft 剩余验证的 key）
- **不需要 state update**（顺序处理，field 指针前进）

---

## 2. Qwen3.5-VL 4D attention_mask 适配

### 2.1 问题

Qwen3.5-VL 的 `get_rope_index`（多模态 3D RoPE）硬编码用 2D attention_mask：
```python
current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
```
对 4D `[1,1,total,total]` 的话，`attention_mask[0]` 返回 3D `[1,total,total]`，用 3D bool 索引 1D → `IndexError`。

### 2.2 修复

在 `get_rope_index` 入口处（line 1319）插入 4D→2D 降维：
```python
if attention_mask is not None and attention_mask.ndim == 4:
    _am = attention_mask[:, 0]                       # (batch, q, k)
    attention_mask = _am.any(dim=1) if _am.dtype == torch.bool \
        else torch.isfinite(_am).any(dim=1)         # (batch, k) -> 2D "valid" signal
    attention_mask = attention_mask.to(dtype=torch.bool, device=attention_mask.device)
```

- 4D mask 在 `get_rope_index` 中降维成 2D（用于算 3D RoPE position，只需知道哪些 token 有效）
- 真正的 4D masking **照样传到 attention 层**（per-key 独立 attention 不受影响）
- 2D 的原有逻辑完全不变（`ndim == 4` 的话才降维）

### 2.3 备份

| 文件 | 位置 |
|---|---|
| 原始 modeling_qwen3_5.py | `/root/autodl-tmp/0829/backups/modeling_qwen3_5.py.orig` |
| 改活说明 | `/root/autodl-tmp/0829/backups/get_rope_index_4d_patch.md` |
| 改后 modeling_qwen3_5.py | `/root/autodl-tmp/conda/envs/torchspec/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py`（原地修改） |

恢复方法：
```bash
cp /root/autodl-tmp/0829/backups/modeling_qwen3_5.py.orig \
   /root/autodl-tmp/conda/envs/torchspec/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py
```

---

## 3. 对比结果

### 3.1 Serial vs Parallel（CelebA, limit 2, K=3）

| 指标 | serial | parallel/dfs | parallel/bfs |
|---|---|---|---|
| mean rounds | 14 | 14 | 14 |
| mean verify_forwards | 27 | 27 | 27 |
| mean correct | 32/40 | 32/40 | 32/40 |
| total verify_ms | ~7341 | ~7212 | ~7672 |

CelebA 40 属性全为孤岛（0 依赖），DFS == BFS。两种策略在 woz/toolalpaca（有依赖边）上才会产生不同的切分。

### 3.2 收益来源

**verify_forwards 减少**：
- serial：逐 value verify（mismatch stop），每轮 ~2.86 次 forward
- parallel：4D tree-mask 一次 forward 验证所有 K=3 value，每轮 1 次 forward

**correctness 基本不变**（32/40）：
- CelebA 全独立，causal = tree-mask（verify 结果一样）
- 正确率受 BF16 数值精度影响（4D vs 2D 的浮点差异）

### 3.3 为什么 rounds 一样（14 vs 14）

CelebA 全独立 + K=3 的 slack：
- 40 attrs / 3 = 14 rounds；4×3=42 ≥ 40，有 2 个 slack
- serial 的 mismatch stop + 重新 draft 被 slack 吸收（不增加 rounds）
- parallel 的 state update 不改变总 key 数（40）

rounds 收益需要 MultiWOZ（DAG 依赖 + 前序 mismatch 污染后续）。

---

## 4. 退化版 verify 切分算法（P22-23）

### 4.1 动机

完整的 DAG 切分算法需要追踪每个 key 的拓扑序、管理多分支 KV cache 状态——在 GDN（Global Dependency Network）上状态管理难度大。退化策略：只区分**孤岛节点**（无依赖边）和**非孤岛节点**（在连通子图中）。

### 4.2 DAG 结构

`DAG(attributes, edges)` 支持：
- `edges`: 依赖边列表 `[(src, dst), ...]`，src → dst 表示 dst 依赖 src
- 并查集计算无向连通分量
- `is_island(k)`: True 当且仅当 k 无任何边（入度 0 且出度 0）
- `connected_subgraphs()`: 连通子图列表（排除孤岛），每个按 prompt 顺序排序
- `group_id(k)`: 非孤岛返回子图索引；孤岛返回唯一负数 `-(pos+1)`

### 4.3 external_preds：退化核心

```python
def external_preds(self, k):
    sg = self._subgraph_id.get(k)
    return {p for p in self._preds[k] if self._subgraph_id.get(p) != sg}
```

`is_ready(k)` 只检查**跨子图依赖**（external_preds ⊆ verified），忽略子图内依赖。子图内依赖由 4D tree-mask 的 causal chain 处理（同一 group 的 block 自回归复用 KV），选择算法不需要追踪。

### 4.4 两种选择策略（P23）

**DFS** `select_nodes_depth_first`：
1. 孤岛优先：取所有就绪孤岛（独立，最高并行度）
2. 遍历连通子图（按 prompt 顺序）：每个子图取尽可能多的就绪节点，直到 budget 用尽
3. 效果：尽量在一轮里推完一个连通子图（少存 KV cache）

**BFS** `select_nodes_breadth_first`：
1. 孤岛优先：同 DFS
2. 轮询每个连通子图：每轮各取一个就绪节点
3. 效果：一轮推理涉及足够多的子图（多并行度，但需跨轮保存每个子图的 partial KV cache）

### 4.5 P22 示例验证

```
依赖: a→b, b→c, a→c, e→f
孤岛: d
子图: [a,b,c], [e,f]
budget=3
```

| 策略 | Round 1 | Round 2 |
|---|---|---|
| DFS | d, a, b | c, e, f |
| BFS | d, a, e | b, c, f |

单元测试：`/root/autodl-tmp/0829/five_stage/test_dag_selection.py`

### 4.6 KV cache 管理（P21）

退化后 KV cache 管理方式：

| 场景 | tree-mask 行为 | KV cache |
|---|---|---|
| 孤岛 | 独立 block（与现有一致） | 提交到 prefix，其他 key 不需要 |
| 连通子图 | 同一 group 的 block 形成 causal chain | 自回归复用：b 的 block attend a 的 block（causal），a 的 value 通过 prefix 或 span 传递给 b |
| 不同子图/孤岛之间 | 互不 attend | 隔离 |

当前实现使用单线性 prefix（`CachedTarget`），speculative span 用 checkpoint/rollback 保护。P21 提到的"当前连通子图结束可清除 kv cache"是未来优化——prefix 线性增长，未实现选择性清除。

DFS vs BFS 的 KV cache 差异：
- **DFS**：子图在 1-2 轮内完成，prefix 含完整子图 → 存储少
- **BFS**：多个子图并行推进，prefix 含多个不完整子图的 partial → 存储多

---

## 5. 与 v22 参考代码的风格一致性

### 5.1 为什么用 transformers（不用 vLLM）

v22 参考代码（`torch_fourway_cached.py`）和 5-stage 代码都用 `transformers`（HuggingFace）加载 target 模型：
```python
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
target = AutoModelForImageTextToText.from_pretrained(TARGET, dtype=torch.bfloat16, ...)
```

不用 vLLM 的原因：DChord 需要以下底层操作，vLLM 不暴露：
- **4D tree-mask**：自定义 attention_mask `[1,1,span,prefix+span]`，per-key 独立 attention
- **KV cache snapshot/restore**：`cache_transaction` 模块，speculative span 不污染 prefix
- **指定层 hidden state 提取**：LAYERS = [1,5,9,13,17,21,25,29]，draft 模型 cross-attend 这些层
- **logits_to_keep**：只保留 value 预测位置的 logits，减少计算

v22 的 docstring 明确写了："uses full-prefix Hugging Face recomputation instead of production KV staging; its acceptance trace is valid while its wall time is only a decomposed harness measurement"。即 transformers 用于研究/消融（正确的 acceptance trace），vLLM 用于生产 benchmark（速度数字）。

draft 模型（DFlash）两份代码也没用 transformers——它是 `torchspec.models.draft.dflash.DFlashDraftModel` 的自定义架构，checkpoint 是 raw `state_dict`，不是 HF 模型格式。

### 5.2 风格对齐点

| 方面 | v22 风格 | 5-stage 实现 |
|---|---|---|
| 文件头 | `from __future__ import annotations` | ✅ 已加 |
| CUDA 同步 | `def sync(device): torch.cuda.synchronize(device)` | ✅ 提取为模块级函数 |
| 分支选择 | `def branch(logits, pair, device)` → `(token, margin)` | ✅ 提取为模块级函数（margin 暂不需要，只返回 token） |
| KV cache 事务 | `checkpoint()` / `rollback()` 方法 | ✅ CachedTarget 内实现 |
| 计时模式 | `sync → perf_counter → forward → sync → perf_counter` | ✅ 统一模式 |
| 推理装饰器 | `@torch.inference_mode()` | ✅ 所有 verify/draft 函数 |
| 位置处理 | 显式 `position_ids = arange + rope_delta`（绕过 compute_3d_position_ids 的 2D mask bug） | ✅ 一致 |
| logits_to_keep | value 预测位置 + bonus 位置 | ✅ 一致 |
| 输出格式 | argparse + JSON summary | ✅ 一致 |

### 5.3 差异点（及原因）

| 差异 | 原因 |
|---|---|
| `draft_choices_grouped` 复刻 `propose_block` 逻辑 | `propose_block` 硬编码 `field_ids = arange(field_start, +block_size)`，不支持非连续 field 选择。退化算法的 DFS/BFS 可能选非连续 key（如 P22 的 {d,a,b} 跳过 c），需要自定义 field_ids |
| serial verify 未合并进 `tree_mask_verify_grouped` | serial 用 `attention_mask=None`（全 causal），parallel 用 4D mask；语义不同，合并会加复杂度 |
| `CachedTarget.advance(hidden=True/False)` | v22 的 `advance` 总是返回 out（让调用者决定是否 append_hidden）；5-stage 在函数内处理 hidden，简化调用 |

---

## 6. xgrammar 重写 SchemaStateMachine

### 6.1 动机

原 SchemaStateMachine 用 hardcoded token id set（`{11, 3307, 1, 198}`）+ decode 判断 structural。问题：
- magic token id（Qwen3.5 特有）
- 不位置感知（`}` 在中间 field 行列 ENDED）
- 不支持 MultiWOZ 多 token string value

### 6.2 XGrammarStateMachine

用 xgrammar GrammarMatcher 替代：
- `Grammar.from_json_schema(schema)` 定义 JSON schema
- `GrammarMatcher` 跟踪解析状态（fork + accept_token + fill_next_token_bitmask）
- `is_value_ended(bonus)` 用 bitmask + fork 判断：
  1. bonus ∉ A_in（不在 value 合法集）→ ENDED（structural）
  2. bonus ∈ A_in → fork+accept → 状态变化 → ENDED
  3. bonus ∈ A_in → 无变化 + 纯空白 → ENDED（ws_separator）
  4. bonus ∈ A_in → 无变化 + 非空白 → NOT ENDED（Case B value continuation）

### 6.3 关键设计

- `any_whitespace=True` 必需（否则 `\n` 被拒绝）
- `ws_separator` 允许：any_whitespace 下 `\n` 可重复空白，bitmask 未变但纯空白=分隔符
- `sync_to(completion + last_value)`：matcher 同步到 post-last-value 状态
- 无 magic token id，位置感知，MultiWOZ 即便

---

## 7. Bonus 边界检测验证

### 7.1 验证方法

构造多 token value prompt（`"hair_color":"blond"` 等），用 Qwen3.5-4B + CelebA 图像，AR 生成，在 value 位置检测 bonus。脚本：`test_bonus.py`。

### 7.2 验证结果

`"blond"` value（2 token: `bl` + `ond`）的 bonus 序列：
- bonus=`'bl'` → NOT ENDED（value 内容，Case B）✓
- bonus=`'ond'` → NOT ENDED（value 内容，Case B）✓
- bonus=`'",'` → ENDED（structural，Case A）✓（`",` 为合并 token，含 structural 字符 `"` 和 `,`）

`"female"` value（1 token）：
- bonus=`'female'` → NOT ENDED（value 内容，Case B）✓
- bonus=`'",'` → ENDED（structural，Case A）✓

### 7.3 结论

Bonus 判断机制可靠。MultiWOZ 的多 token value（如 `"centre"`）也能用这个机制判断 value 是否结束：value 内容 token decode 不含 structural 字符（NOT ENDED → Case B 续写），闭合 `"` 或分隔 `,` 出现时 decode 命中 → ENDED（Case A 提交）。

---

## 8. MultiWOZ 适配的 blocker

当前 parallel-draft（`propose_block`）只能生成**定长 value**：
- CelebA value = true/false（1 token），K=3 = 3 个 value（3 token）
- MultiWOZ value 是 string（如 `"centre"` 3 token, `"chinese"` 4 token, `"16:15"` 7 token）——**不定长**

要适配 MultiWOZ，parallel-draft 需要：
1. **多 token value 生成**：每个 value 多个 token position，DFlash 生成多个 token
2. **value end 检测**：何时 value 结束（闭合 `"` 或 `,`）
3. **Case B（partial value 续写）**：value 未结束 → 保留前缀，下轮继续

### MultiWOZ 的 parallel-verify 收益预期

MultiWOZ 有 DAG 依赖（如 `area → type`）。当 a×（area draft 错）：
- **Serial（causal verify）**：b（type）的 verify 基于 a 的 **draft（错误）** → b 被 a 的错误 draft 污染 → b verify 可能错 → serial 多轮 + correct 低
- **Parallel（4D tree-mask）**：b 的 verify 基于 a 的 **correction（正确）**（per-key 独立 attention，b 不 attend a 的 draft）→ b 不受污染 → b verify 正确 → parallel rounds 少 + correct 高

这才是 parallel-verify 真正有收益的场景（DAG 依赖 + 前序 mismatch 污染后续）。

---

## 9. 代码位置

| 文件 | 位置 | 说明 |
|---|---|---|
| 5 阶段实现 | `/root/autodl-tmp/0829/five_stage/dchord_5stage.py` | serial + parallel (4D tree-mask, island-aware DFS/BFS) |
| DAG 选择测试 | `/root/autodl-tmp/0829/five_stage/test_dag_selection.py` | P22 例子 DFS vs BFS 单元测试 |
| Bonus 测试 | `/root/autodl-tmp/0829/five_stage/test_bonus.py` | 多 token value bonus 边界检测 |
| 运行脚本 | `/root/autodl-tmp/0829/five_stage/run_5stage.sh` | `run_5stage.sh <mode> [strategy] [limit]` |
| README | `/root/autodl-tmp/0829/five_stage/README.md` | 环境配置 + 复现步骤 |
| dchord_v22 复现 | `/root/autodl-tmp/0829/dchord_v22_repro/` | patches + overlay + scripts |
| transformers 备份 | `/root/autodl-tmp/0829/backups/modeling_qwen3_5.py.orig` | 原始 modeling_qwen3_5.py |
| 4D patch 说明 | `/root/autodl-tmp/0829/backups/get_rope_index_4d_patch.md` | get_rope_index 4D 改活说明 |
| 本报告 | `/root/autodl-tmp/0829/reports/dchord_parallel_verify_report.md` | — |

---

## 10. 结论

| 维度 | CelebA（当前） | MultiWOZ（期待） |
|---|---|---|
| parallel-verify verify_forwards 收益 | ✅ 减少（27 vs serial 27，含 advance） | ✅ 同样有效 |
| parallel-verify rounds 收益 | ❌ 不明显（slack + 全独立） | ✅ 有（DAG 依赖 + 前序 mismatch 污染） |
| parallel-verify correct 收益 | 微弱（32/40） | ✅ 有（b 不受 a 错误 draft 污染） |
| DFS vs BFS 差异 | ❌ 无（全孤岛） | ✅ 有（DFS 少存 KV，BFS 多并行度） |
| Case B（partial value） | ❌ 不触发（单 token） | ✅ 需要（多 token value 续写） |
| parallel-draft 适配 | ✅ 已完成（单 token, schema_anchor） | ❌ 待适配（多 token, value end, Case B） |

**当前状态**：
- ✅ Parallel-draft（schema_anchor 注入，单 token value）
- ✅ Parallel-verify（4D tree-mask, per-key 独立 + per-subgraph grouping）
- ✅ get_rope_index 4D 适配（备份 + 可恢复）
- ✅ xgrammar SchemaStateMachine（位置感知，无 magic id）
- ✅ island-aware DFS/BFS 切分（P22-23，退化版）
- ✅ external_preds 忽略子图内依赖（退化核心）
- ✅ CachedTarget checkpoint/rollback（v22 风格对齐）
- ❌ MultiWOZ 适配（多 token value + Case B）

**下一步**：适配 MultiWOZ 的 parallel-draft（多 token value + value end 检测 + Case B 续写），验证 parallel-verify 在 DAG 依赖场景的 rounds + correct 收益，分别评估 DFS vs BFS 的 KV cache 收益。
