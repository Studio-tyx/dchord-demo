# DChord Parallel-verify + Parallel-draft 实现报告

## 1. 实现说明

### 1.1 整体架构（5 阶段）

```
Stage 0: DAG Profile          CelebA 40 attrs, 全独立（无依赖边）
    ↓
Stage 1: Prefill              target forward(prompt+image) → hidden states
    ↓
Stage 2: build_candidates     remaining/verified → pack K=3 keys (按 JSON 顺序)
    ↓
Stage 3: Parallel-draft       DFlash propose_block, schema_anchor 注入, 一次生成 K=3 value
    ↓
Stage 4: Parallel-verify      per-key 独立 verify (batch expansion tree-mask)
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
- `delta` = per-field 可学习参数（40 × 2560），训练后注入 field-specific 信号
- `gate` = 可学习标量（初始 0.1）
- block 全是 mask token，但 embedding 含 schema 信号 → DFlash 生成对应 field 的 value
- 只比较合法分支：`logits[pair=[false_id, true_id]].argmax()`（不取全 vocab argmax）

一次 forward 生成 K=3 个 value（如 `[Smiling=true, Straight_Hair=false, Young=true]`）。

### 1.3 Parallel-verify（batch expansion tree-mask）

**真正的 per-key 独立 verify**：每个 key 单独 forward，candidate = `base_ids[:positions[fi]+1]`（含到该 key 的 value），前序 value 已替换（verified 或 correction）。

```
for each key fi in patch:
    candidate = base_ids[:positions[fi]+1]  # 只含到该 key
    # 前序 value 替换：verified 的用 branches 值，reject 的用 correction
    candidate[positions[prev]] = verified_value or correction
    candidate[positions[fi]] = draft_value  # 本 key 的 draft
    ch = target_choices(candidate, [positions[fi]], [fi])  # 一次 forward
```

- 每个 key 的 verify **不被其他 key 的 draft 污染**（per-key 独立 candidate）
- 这就是 tree-mask 语义（MultiWOZ 有 DAG 依赖时关键）
- CelebA 全独立时，causal = tree-mask（结果等价），但实现是真正的 per-key 独立

**State update（Stage 5）**：
- accept（draft == verify）：commit draft value，verified
- reject（draft != verify）：commit **verify 结果**（target argmax = bonus = 正确值），verified
- 所有关键一次 verify 解决，不重新 draft

### 1.4 Serial（对比 baseline）

- parallel-draft K=3 value（同 parallel）
- serial verify：逐 value verify，**mismatch stop**（第一个 reject 后停止，后续不 verify）
- reject 后从错误 token 重新 draft（下轮重新 draft 未验证的 key）
- **不需要 state update**（顺序处理，field 指针前进）

### 1.5 Qwen3.5-VL 的 tree-mask 限制

Qwen3.5-VL 的 `get_rope_index`（多模态 3D RoPE）硬编码用 2D attention_mask：
```python
current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
```
传 4D attention_mask 时 `attention_mask[0]` 返回 3D，索引 1D input_ids → IndexError。

因此不能用 4D attention_mask 实现单序列 tree-mask，改用 batch expansion（per-key 独立 forward）。

---

## 2. 对比结果（CelebA, 12 samples, Qwen3.5-4B, delta/gate 未训练）

| 指标 | serial | parallel | 差异 |
|---|---|---|---|
| mean rounds | 14.00 | 14.00 | **0** |
| mean verify_forwards | 40.00 | 40.00 | **0** |
| mean correct | 32.50/40 | 32.92/40 | +0.42 |

### a×b√ case 逐项对比

| sample | case | serial rounds | par rounds | serial correct | par correct |
|---|---|---|---|---|---|
| s4 r12 | a√b×c√ (Wearing_Necklace) | 14 | 14 | 31 | 31 |
| s6 r10 | a√b×c√ (Smiling) | 14 | 14 | 37 | 37 |
| s9 r7 | a×b√c√ (Mouth_Slightly_Open) | 14 | 14 | 35 | 35 |

**rounds 和 correct 完全一样**，parallel 没有收益。

---

## 3. 收益不明显的原因

### ① Draft 精度足够高（很少 reject）

12 samples × 14 rounds = 168 个 value decision，只有 14 个 reject（8.3%）。大部分轮次 accept all 3。reject 少 → parallel 的"mismatch 后续也 accept"的收益机会少。

### ② 出错都在 a√b√c× 的位置（被 bonus 修复）

14 个 reject 中，11 个是 reject@2（最后一个 offset，a√b√c×）：
- serial：reject@2 → correction c → 前进（c 已 verified）
- parallel：reject@2 → correction c → c verified

reject@2 时，a,b 已 accept，serial 和 parallel 都不需要重 draft a,b。**bonus（correction）直接修复 c，serial 和 parallel 的行为一样**。

### ③ 即使出现 a×b√，也被 40 key / (3×14 round) 的容错包住

**Sample 9 Round 7 举例**：

```
s9 r7: draft=[Mouth_Slig=T, Mustache=F, Narrow_Eye=F]
       verify=[Mouth_Slig=F, Mustache=F, Narrow_Eye=F]
       → reject@0 (Mouth_Slightly_Open: draft=T, verify=F)
       → a×b√c√
```

- **Serial**：a× → mismatch stop → correction a（F）→ 前进到 b。下轮 draft b,c,d（重新 draft b,c）。但 b,c 上一轮已 draft（=F,F）且 verify 对（=F,F），重 draft 结果一样 → **浪费 b,c 的 draft，但 rounds 不增加**。
- **Parallel**：a× corrected（F）→ b√c√ accept → 所有关键 verified。下轮 draft d,e,f（不重 draft b,c）。

**为什么 rounds 不增加？**

40 attrs / K=3 = 14 rounds（14×3=42 ≥ 40，有 2 个 slack）。serial 的 a×b√ → b,c 重 draft，但 b,c 在下一轮的 K=3 slot 里（b,c,d），不增加总轮数。slack 把 reject 吸收了。

如果 K=1（无 slack），reject@0 会导致 serial 多一轮（重 draft b），parallel 不需要。但 K=3 的 slack 让 serial 和 parallel 的 rounds 一样。

**而且 CelebA 全独立**：serial 重 draft b,c 时，completion 已含 a 的 correction（正确），b,c 的 draft 不受 a 的错误 draft 污染 → 重 draft 结果和 parallel 一样 → correct 也一样。

---

## 4. MultiWOZ 适配的 blocker

当前 parallel-draft（`propose_block`）只能生成**定长 value**：

- CelebA value = true/false（1 token），`propose_block` 的 K=3 = 3 个 value（3 token）
- schema_anchor 注入针对单 token value（pair=[false_id, true_id]）
- MultiWOZ value 是 string（如 `"centre"` 3 token, `"chinese"` 4 token, `"16:15"` 7 token）——**不定长**

要适配 MultiWOZ，parallel-draft 需要：
1. **多 token value 生成**：每个 value 多个 token position，DFlash 生成多个 token（不是 1 token/value）
2. **value end 检测**：何时 value 结束（闭合 `"` 或 `,`）
3. **Case B（partial value 续写）**：value 未结束（接受了前缀 `"ab`，但 `c"` 错或待生成）→ 保留 `"ab`，下轮续写 `c"`

这些目前都没实现。CelebA 单 token value 不需要这些。

**MultiWOZ 的 parallel-verify 收益预期**：

MultiWOZ 有 DAG 依赖（如 `area → type`）。当 a×（area draft 错）：
- **Serial（causal verify）**：b（type）的 verify 基于 a 的 **draft（错误）** → b 被 a 的错误 draft 污染 → b verify 可能错 → serial 多轮 + correct 低
- **Parallel（per-key 独立 verify）**：b 的 verify 基于 a 的 **correction（正确）** → b 不受污染 → b verify 正确 → parallel rounds 少 + correct 高

这才是 parallel-verify 真正有收益的场景（DAG 依赖 + 前序 mismatch 污染后续）。

---

## 5. 代码位置

- 5 阶段实现：`/root/autodl-tmp/0829/five_stage/dchord_5stage.py`
- dchord_v22 复现：`/root/autodl-tmp/0829/dchord_v22_repro/`
- acceptance 结果：`/root/autodl-tmp/0829/reports/dchord_repro_test/`
- 对比 log：`/root/autodl-tmp/reports/log/ser_final.log`, `par_final.log`, `ser12.log`, `par12.log`

## 6. 结论

| 维度 | CelebA（当前） | MultiWOZ（预期） |
|---|---|---|
| parallel-verify rounds 收益 | ❌ 不明显（slack + 全独立 + reject 少） | ✅ 有（DAG 依赖 + 前序 mismatch 污染后续） |
| parallel-verify correct 收益 | 微弱（+0.42/40） | ✅ 有（b 不被 a 的错误 draft 污染） |
| Case B（partial value） | ❌ 不触发（单 token） | ✅ 需要（多 token value 续写） |
| parallel-draft 适配 | ✅ 已完成（单 token, schema_anchor） | ❌ 待适配（多 token, value end, Case B） |

**下一步**：适配 MultiWOZ 的 parallel-draft（多 token value + Case B），验证 parallel-verify 在 DAG 依赖场景的收益。
