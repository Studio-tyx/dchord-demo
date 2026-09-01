#!/usr/bin/env python3
"""DChord 5-stage: serial vs parallel-verify with island-aware splitting.

Degraded verify-splitting (P22-23): only distinguishes island nodes (no
dependency edges) from non-island nodes (in a connected subgraph). Islands
are verified in parallel (isolated tree-mask blocks); connected-subgraph
keys are verified in prompt order as a causal chain (autoregressive KV
reuse within the subgraph, P21). Two strategies:
  - DFS: exhaust one subgraph before the next (less KV storage)
  - BFS: round-robin across subgraphs (more parallelism, more KV storage)
"""
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import xgrammar as xgr
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/TorchSpec_DChord")
ROOT = Path(os.environ.get("DCHORD_REPRO_ROOT", "/root/autodl-tmp"))
TARGET = Path(os.environ.get("DCHORD_TARGET", str(ROOT / "models/Qwen3.5-4B")))
ASSETS = Path(os.environ.get(
    "DCHORD_REPRO_ASSETS",
    str(ROOT / "dchord_v22_torch_repro_code_v2/overlay/experiments/dchord/repro/assets"),
))
EXPORT = Path(os.environ.get(
    "DCHORD_K3_CHECKPOINT",
    str(ROOT / "0822/dchord_k3_random_anchor_1k_export/pytorch_model.bin"),
))
LAYERS = [1, 5, 9, 13, 17, 21, 25, 29]

sys.path.insert(0, str(ROOT / "dchord_v22_torch_repro_code_v2/overlay/experiments/dchord/repro"))
import common as DCC  # noqa: E402
from cache_transaction import snapshot_cache, restore_cache  # noqa: E402


def sync(device):
    torch.cuda.synchronize(device)


def branch(logits, pair, device):
    pair_tensor = torch.tensor(pair, dtype=torch.long, device=device)
    scores = logits[pair_tensor].float()
    return int(pair_tensor[int(scores.argmax())])


# ============================================================
# DAG: dependency graph with island / connected-subgraph split.
#
# Degraded for GDN: only distinguishes islands (no edges) from connected
# subgraphs. The full topological order within subgraphs is NOT tracked --
# subgraph keys are verified in prompt order, which naturally respects
# dependencies because edges follow prompt order (a->b means a precedes b).
# ============================================================
class DAG:
    def __init__(self, attributes: list[str], edges: list[tuple[str, str]] | None = None):
        self.nodes = list(attributes)
        self.pos = {a: i for i, a in enumerate(attributes)}
        self.edges = list(edges) if edges else []
        self._preds: dict[str, set[str]] = {a: set() for a in attributes}
        self._succs: dict[str, set[str]] = {a: set() for a in attributes}
        for src, dst in self.edges:
            self._preds[dst].add(src)
            self._succs[src].add(dst)
        # Union-find for undirected connected components.
        parent = {a: a for a in attributes}

        def find(x):
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        for src, dst in self.edges:
            ps, pd = find(src), find(dst)
            if ps != pd:
                parent[ps] = pd
        self._island = {
            a: (not self._preds[a] and not self._succs[a]) for a in attributes
        }
        comp_members: dict[str, list[str]] = {}
        for a in attributes:
            if not self._island[a]:
                comp_members.setdefault(find(a), []).append(a)
        self._subgraphs = [
            sorted(m, key=lambda k: (self.pos[k], k))
            for m in comp_members.values()
        ]
        self._subgraph_id: dict[str, int] = {}
        for i, members in enumerate(self._subgraphs):
            for a in members:
                self._subgraph_id[a] = i

    def preds(self, k) -> set[str]:
        return self._preds[k]

    def external_preds(self, k) -> set[str]:
        """Predecessors NOT in the same connected subgraph.

        Intra-subgraph dependencies are handled by the tree-mask causal
        chain (autoregressive within the group), so the selection only
        checks external (cross-subgraph) dependencies for readiness."""
        sg = self._subgraph_id.get(k)
        return {p for p in self._preds[k] if self._subgraph_id.get(p) != sg}

    def is_island(self, k) -> bool:
        return self._island[k]

    def group_id(self, k) -> int:
        """Subgraph index for non-island; unique negative for island."""
        return self._subgraph_id.get(k, -(self.pos[k] + 1))

    def connected_subgraphs(self) -> list[list[str]]:
        return self._subgraphs


# ============================================================
# Selection: DFS / BFS (P23).
#
# Islands always go first (independent, highest parallelism). Then:
#   DFS  -- greedily exhaust each connected subgraph in prompt order
#   BFS  -- round-robin one ready node per subgraph per pass
# ============================================================
def _split_isolated_and_subgraphs(remaining, dag):
    isolated = sorted(
        (k for k in remaining if dag.is_island(k)),
        key=lambda k: (dag.pos[k], k),
    )
    groups = []
    for members in dag.connected_subgraphs():
        in_remaining = [k for k in members if k in remaining]
        if in_remaining:
            groups.append(in_remaining)
    return isolated, groups


def _is_ready(k, dag, verified):
    return dag.external_preds(k) <= verified


def select_nodes_depth_first(remaining, dag, verified, budget):
    isolated, connected_groups = _split_isolated_and_subgraphs(remaining, dag)
    selected = []
    for k in isolated:
        if len(selected) >= budget:
            break
        if _is_ready(k, dag, verified):
            selected.append(k)
    if len(selected) >= budget:
        return selected[:budget]
    for group in connected_groups:
        ready = [k for k in group if _is_ready(k, dag, verified)]
        for k in ready:
            if len(selected) >= budget:
                break
            selected.append(k)
        if len(selected) >= budget:
            break
    return selected


def select_nodes_breadth_first(remaining, dag, verified, budget):
    isolated, connected_groups = _split_isolated_and_subgraphs(remaining, dag)
    selected = []
    for k in isolated:
        if len(selected) >= budget:
            break
        if _is_ready(k, dag, verified):
            selected.append(k)
    if len(selected) >= budget:
        return selected[:budget]
    ready_lists = [
        sorted((k for k in g if _is_ready(k, dag, verified)),
               key=lambda k: (dag.pos[k], k))
        for g in connected_groups
    ]
    ready_lists = [r for r in ready_lists if r]
    pointers = [0] * len(ready_lists)
    while len(selected) < budget and any(
        p < len(r) for p, r in zip(pointers, ready_lists)
    ):
        for i, ready in enumerate(ready_lists):
            if len(selected) >= budget:
                break
            if pointers[i] < len(ready):
                selected.append(ready[pointers[i]])
                pointers[i] += 1
    return selected


# ============================================================
# XGrammar state machine (value-end detection via bonus token).
# ============================================================
class XGrammarStateMachine:
    def __init__(self, schema, tok, attributes):
        self.tok = tok
        self.attributes = attributes
        if not isinstance(schema, str):
            schema = json.dumps(schema)
        tokinfo = xgr.TokenizerInfo.from_huggingface(tok)
        self.tokinfo = tokinfo
        compiler = xgr.GrammarCompiler(tokinfo)
        self.compiled = compiler.compile_json_schema(
            schema, any_whitespace=True, strict_mode=True)
        self.matcher = xgr.GrammarMatcher(
            self.compiled, terminate_without_stop_token=True)
        self._bm_shape = xgr.get_bitmask_shape(1, tokinfo.vocab_size)
        self.last_reason = ""

    def _get_bitmask(self, matcher=None):
        m = matcher or self.matcher
        bm = xgr.allocate_token_bitmask(1, self.tokinfo.vocab_size)
        m.fill_next_token_bitmask(bm, 0)
        return bm

    def _in_bitmask(self, bm, token_id):
        word = token_id // 32
        bit = token_id % 32
        if word >= bm.shape[1]:
            return False
        return bool(int(bm[0, word].item()) & (1 << bit))

    def sync_to(self, token_ids):
        self.matcher.reset()
        for t in token_ids:
            if not self.matcher.accept_token(int(t)):
                break

    def is_value_ended(self, fi, bonus_token):
        bonus_token = int(bonus_token)
        bm_in = self._get_bitmask()
        if not self._in_bitmask(bm_in, bonus_token):
            self.last_reason = "notin_Ain"
            return True
        p = self.matcher.fork()
        if not p.accept_token(bonus_token):
            self.last_reason = "accept_reject"
            return True
        bm_after = self._get_bitmask(p)
        if not torch.equal(bm_in, bm_after):
            self.last_reason = "state_changed"
            return True
        btxt = self.tok.decode([bonus_token])
        if btxt.strip() == "":
            self.last_reason = "ws_separator"
            return True
        self.last_reason = "value_continuation"
        return False

    def is_completed(self):
        return self.matcher.is_completed()


# ============================================================
# CachedTarget: prefill once + incremental advance + checkpoint/rollback.
# ============================================================
class CachedTarget:
    def __init__(self, model, encoded, initial_completion, device):
        self.model = model
        self.device = device
        ids, mask, kwargs = DCC.model_inputs(encoded, initial_completion, device)
        sync(device)
        with torch.inference_mode():
            out = model(
                input_ids=ids, attention_mask=mask, output_hidden_states=True,
                use_cache=True, logits_to_keep=1, return_dict=True, **kwargs,
            )
        sync(device)
        self.cache = out.past_key_values
        self.total = int(ids.shape[1])
        self.next_logits = out.logits[0, -1].float()
        self.rope_delta = model.model.rope_deltas.to(device=device, dtype=torch.long)
        self.history = [out.hidden_states[layer + 1] for layer in LAYERS]

    @torch.inference_mode()
    def advance(self, tokens, *, hidden=True):
        if not tokens:
            return 0.0
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
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.total = after
        self.next_logits = out.logits[0, -1].float()
        if hidden:
            self.history = [
                torch.cat([old, out.hidden_states[layer + 1]], dim=1)
                for old, layer in zip(self.history, LAYERS, strict=True)
            ]
        return elapsed_ms

    def checkpoint(self):
        sync(self.device)
        return (snapshot_cache(self.cache), self.total, self.next_logits.clone())

    def rollback(self, cp):
        saved, total, next_logits = cp
        restore_cache(self.cache, saved)
        self.total = total
        self.next_logits = next_logits


# ============================================================
# Draft: generalized for non-contiguous selected fields.
#
# Replicates DChordModel.propose_block but accepts arbitrary field_ids
# (the shared propose_block hardcodes arange(field_start, +block_size)).
# For contiguous selection the results are identical.
# ============================================================
@torch.inference_mode()
def draft_choices_grouped(dchord, hidden_states, selected_fields,
                           absolute_positions, pairs, device):
    block_size = len(selected_fields)
    field_ids = torch.tensor(selected_fields, device=device, dtype=torch.long)
    draft_pos = torch.tensor([absolute_positions], dtype=torch.long, device=device)
    transferred = [h.to(device) for h in hidden_states]
    context_feature = dchord.draft_model.extract_context_feature(transferred)
    bsz, seq_len, _ = context_feature.shape
    mask_ids = torch.full(
        (bsz, block_size), dchord.draft_model.mask_token_id,
        dtype=torch.long, device=device)
    base = dchord.draft_model.embed_tokens(mask_ids)
    schema_anchor = dchord._schema_anchor()[field_ids]
    delta = dchord.draft_model.dchord_schema_delta(field_ids)
    gate = dchord.draft_model.dchord_schema_gate.to(base.dtype)
    noise_embedding = base + gate * schema_anchor.unsqueeze(0) + delta.unsqueeze(0)
    context_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
    online_mask = torch.ones(
        bsz, 1, block_size, seq_len + block_size,
        dtype=torch.bool, device=device)
    sync(device)
    started = time.perf_counter()
    draft_hidden = dchord.draft_model(
        draft_input_ids=None, context_feature=context_feature,
        draft_position_ids=draft_pos, context_position_ids=context_positions,
        block_mask=online_mask, noise_embedding=noise_embedding,
    )
    logits = F.linear(draft_hidden, dchord.draft_model.embed_tokens.weight).float()
    sync(device)
    elapsed_ms = (time.perf_counter() - started) * 1000
    choices = []
    for offset in range(block_size):
        choices.append(branch(logits[0, offset], pairs[selected_fields[offset]], device))
    return choices, elapsed_ms


# ============================================================
# Span: contiguous from first to last selected field. Non-selected
# fields in between are pass-through (base_ids all-false, not verified).
# ============================================================
def grouped_span(base_ids, positions, field_start, field_end, proposals):
    span_end = positions[field_end] if field_end < len(positions) else len(base_ids)
    span = list(base_ids[positions[field_start]:span_end])
    for fi, prop in proposals.items():
        span[positions[fi] - positions[field_start]] = prop
    return span


# ============================================================
# Parallel verify: 4D tree-mask with per-subgraph grouping (P21).
#
# Islands -> isolated blocks (current behaviour). Connected-subgraph
# keys -> same group, causal chain (autoregressive KV reuse within the
# subgraph). The prefix is always fully attended. Only selected_fields
# are verified; non-selected pass-through fields keep base_ids values.
# ============================================================
@torch.inference_mode()
def tree_mask_verify_grouped(session, span_ids, positions, field_start, field_end,
                             selected_fields, dag, pairs, device):
    prefix_len = session.total
    span_len = len(span_ids)
    all_fields = list(range(field_start, field_end))
    val_off_all = [positions[fi] - positions[field_start] for fi in all_fields]
    val_tensor = torch.tensor(val_off_all, dtype=torch.long, device=device)
    s_range = torch.arange(span_len, device=device)
    # block_idx[s] = how many values strictly before s (which field's block)
    block_idx = (val_tensor[None, :] < s_range[:, None]).sum(dim=1).clamp(
        max=len(all_fields) - 1)
    # group per field (islands get unique negative, subgraphs shared positive)
    group_per_field = torch.tensor(
        [dag.group_id(dag.nodes[all_fields[i]]) for i in range(len(all_fields))],
        dtype=torch.long, device=device)
    group_per_pos = group_per_field[block_idx]  # (span_len,)
    same_group = group_per_pos[:, None] == group_per_pos[None, :]
    causal = s_range[:, None] >= s_range[None, :]

    mask = torch.zeros((span_len, prefix_len + span_len), dtype=torch.bool, device=device)
    mask[:, :prefix_len] = True
    mask[:, prefix_len:] = same_group & causal
    mask4d = mask[None, None, :, :]

    cache_position = torch.arange(prefix_len, prefix_len + span_len, device=device)
    position_ids = cache_position.view(1, 1, -1).expand(3, 1, -1)
    position_ids = position_ids + session.rope_delta.view(1, 1, 1)

    # logits_to_keep: predicting position of each selected value (except
    # value 0 which is from cursor) + bonus at last selected value.
    sel = sorted(selected_fields)
    sel_val_off = [positions[fi] - positions[field_start] for fi in sel]
    keep_list = []
    sel_logit_idx = {}  # fi -> index in out.logits
    for j, fi in enumerate(sel):
        if sel_val_off[j] > 0:
            sel_logit_idx[fi] = len(keep_list)
            keep_list.append(sel_val_off[j] - 1)
    keep_list.append(sel_val_off[-1])  # bonus at last selected value
    keep = torch.tensor(keep_list, dtype=torch.long, device=device)

    span_tensor = torch.tensor([span_ids], dtype=torch.long, device=device)
    cp = session.checkpoint()
    sync(device)
    started = time.perf_counter()
    out = session.model(
        input_ids=span_tensor, attention_mask=mask4d, position_ids=position_ids,
        past_key_values=session.cache, cache_position=cache_position,
        output_hidden_states=False, use_cache=False,
        logits_to_keep=keep, return_dict=True,
    )
    sync(device)
    elapsed_ms = (time.perf_counter() - started) * 1000
    session.rollback(cp)

    # Extract verifier choices for selected fields.
    choices = {}
    for fi in sel:
        vo = positions[fi] - positions[field_start]
        if vo == 0:
            choices[fi] = branch(session.next_logits, pairs[fi], device)
        else:
            idx = sel_logit_idx[fi]
            choices[fi] = branch(out.logits[0, idx], pairs[fi], device)
    bonus_token = int(out.logits[0, len(keep_list) - 1, :].argmax())
    return choices, bonus_token, elapsed_ms


# ============================================================
# Serial baseline: parallel-draft + serial verify (mismatch-stop).
# ============================================================
@torch.inference_mode()
def dchord_serial(target, dchord, encoded, base_ids, positions, pairs, attributes,
                  prompt_length, device, K=3):
    completion = list(base_ids[:positions[0]])
    branches = [-1] * len(positions)
    rounds = []
    total_verify_forwards = 0
    total_verify_ms = 0.0
    field = 0
    session = CachedTarget(target, encoded, completion, device)
    pending = []
    while field < len(positions):
        block_size = min(K, len(positions) - field)
        field_start = field
        advanced = bool(pending)
        context_ms = session.advance(pending)
        pending = []
        hidden_states = session.history
        abs_pos = [prompt_length + positions[field + i] for i in range(block_size)]
        sel = list(range(field, field + block_size))
        proposals_list, draft_ms = draft_choices_grouped(
            dchord, hidden_states, sel, abs_pos, pairs, device)
        proposals = {sel[i]: proposals_list[i] for i in range(block_size)}
        span, span_end = DCC.candidate_span(
            base_ids, positions, pairs, field, proposals_list)
        # serial verify: causal (attention_mask=None) over cached prefix
        verify_fields = list(range(field, field + block_size))
        span_tensor = torch.tensor([span], dtype=torch.long, device=device)
        cache_position = torch.arange(session.total, session.total + len(span), device=device)
        position_ids = cache_position.view(1, 1, -1).expand(3, 1, -1)
        position_ids = position_ids + session.rope_delta.view(1, 1, 1)
        val_off = [positions[verify_fields[i]] - positions[field] for i in range(block_size)]
        if block_size > 1:
            keep_list = [val_off[i] - 1 for i in range(1, block_size)] + [val_off[-1]]
        else:
            keep_list = [val_off[0]]
        keep = torch.tensor(keep_list, dtype=torch.long, device=device)
        cp = session.checkpoint()
        sync(device)
        v_started = time.perf_counter()
        v_out = session.model(
            input_ids=span_tensor, attention_mask=None, position_ids=position_ids,
            past_key_values=session.cache, cache_position=cache_position,
            output_hidden_states=False, use_cache=False,
            logits_to_keep=keep, return_dict=True,
        )
        sync(device)
        verify_ms = (time.perf_counter() - v_started) * 1000
        session.rollback(cp)
        total_verify_forwards += 1 + (1 if advanced else 0)
        total_verify_ms += verify_ms + context_ms
        verifier = [branch(session.next_logits, pairs[verify_fields[0]], device)]
        for i in range(1, block_size):
            verifier.append(branch(v_out.logits[0, i - 1], pairs[verify_fields[i]], device))
        mismatch = next(
            (o for o in range(block_size) if proposals_list[o] != verifier[o]), None)
        if mismatch is not None:
            rej_field = field + mismatch
            rej_pos = positions[rej_field]
            accept_span = list(span[:rej_pos - positions[field]])
            correction = verifier[mismatch]
            nf = rej_field + 1
            fixed_end = positions[nf] if nf < len(positions) else len(base_ids)
            fixed = list(base_ids[rej_pos + 1:fixed_end])
            new_tokens = accept_span + [correction] + fixed
            completion.extend(new_tokens)
            branches[rej_field] = pairs[rej_field].index(correction)
            for i in range(mismatch):
                branches[field + i] = pairs[field + i].index(proposals_list[i])
            pending = new_tokens
            field = rej_field + 1
            rounds.append({"result": f"reject@{mismatch}", "vms": verify_ms, "ctx_ms": context_ms})
        else:
            completion = completion + list(span)
            for i in range(block_size):
                branches[field + i] = pairs[field + i].index(proposals_list[i])
            pending = list(span)
            field += block_size
            rounds.append({"result": f"accept {block_size}", "vms": verify_ms, "ctx_ms": context_ms})
        if len(rounds) <= 20 or len(rounds) % 20 == 0:
            dk = [f"{attributes[field_start+i][:10]}={'T' if proposals_list[i]==pairs[field_start+i][1] else 'F'}" for i in range(block_size)]
            vk = [f"{attributes[field_start+i][:10]}={'T' if verifier[i]==pairs[field_start+i][1] else 'F'}" for i in range(block_size)]
            print(f"  r{len(rounds)-1}: draft={dk} verify={vk} -> {rounds[-1]['result']}", flush=True)
    return completion, branches, rounds, total_verify_forwards, total_verify_ms


# ============================================================
# Parallel: island-aware DFS/BFS selection + grouped tree-mask verify.
# ============================================================
@torch.inference_mode()
def dchord_parallel(target, dchord, encoded, base_ids, positions, pairs, attributes,
                    prompt_length, device, tokenizer, K=3, strategy="dfs",
                    dag_edges=None):
    dag = DAG(attributes, dag_edges)
    schema_def = {
        "type": "object",
        "properties": {a: {"type": "boolean"} for a in attributes},
        "required": list(attributes),
        "additionalProperties": False,
    }
    schema = XGrammarStateMachine(schema_def, tokenizer, attributes)
    select = (select_nodes_depth_first if strategy == "dfs"
              else select_nodes_breadth_first)
    remaining = set(attributes)
    verified = set()
    completion = list(base_ids[:positions[0]])
    branches = [-1] * len(positions)
    rounds = []
    total_verify_forwards = 0
    total_verify_ms = 0.0
    field_of = {a: i for i, a in enumerate(attributes)}
    in_progress = {}
    case_b_count = {}

    session = CachedTarget(target, encoded, completion, device)
    pending = []

    while remaining:
        # Stage 2: select nodes (DFS or BFS)
        patch = select(remaining, dag, verified, K)
        if not patch:
            break
        patch_sorted = sorted(patch, key=lambda a: (dag.pos[a], a))
        sel_fields = [field_of[a] for a in patch_sorted]
        field_start = min(sel_fields)
        field_end = max(sel_fields) + 1
        block_size = len(sel_fields)

        # Stage 1: context update
        advanced = bool(pending)
        context_ms = session.advance(pending)
        pending = []
        hidden_states = session.history

        # Stage 3: draft (for selected fields)
        abs_pos = [prompt_length + positions[fi] for fi in sel_fields]
        proposals_list, draft_ms = draft_choices_grouped(
            dchord, hidden_states, sel_fields, abs_pos, pairs, device)
        proposals = {sel_fields[i]: proposals_list[i] for i in range(block_size)}

        # Stage 4: parallel verify (grouped tree-mask)
        span = grouped_span(base_ids, positions, field_start, field_end, proposals)
        verifier, bonus_token, verify_ms = tree_mask_verify_grouped(
            session, span, positions, field_start, field_end,
            sel_fields, dag, pairs, device)
        total_verify_forwards += 1 + (1 if advanced else 0)
        total_verify_ms += verify_ms + context_ms

        # Stage 5: state update
        span_v = list(span)
        n_accept = n_reject = 0
        for fi in sel_fields:
            val_off = positions[fi] - positions[field_start]
            if proposals[fi] == verifier[fi]:
                branches[fi] = pairs[fi].index(proposals[fi])
                n_accept += 1
            else:
                branches[fi] = pairs[fi].index(verifier[fi])
                span_v[val_off] = verifier[fi]
                n_reject += 1

        # bonus boundary check on last selected field
        last_fi = max(sel_fields)
        last_key = attributes[last_fi]
        last_val_off = positions[last_fi] - positions[field_start]
        bonus_repr = repr(tokenizer.decode([bonus_token]))
        schema.sync_to(completion + span_v[:last_val_off + 1])
        ended = schema.is_value_ended(last_fi, bonus_token)

        if ended:
            for fi in sel_fields:
                verified.add(attributes[fi])
                remaining.discard(attributes[fi])
            in_progress.pop(last_key, None)
            completion = completion + span_v
            pending = span_v
            result = f"accept {n_accept}, reject {n_reject} (Case A)"
        else:
            for fi in sel_fields[:-1]:
                verified.add(attributes[fi])
                remaining.discard(attributes[fi])
            prefix = span_v[:last_val_off + 1]
            completion = completion + prefix
            pending = prefix
            in_progress[last_key] = in_progress.get(last_key, []) + [verifier[last_fi]]
            case_b_count[last_key] = case_b_count.get(last_key, 0) + 1
            if case_b_count[last_key] > 1:
                verified.add(last_key)
                remaining.discard(last_key)
                in_progress.pop(last_key, None)
                result = f"accept {n_accept}, reject {n_reject} (Case B->force)"
            else:
                result = f"accept {n_accept}, reject {n_reject} (Case B)"
        rounds.append({"result": result, "vms": verify_ms, "ctx_ms": context_ms})
        if len(rounds) <= 20 or len(rounds) % 20 == 0:
            dk = [f"{attributes[fi][:10]}={'T' if proposals[fi]==pairs[fi][1] else 'F'}" for fi in sel_fields]
            vk = [f"{attributes[fi][:10]}={'T' if verifier[fi]==pairs[fi][1] else 'F'}" for fi in sel_fields]
            print(f"  r{len(rounds)-1}: draft={dk} verify={vk} bonus={bonus_repr} "
                  f"ended={ended}[{schema.last_reason}] -> {result} "
                  f"remaining={len(remaining)}", flush=True)
    return completion, branches, rounds, total_verify_forwards, total_verify_ms


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mode", type=str, default="parallel", choices=["serial", "parallel"])
    parser.add_argument("--strategy", type=str, default="dfs", choices=["dfs", "bfs"])
    args = parser.parse_args()
    device = torch.device(args.device)

    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    import pyarrow.parquet as pq
    import io
    tokenizer = AutoTokenizer.from_pretrained(TARGET, local_files_only=True)
    processor = AutoProcessor.from_pretrained(TARGET, local_files_only=True)
    pcs = json.loads((ASSETS / "profiled_surface_spec_v1.json").read_text())
    attributes = list(pcs["schema"]["attributes"])
    segments = list(pcs["surface"]["segments"])
    base_ids, positions, pairs = DCC.build_contract(tokenizer, segments, attributes)
    print(f"DAG: {len(attributes)} attrs, {len(positions)} value positions, "
          f"strategy={args.strategy}")

    target = AutoModelForImageTextToText.from_pretrained(
        TARGET, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True,
        attn_implementation="sdpa").to(device).eval()
    dchord = DCC.build_dchord(tokenizer, attributes, pairs, device, EXPORT)
    print("models loaded")

    TRAIN_PARQUET = str(ROOT / "cache/hf/hub/datasets--huggan--CelebA-faces-with-attributes/snapshots/b47e27a7c6bc578361ce132da8c8dad573b98d9e/data/train-00000-of-00132.parquet")
    pf = pq.ParquetFile(TRAIN_PARQUET)
    prompt = (ASSETS / "prompt_schema_oneshot_qwen35_v1.txt").read_text()
    all_results = []
    wall0 = time.perf_counter()
    for idx in range(args.limit):
        row = pf.read_row_group(0).slice(idx, 1).to_pylist()[0]
        image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        labels = [1 if row[a] == 1 else 0 for a in attributes]
        encoded = DCC.processor_inputs(processor, image, prompt)
        prompt_length = int(encoded["input_ids"].shape[1])
        print(f"\n=== sample {idx} [{args.mode}/{args.strategy}] ===")
        common_args = (target, dchord, encoded, base_ids, positions, pairs, attributes,
                       prompt_length, device)
        if args.mode == "serial":
            completion, branches, rounds, n_vf, vms = dchord_serial(*common_args, K=3)
        else:
            completion, branches, rounds, n_vf, vms = dchord_parallel(
                *common_args, tokenizer, K=3, strategy=args.strategy)
        correct = sum(1 for i in range(len(branches))
                      if i < len(labels) and branches[i] == labels[i])
        print(f"  rounds={len(rounds)}, verify_forwards={n_vf}, "
              f"correct={correct}/{len(branches)}, verify_ms={vms:.1f}")
        all_results.append({"idx": idx, "rounds": len(rounds),
                            "verify_forwards": n_vf, "correct": correct, "vms": vms})

    wall = time.perf_counter() - wall0
    n = len(all_results)
    print(f"\n=== Summary [{args.mode}/{args.strategy}] ({n} samples, wall={wall:.2f}s) ===")
    print(f"  mean rounds: {sum(r['rounds'] for r in all_results)/n:.2f}")
    print(f"  mean verify_forwards: {sum(r['verify_forwards'] for r in all_results)/n:.2f}")
    print(f"  mean correct: {sum(r['correct'] for r in all_results)/n:.2f}/{len(branches)}")
    print(f"  total verify_ms: {sum(r['vms'] for r in all_results):.1f}")
    print("DONE")


if __name__ == "__main__":
    main()
