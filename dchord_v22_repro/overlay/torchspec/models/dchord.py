"""DChord K-way schema-value training objective for TorchSpec online training."""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchspec.models.dflash import _dpace_position_weights


class DChordModel(nn.Module):
    """Predict only schema values while fixed structure is supplied by a compiler.

    The target-side context is the ordinary full prompt plus compiled assistant
    response produced by TorchSpec's inference engine.  For every group of K
    consecutive schema fields, all K value queries see the target context before
    the first value in the group and see one another bidirectionally.  They cannot
    see any target token at or after that first value, which prevents future-label
    leakage.
    """

    def __init__(
        self,
        draft_model,
        schema_token_ids: list[list[int]],
        branch_token_ids: list[list[int]],
        k: int = 3,
        num_anchors: int | None = None,
        dpace_alpha: float = 0.5,
    ):
        super().__init__()
        if k <= 0:
            raise ValueError(f"DChord K must be positive, got {k}")
        if len(schema_token_ids) != len(branch_token_ids):
            raise ValueError("schema and branch tables must have the same field count")
        if any(len(pair) != 2 or pair[0] == pair[1] for pair in branch_token_ids):
            raise ValueError("CelebA DChord requires two distinct branch tokens per field")

        self.draft_model = draft_model
        self.k = int(k)
        self.num_anchors = int(num_anchors or math.ceil(len(schema_token_ids) / self.k))
        if not 0 < self.num_anchors <= len(schema_token_ids):
            raise ValueError(
                f"DChord num_anchors must be in [1, {len(schema_token_ids)}], "
                f"got {self.num_anchors}"
            )
        self.dpace_alpha = float(dpace_alpha)
        self.num_fields = len(schema_token_ids)

        max_tokens = max(len(ids) for ids in schema_token_ids)
        ids = torch.zeros(self.num_fields, max_tokens, dtype=torch.long)
        keep = torch.zeros(self.num_fields, max_tokens, dtype=torch.bool)
        for index, values in enumerate(schema_token_ids):
            ids[index, : len(values)] = torch.tensor(values, dtype=torch.long)
            keep[index, : len(values)] = True
        self.register_buffer("schema_token_ids", ids, persistent=True)
        self.register_buffer("schema_token_mask", keep, persistent=True)
        self.register_buffer(
            "branch_token_ids", torch.tensor(branch_token_ids, dtype=torch.long), persistent=True
        )

        # Attach adapters to draft_model so TorchSpec's existing BF16 optimizer
        # and checkpoint path include them without a parallel optimizer stack.
        dtype = draft_model.embed_tokens.weight.dtype
        device = draft_model.embed_tokens.weight.device
        if not hasattr(draft_model, "dchord_schema_delta"):
            draft_model.dchord_schema_delta = nn.Embedding(
                self.num_fields, draft_model.hidden_size, device=device, dtype=dtype
            )
            nn.init.zeros_(draft_model.dchord_schema_delta.weight)
        if not hasattr(draft_model, "dchord_schema_gate"):
            draft_model.dchord_schema_gate = nn.Parameter(
                torch.tensor(0.1, device=device, dtype=dtype)
            )

    @property
    def uses_target_hidden_states(self) -> bool:
        return False

    def _schema_anchor(self) -> torch.Tensor:
        embeddings = self.draft_model.embed_tokens(self.schema_token_ids)
        weights = self.schema_token_mask.unsqueeze(-1).to(embeddings.dtype)
        return (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

    def _value_positions(self, input_ids: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        """Locate and validate the 40 contextual schema-value tokens per sample."""
        positions = []
        allowed = self.branch_token_ids
        for batch_index in range(input_ids.shape[0]):
            supervised = loss_mask[batch_index] > 0
            candidates = supervised & (
                (input_ids[batch_index, :, None, None] == allowed[None, :, :]).any(dim=(1, 2))
            )
            found = torch.nonzero(candidates, as_tuple=False).flatten()
            if found.numel() != self.num_fields:
                raise ValueError(
                    f"DChord expected {self.num_fields} schema values, found {found.numel()} "
                    f"in batch item {batch_index}"
                )
            observed = input_ids[batch_index, found]
            field_valid = (observed[:, None] == allowed).any(dim=1)
            if not bool(field_valid.all()):
                bad = torch.nonzero(~field_valid, as_tuple=False).flatten().tolist()
                raise ValueError(f"DChord field-token order mismatch at fields {bad}")
            positions.append(found)
        return torch.stack(positions, dim=0)

    def _sample_field_anchors(self, bsz: int, device: torch.device) -> torch.Tensor:
        """Sample unique decision starts per sample, matching DFlash random anchors."""
        scores = torch.rand(bsz, self.num_fields, device=device)
        anchors = scores.argsort(dim=1)[:, : self.num_anchors]
        return anchors.sort(dim=1).values

    def propose_block(
        self,
        hidden_states_list: List[torch.Tensor],
        field_start: int,
        draft_position_ids: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Propose one online decision block without teacher-forced future context.

        ``hidden_states_list`` contains only the committed target prefix.  The
        compiler supplies the absolute positions of the value tokens that the
        current block would occupy after fixed structure is expanded.  All
        queries see the committed prefix and one another, exactly matching one
        block of the training mask, but no later target states exist in this
        online path.
        """
        if not 0 <= field_start < self.num_fields:
            raise ValueError(f"field_start out of range: {field_start}")
        if draft_position_ids.ndim != 2:
            raise ValueError("draft_position_ids must have shape [batch, block]")
        block_size = draft_position_ids.shape[1]
        expected = min(self.k, self.num_fields - field_start)
        if block_size != expected:
            raise ValueError(f"expected online block size {expected}, got {block_size}")

        context_feature = self.draft_model.extract_context_feature(hidden_states_list)
        bsz, sequence_length, _ = context_feature.shape
        if draft_position_ids.shape[0] != bsz:
            raise ValueError("draft positions and target context batch sizes differ")
        device = context_feature.device
        field_ids = torch.arange(
            field_start, field_start + block_size, device=device, dtype=torch.long
        )
        mask_ids = torch.full(
            (bsz, block_size), self.draft_model.mask_token_id, dtype=torch.long, device=device
        )
        base = self.draft_model.embed_tokens(mask_ids)
        schema_anchor = self._schema_anchor()[field_ids]
        delta = self.draft_model.dchord_schema_delta(field_ids)
        gate = self.draft_model.dchord_schema_gate.to(base.dtype)
        noise_embedding = base + gate * schema_anchor.unsqueeze(0) + delta.unsqueeze(0)

        context_positions = torch.arange(sequence_length, device=device).unsqueeze(0).expand(bsz, -1)
        online_mask = torch.ones(
            bsz,
            1,
            block_size,
            sequence_length + block_size,
            dtype=torch.bool,
            device=device,
        )
        draft_hidden = self.draft_model(
            draft_input_ids=None,
            context_feature=context_feature,
            draft_position_ids=draft_position_ids.to(device),
            context_position_ids=context_positions,
            block_mask=online_mask,
            noise_embedding=noise_embedding,
        )
        return F.linear(draft_hidden, lm_head_weight).float()

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
        last_hidden_states: torch.Tensor | None = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        del last_hidden_states
        bsz, sequence_length = input_ids.shape
        device = input_ids.device
        value_positions = self._value_positions(input_ids, loss_mask)
        context_feature = self.draft_model.extract_context_feature(hidden_states_list)

        blocks = self.num_anchors
        padded = blocks * self.k
        anchor_fields = self._sample_field_anchors(bsz, device)
        offsets = torch.arange(self.k, device=device).view(1, 1, self.k)
        raw_field_ids = anchor_fields.unsqueeze(-1) + offsets
        valid_mask = raw_field_ids < self.num_fields
        field_ids = raw_field_ids.clamp(max=self.num_fields - 1)
        flat_field_ids = field_ids.view(bsz, padded)
        query_positions = torch.gather(value_positions, 1, flat_field_ids)
        anchors = torch.gather(value_positions, 1, anchor_fields)

        mask_ids = torch.full(
            (bsz, padded), self.draft_model.mask_token_id, dtype=torch.long, device=device
        )
        base = self.draft_model.embed_tokens(mask_ids)
        schema_anchor = self._schema_anchor()[flat_field_ids]
        delta = self.draft_model.dchord_schema_delta(flat_field_ids)
        gate = self.draft_model.dchord_schema_gate.to(base.dtype)
        noise_embedding = base + gate * schema_anchor + delta

        context_positions = torch.arange(sequence_length, device=device).unsqueeze(0).expand(bsz, -1)
        block_mask = torch.zeros(
            bsz,
            1,
            padded,
            sequence_length + padded,
            dtype=torch.bool,
            device=device,
        )
        for block_index in range(blocks):
            q_begin = block_index * self.k
            q_end = q_begin + self.k
            for batch_index in range(bsz):
                anchor = int(anchors[batch_index, block_index])
                block_mask[batch_index, :, q_begin:q_end, :anchor] = True
            block_mask[:, :, q_begin:q_end, sequence_length + q_begin : sequence_length + q_end] = True

        draft_hidden = self.draft_model(
            draft_input_ids=None,
            context_feature=context_feature,
            draft_position_ids=query_positions,
            context_position_ids=context_positions,
            block_mask=block_mask,
            noise_embedding=noise_embedding,
        )
        logits = F.linear(draft_hidden, lm_head_weight).float().view(bsz, blocks, self.k, -1)

        gathered = torch.gather(input_ids, 1, value_positions)
        targets = torch.gather(gathered, 1, flat_field_ids)
        targets = targets.view(bsz, blocks, self.k)

        ce = F.cross_entropy(
            logits.flatten(0, 2), targets.flatten(), reduction="none"
        ).view(bsz, blocks, self.k)
        with torch.no_grad():
            confidence = torch.exp(-ce.detach())
            position_weights = _dpace_position_weights(confidence, alpha=self.dpace_alpha)
        weights = position_weights * valid_mask.float()
        numerator = (ce * weights).sum()
        denominator = weights.sum().clamp_min(1.0)
        loss = numerator / denominator

        predictions = logits.argmax(dim=-1)
        correct = predictions.eq(targets) & valid_mask
        # Read-only evaluation hook; excluded from state_dict and gradients.
        self.last_correct = correct.detach()
        self.last_predictions = predictions.detach()
        self.last_targets = targets.detach()
        self.last_anchor_fields = anchor_fields.detach()
        count_per_position = valid_mask.sum(dim=(0, 1)).float()
        loss_per_position = (ce * valid_mask).sum(dim=(0, 1)) / count_per_position.clamp_min(1)
        acc_per_position = correct.sum(dim=(0, 1)).float() / count_per_position.clamp_min(1)
        accuracy = correct.sum().float() / valid_mask.sum().clamp_min(1)

        return (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            {},
            (numerator, denominator),
        )
