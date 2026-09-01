"""TorchSpec trainer for the DChord schema-value objective."""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

from torchspec.models.dchord import DChordModel
from torchspec.training.dflash_trainer import DFlashTrainer
from torchspec.utils.logging import logger


class DChordTrainer(DFlashTrainer):
    """DFlash-derived online trainer whose prediction unit is a schema value."""

    _anchor_slot_offset = 0

    def _build_training_wrapper(self, draft_model):
        pcs_path = Path(self.args.dchord_profiled_surface_path)
        pcs = json.loads(pcs_path.read_text())
        attributes = list(pcs["schema"]["attributes"])
        segments = list(pcs["surface"]["segments"])
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.target_model_path,
            trust_remote_code=getattr(self.args, "trust_remote_code", True),
            local_files_only=True,
        )
        schema_ids = [
            [int(token) for token in tokenizer.encode(f'"{field}"', add_special_tokens=False)]
            for field in attributes
        ]

        def render(value: bool) -> str:
            text = []
            for index, segment in enumerate(segments):
                text.append(segment)
                if index < len(attributes):
                    text.append("true" if value else "false")
            return "".join(text)

        false_ids = tokenizer.encode(render(False), add_special_tokens=False)
        true_ids = tokenizer.encode(render(True), add_special_tokens=False)
        if len(false_ids) != len(true_ids):
            raise ValueError("DChord boolean branches change the compiled token length")
        changed = [index for index, pair in enumerate(zip(false_ids, true_ids)) if pair[0] != pair[1]]
        if len(changed) != len(attributes):
            raise ValueError(
                f"expected {len(attributes)} contextual boolean token positions, found {len(changed)}"
            )
        branch_ids = [[int(false_ids[index]), int(true_ids[index])] for index in changed]
        logger.info(
            "DChord compiler contract: fields=%d K=%d random_anchors=%d "
            "branch_positions=%d PCS=%s",
            len(attributes),
            self.args.dchord_k,
            self.args.dchord_num_anchors,
            len(changed),
            pcs_path,
        )
        return DChordModel(
            draft_model=draft_model,
            schema_token_ids=schema_ids,
            branch_token_ids=branch_ids,
            k=self.args.dchord_k,
            num_anchors=self.args.dchord_num_anchors,
            dpace_alpha=self.dpace_alpha,
        )
