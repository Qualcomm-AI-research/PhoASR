# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Alignment text normalization."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .backends.seq2seq import generate_text
from .config import PipelineConfig
from .io import ensure_object_columns, relativize_to_repo
from .mapping_utils import create_alignment_input, valid_input


def apply_num2word(df: pd.DataFrame, config: PipelineConfig, stage_dir: str | Path) -> pd.DataFrame:
    """Normalize transcripts for alignment and persist num2word mappings when needed."""
    out = ensure_object_columns(
        df.copy(),
        [
            "status",
            "drop_reason",
            "text_normalized_for_alignment",
            "alignment_input_text",
            "mapping_json_path",
        ],
    )

    backend = config.get("num2word.model", {}) or {}
    mappings_dir = Path(stage_dir) / "mappings"
    mappings_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in out.iterrows():
        if row["status"] != "active":
            continue
        base_text = (
            row["text_with_punc"]
            if isinstance(row["text_with_punc"], str) and row["text_with_punc"]
            else row["transcript"]
        )
        if any(ch.isdigit() for ch in base_text):
            normalized = generate_text(base_text, backend, sample_id=row["sample_id"])
        else:
            normalized = base_text
        out.at[idx, "text_normalized_for_alignment"] = normalized

        if valid_input(base_text):
            mappings, alignment_input, has_empty_mapping = create_alignment_input(
                base_text, normalized
            )
            if has_empty_mapping:
                out.at[idx, "status"] = "dropped"
                out.at[idx, "drop_reason"] = "empty_num2word_mapping"
                out.at[idx, "alignment_input_text"] = ""
                out.at[idx, "mapping_json_path"] = ""
                continue
            mapping_path = mappings_dir / f"{row['sample_id']}_mapping.json"
            with mapping_path.open("w", encoding="utf-8") as handle:
                json.dump(mappings, handle, ensure_ascii=False, indent=2)
            out.at[idx, "alignment_input_text"] = alignment_input
            out.at[idx, "mapping_json_path"] = relativize_to_repo(mapping_path, config.repo_root)
        else:
            out.at[idx, "alignment_input_text"] = base_text
            out.at[idx, "mapping_json_path"] = ""
    return out
