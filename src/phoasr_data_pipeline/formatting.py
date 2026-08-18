# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Punctuation and capitalization restoration."""

from __future__ import annotations

import pandas as pd

from .backends.seq2seq import generate_text
from .config import PipelineConfig
from .io import ensure_object_columns
from .text import normalize_for_punc_match


def apply_punctuation_and_capitalization(
    df: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Apply punctuation/capitalization recovery."""
    out = ensure_object_columns(df.copy(), ["status", "drop_reason", "text_with_punc"])
    backend = config.get("punc_cap.model", {}) or {}
    drop_on_mismatch = bool(config.get("punc_cap.drop_on_mismatch", True))
    for idx, row in out.iterrows():
        if row["status"] != "active":
            continue
        source_text = row["transcript"]
        model_input_text = source_text.lower() if isinstance(source_text, str) else source_text
        restored = generate_text(model_input_text, backend, sample_id=row["sample_id"])
        if normalize_for_punc_match(restored) != normalize_for_punc_match(model_input_text):
            if drop_on_mismatch:
                out.at[idx, "status"] = "dropped"
                out.at[idx, "drop_reason"] = "punc_cap_mismatch"
                continue
        out.at[idx, "text_with_punc"] = restored
    return out
