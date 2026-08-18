# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Alignment artifact generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .backends.alignment import align_audio_text
from .config import PipelineConfig
from .io import ensure_object_columns, relativize_to_repo


def align_samples(df: pd.DataFrame, config: PipelineConfig, stage_dir: str | Path) -> pd.DataFrame:
    """Generate timestamp JSON artifacts for active rows."""
    out = ensure_object_columns(
        df.copy(), ["status", "drop_reason", "timestamp_json_path", "alignment_input_text"]
    )
    backend = config.get("alignment.backend", {}) or {}
    artifacts_dir = Path(stage_dir) / "timestamps"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in out.iterrows():
        if row["status"] != "active":
            continue
        text = (
            row["alignment_input_text"]
            if isinstance(row.get("alignment_input_text"), str) and row.get("alignment_input_text")
            else row["text_normalized_for_alignment"]
        )
        out.at[idx, "alignment_input_text"] = text
        alignment = run_alignment_backend(row["sample_id"], row["audio_path"], text, backend)
        reconstructed = " ".join(segment["word"] for segment in alignment["segments"])
        if reconstructed != text:
            out.at[idx, "status"] = "dropped"
            out.at[idx, "drop_reason"] = "alignment_reconstruction_mismatch"
            continue
        json_path = artifacts_dir / f"{row['sample_id']}.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(alignment, handle, ensure_ascii=False, indent=2)
        out.at[idx, "timestamp_json_path"] = relativize_to_repo(json_path, config.repo_root)
    return out


def run_alignment_backend(sample_id: str, audio_path: str, text: str, backend: dict) -> dict:
    """Run the configured alignment backend for one sample."""
    del sample_id
    return align_audio_text(audio_path, text, backend)


def build_even_alignment(audio_path: str, text: str, duration: float) -> dict:
    """Build a synthetic evenly spaced alignment artifact for tests."""
    words = [token for token in text.split() if token]
    if not words:
        words = [""]
    step = duration / max(len(words), 1)
    segments = []
    for idx, word in enumerate(words):
        start = round(idx * step, 2)
        end = round((idx + 1) * step, 2)
        segments.append({"word": word, "start": start, "end": end})
    return {
        "audio_path": str(audio_path),
        "start": 0.0,
        "end": round(duration, 2),
        "text": text,
        "segments": segments,
    }
