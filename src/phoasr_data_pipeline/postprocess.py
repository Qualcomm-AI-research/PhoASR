# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Final formatting helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .io import ensure_object_columns, relativize_to_repo, resolve_repo_relative_path
from .mapping_utils import aggregate_timestamps


def finalize_dataset(
    df: pd.DataFrame, stage_dir: str | Path, repo_root: str | Path
) -> pd.DataFrame:
    """Rewrite final timestamp artifacts against the original text surface."""
    out = ensure_object_columns(
        df.copy(),
        [
            "final_transcript",
            "timestamp_json_path",
            "mapping_json_path",
            "text_with_punc",
            "transcript",
        ],
    )

    final_timestamps_dir = Path(stage_dir) / "timestamps"
    final_timestamps_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in out.iterrows():
        if row["status"] != "active":
            continue
        timestamp_path = row["timestamp_json_path"]
        if not isinstance(timestamp_path, str) or not timestamp_path:
            continue
        with resolve_repo_relative_path(timestamp_path, repo_root).open(
            "r", encoding="utf-8"
        ) as handle:
            payload = json.load(handle)

        mapping_path = row.get("mapping_json_path", "")
        if isinstance(mapping_path, str) and mapping_path:
            with resolve_repo_relative_path(mapping_path, repo_root).open(
                "r", encoding="utf-8"
            ) as handle:
                mappings = json.load(handle)
            aggregated = aggregate_timestamps(payload, mappings)
            new_segments = []
            for segment in aggregated:
                new_segments.append(
                    {
                        "word": " ".join(segment["words"]),
                        "start": round(float(segment["start"]), 2),
                        "end": round(float(segment["end"]), 2),
                    }
                )
            payload["segments"] = new_segments
            payload["text"] = " ".join(segment["word"] for segment in new_segments)
        else:
            final_text = (
                row["text_with_punc"]
                if isinstance(row["text_with_punc"], str) and row["text_with_punc"]
                else row["transcript"]
            )
            payload["text"] = final_text
            for segment in payload.get("segments", []):
                segment["start"] = round(float(segment["start"]), 2)
                segment["end"] = round(float(segment["end"]), 2)

        final_path = final_timestamps_dir / Path(timestamp_path).name
        with final_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        out.at[idx, "final_transcript"] = payload.get("text", row.get("final_transcript", ""))
        out.at[idx, "timestamp_json_path"] = relativize_to_repo(final_path, repo_root)
    return out
