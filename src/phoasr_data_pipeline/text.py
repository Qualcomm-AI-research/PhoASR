# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Text normalization utilities."""

from __future__ import annotations

import re


def normalize_for_compare(text: str | None) -> str:
    """Lowercase text and strip punctuation for comparison-oriented checks."""
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def normalize_for_punc_match(text: str | None) -> str:
    """Normalize text before checking punctuation-model output against its input."""
    return normalize_for_compare(text)


def simple_wer(reference: str | None, hypothesis: str | None) -> float | None:
    """Compute a lightweight word error rate over normalized tokens."""
    ref_tokens = normalize_for_compare(reference).split()
    hyp_tokens = normalize_for_compare(hypothesis).split()
    if not ref_tokens:
        return None
    distance = _edit_distance(ref_tokens, hyp_tokens)
    return distance / len(ref_tokens)


def _edit_distance(a: list[str], b: list[str]) -> int:
    """Compute Levenshtein distance between two token sequences."""
    prev = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        curr = [i]
        for j, token_b in enumerate(b, start=1):
            cost = 0 if token_a == token_b else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]
