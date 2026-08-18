# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Standalone seq2seq backends used by punctuation and normalization."""

from __future__ import annotations

from typing import Any

import torch

_SEQ2SEQ_MODELS: dict[tuple[str, str], Any] = {}


def generate_text(text: str, backend: dict, sample_id: str | None = None) -> str:
    """Generate normalized text with the configured seq2seq backend."""
    del sample_id
    backend_type = backend.get("type")
    if backend_type == "seq2seq":
        return _run_seq2seq(text, backend)
    if backend_type == "command":
        raise ValueError("command backends are no longer supported in the standalone package")
    raise ValueError(
        "Unsupported text backend type: " f"{backend_type!r}. Supported text backend is `seq2seq`."
    )


def _run_seq2seq(text: str, backend: dict) -> str:
    """Run one seq2seq generation call, caching models by path and device."""
    model_id_or_path = backend.get("model_id_or_path")
    if not model_id_or_path:
        raise ValueError("seq2seq backend requires `model_id_or_path`")
    device = backend.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    num_beams = int(backend.get("num_beams", 5))
    max_length = int(backend.get("max_length", 1024))
    key = (str(model_id_or_path), str(device))
    bundle = _SEQ2SEQ_MODELS.get(key)
    if bundle is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id_or_path)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            model = model.to(device)
        bundle = (tokenizer, model, device)
        _SEQ2SEQ_MODELS[key] = bundle
    tokenizer, model, resolved_device = bundle
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    if str(resolved_device).startswith("cuda") and torch.cuda.is_available():
        inputs = {name: tensor.to(resolved_device) for name, tensor in inputs.items()}
    with torch.no_grad():
        generated = model.generate(**inputs, max_length=max_length, num_beams=num_beams)
    return tokenizer.decode(generated[0], skip_special_tokens=True).strip()
