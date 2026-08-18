# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Standalone alignment backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch

_ALIGNMENT_BUNDLES: dict[tuple[str, str, str], Any] = {}
_PARAKEET_ALIGN_MODELS: dict[tuple[str, str], Any] = {}


def align_audio_text(audio_path: str, text: str, backend: dict) -> dict:
    """Align one audio file against one transcript with the configured backend."""
    backend_type = backend.get("type")
    if backend_type == "whisperx_align":
        return _run_whisperx_alignment(audio_path, text, backend)
    if backend_type == "parakeet_nemo_align":
        return _run_parakeet_nemo_alignment(audio_path, text, backend)
    if backend_type == "command":
        raise ValueError("command backends are no longer supported in the standalone package")
    raise ValueError(
        "Unsupported alignment backend type: "
        f"{backend_type!r}. Supported alignment backends are `whisperx_align` and `parakeet_nemo_align`."
    )


def _find_start_end(audio_path: str) -> tuple[float, float]:
    """Estimate trimmed speech boundaries for an audio file."""
    y, sr = librosa.load(audio_path, sr=None)
    _, index = librosa.effects.trim(y, top_db=20)
    return index[0] / sr, index[1] / sr


def _run_whisperx_alignment(audio_path: str, text: str, backend: dict) -> dict:
    """Run WhisperX alignment and return the package's timestamp payload shape."""
    import whisperx

    device = backend.get("device", "cpu")
    language = backend.get("language", "vi")
    align_model_id = backend.get("align_model_id", "khanhld/wav2vec2-base-vietnamese-160h")
    sample_rate = int(backend.get("sample_rate", 16000))
    key = (str(language), str(device), str(align_model_id))
    bundle = _ALIGNMENT_BUNDLES.get(key)
    if bundle is None:
        model_a, metadata = whisperx.load_align_model(
            language_code=language,
            device=device,
            model_name=align_model_id,
        )
        bundle = (model_a, metadata)
        _ALIGNMENT_BUNDLES[key] = bundle
    model_a, metadata = bundle
    audio = whisperx.load_audio(audio_path, sr=sample_rate)
    start_time, end_time = _find_start_end(audio_path)
    alignment_input = [{"text": text, "start": round(start_time, 2), "end": round(end_time, 2)}]
    result_align = whisperx.align(
        alignment_input,
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    segments = []
    for segment in result_align.get("segments", []):
        segments.extend(segment.get("words", []))
    return {
        "audio_path": str(Path(audio_path)),
        "start": round(start_time, 2),
        "end": round(end_time, 2),
        "text": text,
        "segments": segments,
    }


# ---------------------------------------------------------------------------
# Parakeet NeMo forced CTC alignment backend
# ---------------------------------------------------------------------------


def _load_parakeet_align_model(backend: dict) -> Any:
    """Load Parakeet model for alignment, cached by (model_id, device)."""
    import glob

    import nemo.collections.asr as nemo_asr

    model_id_or_path = backend.get("model_id_or_path") or backend.get("align_model_id")
    if not model_id_or_path:
        raise ValueError("parakeet_nemo_align backend requires `model_id_or_path`")
    device = str(backend.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    key = (str(model_id_or_path), device)
    if key not in _PARAKEET_ALIGN_MODELS:
        model_path = Path(model_id_or_path)
        if model_path.exists() and model_path.suffix == ".nemo":
            model = nemo_asr.models.ASRModel.restore_from(str(model_path), map_location=device)
        else:
            try:
                model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=str(model_id_or_path), map_location=device
                )
            except (FileNotFoundError, OSError):
                cache_pattern = str(Path.home() / ".cache/torch/NeMo/**/*.nemo")
                nemo_files = glob.glob(cache_pattern, recursive=True)
                matching = [f for f in nemo_files if "parakeet-ctc" in f and "vi" in f.lower()]
                if not matching:
                    raise FileNotFoundError(
                        f"Could not find downloaded .nemo file for {model_id_or_path}."
                    )
                model = nemo_asr.models.ASRModel.restore_from(matching[0], map_location=device)
        model = model.to(device)
        model.eval()
        _PARAKEET_ALIGN_MODELS[key] = model
    return _PARAKEET_ALIGN_MODELS[key]


def _run_parakeet_nemo_alignment(audio_path: str, text: str, backend: dict) -> dict:
    """Run forced CTC alignment using Parakeet NeMo model.

    Extracts CTC log-probabilities from the encoder, tokenizes the known text,
    and runs Viterbi forced alignment to produce word-level timestamps.
    """
    model = _load_parakeet_align_model(backend)
    sample_rate = int(backend.get("sample_rate", 16000))

    # Load and preprocess audio
    audio_np, sr = librosa.load(audio_path, sr=sample_rate)
    start_time, end_time = _find_start_end(audio_path)

    # Get CTC log-probabilities from the model
    log_probs, encoded_len = _get_ctc_log_probs(model, audio_np, sample_rate)

    # Tokenize the known text
    words = text.split()
    token_ids_per_word = _tokenize_words(model, words)

    # Flatten token sequence for forced alignment
    flat_token_ids = []
    word_boundaries = []  # (start_idx, end_idx) in flat_token_ids for each word
    for token_ids in token_ids_per_word:
        start_idx = len(flat_token_ids)
        flat_token_ids.extend(token_ids)
        word_boundaries.append((start_idx, len(flat_token_ids)))

    # Run Viterbi forced alignment
    frame_alignment = _viterbi_forced_align(log_probs, flat_token_ids, encoded_len)

    # Convert frame indices to word-level timestamps
    total_frames = encoded_len
    frame_duration = (end_time - start_time) / max(total_frames, 1)

    segments = []
    for word, (tok_start, tok_end) in zip(words, word_boundaries):
        # Find frame range for this word's tokens
        word_frames = [f for f, t in enumerate(frame_alignment) if tok_start <= t < tok_end]
        if word_frames:
            w_start = start_time + min(word_frames) * frame_duration
            w_end = start_time + (max(word_frames) + 1) * frame_duration
        else:
            # Fallback: evenly distribute
            word_idx = words.index(word)
            w_start = start_time + word_idx * (end_time - start_time) / len(words)
            w_end = start_time + (word_idx + 1) * (end_time - start_time) / len(words)
        segments.append(
            {
                "word": word,
                "start": round(w_start, 3),
                "end": round(w_end, 3),
            }
        )

    return {
        "audio_path": str(Path(audio_path)),
        "start": round(start_time, 2),
        "end": round(end_time, 2),
        "text": text,
        "segments": segments,
    }


def _get_ctc_log_probs(
    model: Any, audio_np: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, int]:
    """Extract CTC log-probabilities from the Parakeet model encoder."""
    device = next(model.parameters()).device
    audio_tensor = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0).to(device)
    audio_len = torch.tensor([len(audio_np)], dtype=torch.long).to(device)

    with torch.no_grad():
        # NeMo CTC models require preprocessing (mel spectrogram) before encoder
        processed_signal, processed_signal_length = model.preprocessor(
            input_signal=audio_tensor, length=audio_len
        )
        encoded, encoded_len = model.encoder(
            audio_signal=processed_signal, length=processed_signal_length
        )
        log_probs = model.decoder(encoder_output=encoded)
        log_probs = torch.nn.functional.log_softmax(log_probs, dim=-1)

    return log_probs[0].cpu().numpy(), int(encoded_len[0].item())


def _tokenize_words(model: Any, words: list[str]) -> list[list[int]]:
    """Tokenize each word separately using the model's tokenizer."""
    tokenizer = model.tokenizer
    token_ids_per_word = []
    for word in words:
        ids = tokenizer.text_to_ids(word)
        if not ids:
            # Fallback: try with space prefix (BPE convention)
            ids = tokenizer.text_to_ids(" " + word)
        if not ids:
            ids = [0]  # blank token as fallback
        token_ids_per_word.append(ids)
    return token_ids_per_word


def _viterbi_forced_align(
    log_probs: np.ndarray, token_ids: list[int], num_frames: int
) -> list[int]:
    """Viterbi forced alignment of token sequence against CTC log-probs.

    Returns a list of length num_frames, where each element is the index into
    token_ids that frame is aligned to (or -1 for blank frames).
    """
    num_tokens = len(token_ids)
    if num_tokens == 0:
        return [-1] * num_frames

    blank_id = 0  # CTC blank is typically index 0 in NeMo

    # State space: for each token i, we have states (blank_before_i, token_i)
    # Total states = 2 * num_tokens + 1 (initial blank + pairs)
    # Simplified: use standard CTC forced alignment with interleaved blanks
    # States: b0, t0, b1, t1, b2, t2, ... where bi=blank, ti=token_i
    num_states = 2 * num_tokens + 1

    # Viterbi DP
    NEG_INF = -1e30
    viterbi = np.full((num_frames, num_states), NEG_INF, dtype=np.float64)
    backptr = np.zeros((num_frames, num_states), dtype=np.int32)

    # Initialize first frame
    viterbi[0, 0] = log_probs[0, blank_id]  # start with blank
    if num_tokens > 0:
        viterbi[0, 1] = log_probs[0, token_ids[0]]  # or start with first token

    # Fill DP
    for t in range(1, num_frames):
        for s in range(num_states):
            if s % 2 == 0:
                # Blank state (index s // 2)
                emit_prob = log_probs[t, blank_id]
            else:
                # Token state (index s // 2)
                tok_idx = s // 2
                emit_prob = log_probs[t, token_ids[tok_idx]]

            # Possible previous states
            candidates = []
            # Stay in same state
            candidates.append((viterbi[t - 1, s], s))
            # From previous state (left-to-right)
            if s > 0:
                candidates.append((viterbi[t - 1, s - 1], s - 1))
            # Skip blank (token to next token if different)
            if s > 1 and s % 2 == 1:
                tok_idx = s // 2
                prev_tok_idx = (s - 2) // 2
                if tok_idx > 0 and token_ids[tok_idx] != token_ids[prev_tok_idx]:
                    candidates.append((viterbi[t - 1, s - 2], s - 2))

            best_val, best_state = max(candidates, key=lambda x: x[0])
            viterbi[t, s] = best_val + emit_prob
            backptr[t, s] = best_state

    # Backtrace from best final state (must end at last token or trailing blank)
    final_states = [num_states - 1, num_states - 2]  # trailing blank or last token
    best_final = max(final_states, key=lambda s: viterbi[num_frames - 1, s])

    # Trace back
    state_path = [0] * num_frames
    state_path[num_frames - 1] = best_final
    for t in range(num_frames - 2, -1, -1):
        state_path[t] = backptr[t + 1, state_path[t + 1]]

    # Convert state path to token indices (-1 for blank frames)
    frame_to_token = []
    for s in state_path:
        if s % 2 == 0:
            # Blank state - assign to nearest token
            tok_idx = s // 2
            if tok_idx >= num_tokens:
                tok_idx = num_tokens - 1
            frame_to_token.append(tok_idx)
        else:
            frame_to_token.append(s // 2)

    return frame_to_token
