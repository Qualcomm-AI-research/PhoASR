# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""ASR backends for transcription and verification."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch


def transcribe(
    audio_path: str,
    backend: dict,
    sample_id: str | None = None,
    loaded_model: Any | None = None,
) -> str:
    """Transcribe one audio file with the configured ASR backend."""
    backend_type = backend.get("type")
    if backend_type in ("chunkformer", "parakeet_nemo"):
        return transcribe_batch(
            [audio_path], backend, sample_ids=[sample_id], loaded_model=loaded_model
        )[0]
    return transcribe_batch([audio_path], backend, sample_ids=[sample_id])[0]


def transcribe_batch(
    audio_paths: list[str],
    backend: dict,
    sample_ids: list[str | None] | None = None,
    loaded_model: Any | None = None,
) -> list[str]:
    """Transcribe multiple audio files and preserve input ordering."""
    backend_type = backend.get("type")
    if backend_type == "chunkformer":
        model = loaded_model if loaded_model is not None else load_chunkformer_model(backend)
        return chunkformer_batch_transcribe(model, audio_paths, backend)
    if backend_type == "parakeet_nemo":
        model = loaded_model if loaded_model is not None else load_parakeet_model(backend)
        return parakeet_batch_transcribe(model, audio_paths, backend)
    if backend_type == "phowhisper_ctranslate2":
        return _run_phowhisper_ctranslate2_batch(audio_paths, backend)
    if backend_type == "command":
        raise ValueError("command backends are no longer supported in the standalone package")
    raise ValueError(
        "Unsupported ASR backend type: "
        f"{backend_type!r}. Supported ASR backends are `chunkformer`, `parakeet_nemo`, and `phowhisper_ctranslate2`."
    )


def load_chunkformer_model(backend: dict) -> Any:
    """Load the configured ChunkFormer checkpoint once for batch reuse."""
    model_id_or_path = backend.get("model_id_or_path") or backend.get("model_path")
    if not model_id_or_path:
        raise ValueError("chunkformer backend requires `model_id_or_path`")

    from chunkformer import ChunkFormerModel

    return ChunkFormerModel.from_pretrained(str(model_id_or_path))


def chunkformer_batch_transcribe(model: Any, audio_paths: list[str], backend: dict) -> list[str]:
    """Batch transcribe audio paths with a preloaded ChunkFormer model."""
    if not audio_paths:
        return []
    results = model.batch_decode(
        audio_paths=[str(path) for path in audio_paths],
        chunk_size=int(backend.get("chunk_size", 64)),
        left_context_size=int(backend.get("left_context_size", 128)),
        right_context_size=int(backend.get("right_context_size", 128)),
        total_batch_duration=int(backend.get("total_batch_duration", 1800)),
    )
    return [_normalize_asr_output(result) for result in results]


# ---------------------------------------------------------------------------
# Parakeet NeMo backend
# ---------------------------------------------------------------------------

_PARAKEET_MODELS: dict[tuple[str, str], Any] = {}


def load_parakeet_model(backend: dict) -> Any:
    """Load the configured Parakeet NeMo model once for batch reuse."""
    import nemo.collections.asr as nemo_asr

    model_id_or_path = backend.get("model_id_or_path") or backend.get("model_path")
    if not model_id_or_path:
        raise ValueError("parakeet_nemo backend requires `model_id_or_path`")
    device = str(backend.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    key = (str(model_id_or_path), device)
    if key not in _PARAKEET_MODELS:
        model_path = Path(model_id_or_path)
        if model_path.exists() and model_path.suffix == ".nemo":
            model = nemo_asr.models.ASRModel.restore_from(str(model_path), map_location=device)
        else:
            # Try from_pretrained; if it fails due to missing config, find the .nemo file
            try:
                model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=str(model_id_or_path), map_location=device
                )
            except (FileNotFoundError, OSError):
                # Locate the downloaded .nemo file in the cache
                import glob

                cache_pattern = str(Path.home() / ".cache/torch/NeMo/**/*.nemo")
                nemo_files = glob.glob(cache_pattern, recursive=True)
                matching = [f for f in nemo_files if "parakeet-ctc" in f and "vi" in f.lower()]
                if not matching:
                    raise FileNotFoundError(
                        f"Could not find downloaded .nemo file for {model_id_or_path}. "
                        "Try downloading manually and passing the path directly."
                    )
                model = nemo_asr.models.ASRModel.restore_from(matching[0], map_location=device)
        model = model.to(device)
        model.eval()
        _PARAKEET_MODELS[key] = model
    return _PARAKEET_MODELS[key]


def parakeet_batch_transcribe(model: Any, audio_paths: list[str], backend: dict) -> list[str]:
    """Batch transcribe audio paths with a preloaded Parakeet NeMo model."""
    if not audio_paths:
        return []
    batch_size = int(backend.get("batch_size", 8))
    results: list[str] = []
    for start in range(0, len(audio_paths), batch_size):
        batch = [str(p) for p in audio_paths[start : start + batch_size]]
        outputs = model.transcribe(batch)
        # NeMo returns list of strings or Hypothesis objects
        for out in outputs:
            if hasattr(out, "text"):
                results.append(str(out.text).strip())
            else:
                results.append(str(out).strip())
    return results


def _run_phowhisper_ctranslate2_batch(audio_paths: list[str], backend: dict) -> list[str]:
    """Run batched PhoWhisper CTranslate2 inference and collect `.txt` outputs."""
    if not audio_paths:
        return []

    model_dir = _ensure_ct2_model_dir(backend)
    batch_size = int(backend.get("batch_size", 16))
    chunk_size = int(backend.get("chunk_size", 100))
    language = str(backend.get("language", "vi"))
    threads = int(backend.get("threads", 8))
    compute_type = str(backend.get("compute_type", "float16"))
    vad_filter = bool(backend.get("vad_filter", True))
    device = str(backend.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    outputs_by_stem: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="phowhisper_ct2_") as temp_dir:
        output_dir = Path(temp_dir)
        for start in range(0, len(audio_paths), chunk_size):
            chunk_audio_paths = audio_paths[start : start + chunk_size]
            command = [
                "whisper-ctranslate2",
                *[str(Path(path).resolve()) for path in chunk_audio_paths],
                "--model_directory",
                str(model_dir),
                "--output_dir",
                str(output_dir),
                "--batched",
                "True",
                "--batch_size",
                str(batch_size),
                "--language",
                language,
                "--output_format",
                "txt",
                "--threads",
                str(threads),
                "--compute_type",
                compute_type,
                "--device",
                device,
            ]
            if vad_filter:
                command.extend(["--vad_filter", "True"])
            subprocess.run(command, check=True, capture_output=True, text=True)

        ordered_outputs: list[str] = []
        for audio_path in audio_paths:
            stem = Path(audio_path).stem
            if stem not in outputs_by_stem:
                transcript_path = output_dir / f"{stem}.txt"
                if not transcript_path.exists():
                    raise FileNotFoundError(
                        f"PhoWhisper CTranslate2 output not found for {audio_path}: {transcript_path}"
                    )
                outputs_by_stem[stem] = transcript_path.read_text(encoding="utf-8").strip()
            ordered_outputs.append(outputs_by_stem[stem])
    return ordered_outputs


def _ensure_ct2_model_dir(backend: dict) -> Path:
    """Validate and return the configured PhoWhisper CTranslate2 model directory."""
    model_dir = backend.get("model_dir")
    if not model_dir:
        raise ValueError("phowhisper_ctranslate2 backend requires `model_dir`")
    path = Path(model_dir)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(
            f"PhoWhisper CTranslate2 model directory not found: {path}. "
            "Convert the checkpoint first with `ct2-transformers-converter --model vinai/PhoWhisper-large --output_dir checkpoints/phowhisper-large-ctranslate`."
        )
    expected_files = [path / "model.bin", path / "config.json"]
    missing = [candidate.name for candidate in expected_files if not candidate.exists()]
    if missing:
        raise FileNotFoundError(
            f"PhoWhisper CTranslate2 model directory is incomplete: {path}. Missing: {', '.join(missing)}"
        )
    return path


def _normalize_asr_output(result: Any) -> str:
    """Extract plain transcript text from backend-specific result payloads."""
    if isinstance(result, dict):
        if "text" in result:
            return str(result["text"]).strip()
        if "transcription" in result:
            return str(result["transcription"]).strip()
    return str(result).strip()
