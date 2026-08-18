# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Shared constants for the pipeline."""

STAGES = [
    "validate",
    "transcribe",
    "score_wer",
    "filter",
    "punc_cap",
    "num2word",
    "align",
    "finalize",
]

STAGE_DIRS = {
    "validate": "01_validated",
    "transcribe": "02_transcribed",
    "score_wer": "03_wer_scored",
    "filter": "04_filtered",
    "punc_cap": "05_punc_cap",
    "num2word": "06_num2word",
    "align": "07_aligned",
    "finalize": "08_final",
}

REQUIRED_INPUT_COLUMNS = ["sample_id", "audio_path"]
