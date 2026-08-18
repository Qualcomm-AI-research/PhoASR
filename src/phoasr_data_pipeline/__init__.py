# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""PhoASR data pipeline package."""

from .io import detect_transcript_availability
from .pipeline import run_pipeline

__all__ = ["detect_transcript_availability", "run_pipeline"]
