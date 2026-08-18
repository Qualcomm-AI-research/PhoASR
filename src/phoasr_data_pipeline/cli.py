# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""CLI entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .constants import STAGES
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser and subcommands."""
    parser = argparse.ArgumentParser(prog="phoasr-data-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the pipeline")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--input", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--start-stage", choices=STAGES)
    run_parser.add_argument("--end-stage", choices=STAGES)

    validate_parser = subparsers.add_parser("validate-input", help="Validate input manifest only")
    validate_parser.add_argument("--config", required=True)
    validate_parser.add_argument("--input", required=True)
    validate_parser.add_argument("--output", required=True)

    subparsers.add_parser("stages", help="List supported stages")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a manifest")
    inspect_parser.add_argument("--manifest", required=True)

    return parser


def main() -> None:
    """Dispatch the requested CLI command."""
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "stages":
        for stage in STAGES:
            print(stage)
        return
    if args.command == "inspect":
        manifest_path = Path(args.manifest)
        print(json.dumps({"manifest": str(manifest_path.resolve())}, indent=2))
        return
    if args.command == "validate-input":
        result = run_pipeline(
            config_path=args.config,
            input_manifest=args.input,
            output_dir=args.output,
            start_stage="validate",
            end_stage="validate",
        )
        print(result)
        return
    if args.command == "run":
        result = run_pipeline(
            config_path=args.config,
            input_manifest=args.input,
            output_dir=args.output,
            start_stage=args.start_stage,
            end_stage=args.end_stage,
        )
        print(result)
        return
    parser.error(f"Unknown command: {args.command}")
