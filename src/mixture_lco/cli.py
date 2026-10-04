"""Command-line boundary with explicit inputs and no network operations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from mixture_lco.config import load_settings
from mixture_lco.data import load_dataset
from mixture_lco.features import build_features
from mixture_lco.io import json_bytes
from mixture_lco.runner import run
from mixture_lco.synthetic import generate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mixture-lco")
    commands = parser.add_subparsers(dest="command", required=True)
    synthetic = commands.add_parser("synthetic", help="create invented, nonscientific CSV fixtures")
    synthetic.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate", help="validate the three exact-schema CSV files")
    validate.add_argument("--data", type=Path, required=True)
    validate.add_argument("--embeddings", type=Path)
    show_config = commands.add_parser("config", help="print a complete editable configuration")
    show_config.add_argument("--preset", choices=("paper", "smoke"), default="paper")
    for name in ("cv", "lco"):
        command = commands.add_parser(name)
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--embeddings", type=Path)
        config = command.add_mutually_exclusive_group()
        config.add_argument("--config", type=Path)
        config.add_argument("--preset", choices=("paper", "smoke"), default="paper")
        if name == "lco":
            command.add_argument("--ingredient", action="append", required=True,
                                 help="explicit registry ID; repeat for multiple ingredients")
    args = parser.parse_args(argv)
    try:
        if args.command == "synthetic":
            generate(args.output)
            print("Created invented software fixtures; no scientific or thermochromic validity.")
        elif args.command == "config":
            print(json_bytes(load_settings(preset=args.preset)[0].payload).decode(), end="")
        elif args.command == "validate":
            dataset = load_dataset(args.data)
            _, representation = build_features(dataset, args.embeddings)
            print(json_bytes({**dataset.validation_report(), "representation": representation}).decode(),
                  end="")
        else:
            manifest = run(args.data, args.output, protocol=args.command,
                           ingredients=getattr(args, "ingredient", None), embeddings=args.embeddings,
                           config=args.config, preset=args.preset)
            print(json.dumps({"status": manifest["status"], "run_id": manifest["run_id"],
                              "fit_count": manifest["fit_count"]}, sort_keys=True))
    except (ValueError, OSError, RuntimeError, csv.Error) as exc:
        print(f"mixture-lco: {exc}", file=sys.stderr)
        return 2
    return 0
