#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from morse import MORSE


def main():
    parser = argparse.ArgumentParser(description="Run final MORSE global search")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compressor", choices=("1p", "isd"), default="1p")
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("reference", "fast"), default="fast")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path)
    args = parser.parse_args()
    value = json.loads(args.input.read_text())
    gpus = [int(item) for item in args.gpus.split(",") if item.strip()]
    with MORSE(args.compressor, args.K, args.seed, args.mode, gpus, args.device,
               args.namespace, args.local_files_only, checkpoint_dir=args.checkpoint_dir) as engine:
        result = engine.compress(value["query"], value["contexts"], value["target_budget"],
                                 example_id=value.get("example_id"))
    rendered = json.dumps(dataclasses.asdict(result), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
