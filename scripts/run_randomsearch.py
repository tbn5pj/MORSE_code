#!/usr/bin/env python3
"""Run RandomSearch-K with the same deterministic random stream as MORSE."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from morse.candidates import generate_randomsearch_candidates
from morse.compression import compress_one_candidate, compress_isd_candidate
from morse.ordering import reverse_order
from morse.scoring import ScalarNLL, load_compression_model
from morse.utils import normalize_contexts

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--K", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--compressor", choices=("1p", "isd"), default="1p")
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()
source = json.loads(args.input.read_text())
contexts = normalize_contexts(source["contexts"])
example = {"example_id": source.get("example_id", "example"), "question": source["query"],
           "contexts": contexts, "gold_pairs": []}
tokenizer, model = load_compression_model(args.device)
nll = ScalarNLL(tokenizer, model)
reverse = reverse_order(example["question"], contexts, nll)
candidates = generate_randomsearch_candidates([row["id"] for row in contexts], example["example_id"],
                                               args.seed, args.K, reverse_order=reverse)
function = compress_one_candidate if args.compressor == "1p" else compress_isd_candidate
rows = [function(example, candidate.permutation, source["target_budget"], tokenizer, nll) for candidate in candidates]
winner = max(range(len(rows)), key=lambda index: (rows[index]["j_b"], -index))
print(json.dumps({"winner": candidates[winner].to_dict(), "result": rows[winner]}, indent=2))

