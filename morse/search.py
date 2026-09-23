"""Reference compression-aware candidate selection."""
from __future__ import annotations

from .candidates import generate_morse_candidates, generate_randomsearch_candidates
from .compression import compress_isd_candidate, compress_one_candidate
from .ordering import reverse_order


def run_search(example, budget, tokenizer, nll, *, compressor="1p", K=5, seed=42,
               method="morse", namespace="default", candidate_executor=None):
    ids = [str(row["id"]) for row in example["contexts"]]
    reverse = reverse_order(example["question"], example["contexts"], nll)
    if method == "morse":
        candidates = generate_morse_candidates(ids, reverse, example["example_id"], seed, K, namespace)
    elif method == "randomsearch":
        candidates = generate_randomsearch_candidates(ids, example["example_id"], seed, K, namespace, reverse)
    elif method == "reverse":
        candidates = generate_morse_candidates(ids, reverse, example["example_id"], seed, 1, namespace)
    else:
        raise ValueError("method must be morse, randomsearch, or reverse")
    function = compress_one_candidate if compressor == "1p" else compress_isd_candidate
    if candidate_executor is None:
        results = [function(example, candidate.permutation, budget, tokenizer, nll) for candidate in candidates]
    else:
        results = candidate_executor(function, example, candidates, budget)
    winner_index = max(range(len(results)), key=lambda index: (
        float(results[index]["j_b"]), candidates[index].candidate_id == "Reverse", -index))
    return candidates, results, winner_index
