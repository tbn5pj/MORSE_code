"""No-download structural demonstration of the actual MORSE search pipeline.

The deliberately synthetic tokenizer/NLL below tests wiring, not model accuracy.
Run real scientific inference with scripts/run_morse.py on a BF16 GPU.
"""

import json
from pathlib import Path

from morse.presentation import reconstruct_canonical
from morse.search import run_search
from morse.utils import normalize_contexts


class ToyTokenizer:
    def __call__(self, text, add_special_tokens=False):
        class Encoding:
            input_ids = [sum(map(ord, word)) % 97 + 1 for word in str(text).split()]

        return Encoding()


class ToyNLL:
    def __call__(self, prefix, target):
        # Stable stand-in only; this is NOT a probabilistic language model.
        tokens = str(target).split()
        total = sum(ord(char) for char in str(prefix) + "|" + str(target)) % 1009 / 17.0
        return {"total_nll": total, "avg_nll": total / max(1, len(tokens)),
                "num_tokens": len(tokens)}


def main():
    path = Path(__file__).parent / "example_input.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    example = {"example_id": source["example_id"], "question": source["query"],
               "contexts": normalize_contexts(source["contexts"]), "gold_pairs": []}
    candidates, results, winner = run_search(
        example, source["target_budget"], ToyTokenizer(), ToyNLL(),
        compressor="1p", K=5, seed=42, namespace="cpu-smoke")
    chosen = results[winner]
    compressed, _ = reconstruct_canonical(example, chosen["retained_units"])
    print("Synthetic structural demonstration (not a model-backed research result)")
    print(f"Candidate IDs: {[row.candidate_id for row in candidates]}")
    print(f"Winner: {candidates[winner].candidate_id}; J_B: {chosen['j_b']:.4f}")
    print("Compressed text:")
    print(compressed)


if __name__ == "__main__":
    main()
