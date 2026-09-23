"""Public MORSE API."""
from __future__ import annotations

from dataclasses import dataclass

from .presentation import reconstruct_canonical, reconstruct_postcompression_reverse
from .provenance import Provenance
from .scoring import ScalarNLL, load_compression_model
from .search import run_search
from .utils import normalize_contexts, sha256_json
from .resume import CandidateStore


@dataclass
class MORSEOutput:
    selected_order: list[str]
    selected_candidate_id: str
    retained_units: list[dict]
    compressed_text: str
    J_B: float
    candidate_results: list[dict]
    metadata: dict


class MORSE:
    def __init__(self, compressor="1p", K=5, seed=42, mode="reference", gpus=None,
                 device="cuda:0", namespace="default", local_files_only=False,
                 tokenizer=None, model=None, checkpoint_dir=None):
        if compressor not in {"1p", "isd"} or mode not in {"reference", "fast"}:
            raise ValueError("compressor must be 1p/isd and mode must be reference/fast")
        if K < 1:
            raise ValueError("K must be at least one")
        self.compressor, self.K, self.seed, self.mode = compressor, int(K), int(seed), mode
        self.gpus, self.device, self.namespace = gpus, device, namespace
        self.store = CandidateStore(checkpoint_dir) if checkpoint_dir else None
        self.runtime = None
        if mode == "fast":
            if not gpus:
                raise ValueError("fast mode requires an explicit GPU list")
            from runtime import PersistentCandidateQueue
            self.runtime = PersistentCandidateQueue(gpus, local_files_only)
            self.tokenizer = self.model = self.nll = None
        else:
            if tokenizer is None or model is None:
                tokenizer, model = load_compression_model(device, local_files_only)
            self.tokenizer, self.model = tokenizer, model
            self.nll = ScalarNLL(tokenizer, model)

    def compress(self, query, contexts, target_budget, *, example_id=None, include_candidates=True):
        contexts = normalize_contexts(contexts)
        example = {"example_id": str(example_id or sha256_json({"query": query, "contexts": contexts})),
                   "question": str(query), "contexts": contexts, "gold_pairs": []}
        if self.runtime is None:
            candidates, results, winner = run_search(example, int(target_budget), self.tokenizer, self.nll,
                compressor=self.compressor, K=self.K, seed=self.seed, namespace=self.namespace)
        else:
            if self.store is None:
                candidates, results = self.runtime.search(example, self.K, self.seed, self.namespace,
                                                          int(target_budget), self.compressor)
            else:
                reverse = self.runtime.reverse(example)
                from .candidates import generate_morse_candidates
                candidates = generate_morse_candidates([row["id"] for row in contexts], reverse,
                    example["example_id"], self.seed, self.K, self.namespace)
                identities = [{"example_hash": sha256_json(example), "seed": self.seed,
                    "candidate_id": row.candidate_id, "permutation_hash": row.permutation_hash,
                    "compressor": self.compressor, "budget": int(target_budget)} for row in candidates]
                results = [self.store.load(identity) for identity in identities]
                missing_indices = [index for index, value in enumerate(results) if value is None]
                if missing_indices:
                    computed = self.runtime.candidates(example, [candidates[index] for index in missing_indices],
                                                       int(target_budget), self.compressor)
                    for index, value in zip(missing_indices, computed):
                        results[index] = value
                        self.store.save(identities[index], value)
            winner = max(range(len(results)), key=lambda index: (
                float(results[index]["j_b"]), candidates[index].candidate_id == "Reverse", -index))
        selected = results[winner]
        if self.compressor == "1p":
            text, presentation_order = reconstruct_canonical(example, selected["retained_units"])
        elif self.runtime is None:
            text, presentation_order = reconstruct_postcompression_reverse(
                example, selected["retained_units"], self.nll)
        else:
            text, presentation_order = self.runtime.presentation(example, selected["retained_units"])
        provenance = Provenance.create(compressor=self.compressor, method="morse", K=self.K,
            seed=self.seed, example=example, config={"budget": int(target_budget), "mode": self.mode})
        return MORSEOutput(list(selected["order"]), candidates[winner].candidate_id,
            selected["retained_units"], text, float(selected["j_b"]), results if include_candidates else [],
            {**provenance.metadata(), "effective_K": len(candidates),
             "presentation_order": presentation_order,
             "candidate_permutation_hashes": [row.permutation_hash for row in candidates]})

    def close(self):
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
