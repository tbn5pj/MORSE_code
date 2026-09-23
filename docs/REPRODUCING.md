# Running MORSE and interpreting the release

This guide describes the **released implementation** of MORSE, not an automated reproduction of all manuscript experiments. For the scientific definitions and dataset-level settings, use the accompanying paper.

## 1. Environment

For the tested algorithmic path, use Python 3.10/3.11, an NVIDIA BF16-capable GPU, an appropriately matched CUDA-enabled PyTorch installation and a Transformers version meeting `pyproject.toml`'s bounds. Install the matching PyTorch build for your local CUDA driver before running `pip install -e '.[test]'` if needed.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q tests
```

The tests validate structural logic on CPU using deterministic **synthetic** NLL responses, not inference from Qwen. A separate no-download check is available with `python examples/cpu_smoke_test.py`.

`configs/models.yaml` pins the compression scorer to `Qwen/Qwen2.5-0.5B-Instruct` at revision `7ae557604adf67be50417f59c2c2f167def9a775`. Loading it requires sufficient GPU memory, internet access for the first download and permission to execute third-party remote model code. The model is loaded in BF16 and the likelihood calls use batch dimension one. For ISD, KV caching remains disabled. The separately listed 7B checkpoint is a paper downstream-evaluation model; it is **not** automatically loaded by the core MORSE CLI.

## 2. Input format

Pass one JSON object to `scripts/run_morse.py`:

```json
{
  "example_id": "example-001",
  "query": "Which city is home to the Eiffel Tower?",
  "target_budget": 28,
  "contexts": [
    {"id": "c0", "title": "Eiffel Tower", "sentences": ["The Eiffel Tower is in Paris.", "It opened in 1889."]},
    {"id": "c1", "title": "Berlin", "sentences": ["Berlin is the capital of Germany."]}
  ]
}
```

The `target_budget` is the **retained sentence-token budget** counted using the compression model's tokenizer. Each context must have a unique ID; unspecified IDs are assigned deterministically by position. Sentence segmentation and any paper-specific context truncation must happen upstream: the generic `normalize_contexts` method does **not** reproduce the paper's dataset preprocessing for you. An already-prepared example is in `examples/example_input.json`.

## 3. One-pass MORSE (1P)

```bash
python scripts/run_morse.py \
  --input examples/example_input.json \
  --output outputs/one_pass_result.json \
  --compressor 1p --K 5 --seed 42 \
  --mode reference --device cuda:0
```

The Reverse anchor orders complete contexts by their standalone reverse query-evidence scores. The candidate generator adds four unique deterministic random global permutations. For each candidate, the one-pass compressor computes order-dependent sentence scores and selects full sentences under the same target budget; the retained set is reconstructed in canonical input order and scored by `J_B`. The best scoring candidate is returned. `K=1` runs the Reverse anchor only.

## 4. Iterative sentence deletion (ISD)

```bash
python scripts/run_morse.py \
  --input examples/example_input.json \
  --output outputs/isd_result.json \
  --compressor isd --K 5 --seed 42 \
  --mode reference --device cuda:0
```

ISD repeatedly removes the current lowest-scoring *complete sentence*, updates the affected suffix after each deletion, and stops when within the token budget. As in the source release, candidate scoring remains scalar BF16 with `use_cache=False`. The retained units undergo the paper's postcompression Reverse presentation policy; output order should not be conflated with the winning **compression candidate order**.

## 5. Optional multi-GPU execution

```bash
python scripts/run_morse.py \
  --input examples/example_input.json \
  --output outputs/isd_fast_result.json \
  --compressor isd --K 5 --seed 42 \
  --mode fast --gpus 0,1,2,3,4 \
  --checkpoint-dir checkpoints/example
```

Each candidate is a complete independent trajectory. Multi-GPU speedups depend on queue load, hardware, number of available candidates and model-download readiness; this release makes no new throughput guarantees. Physical GPU indices must be set explicitly, and each worker loads its own model. Checkpoint identities include example, seed, permutation, compressor and budget, and invalid cached candidates are not reused.

## 6. Compare with Reverse and RandomSearch-K

```bash
python scripts/run_reverse.py --input examples/example_input.json --compressor 1p --device cuda:0
python scripts/run_randomsearch.py --input examples/example_input.json --compressor 1p --K 5 --device cuda:0
```

The RandomSearch-K generator uses the same deterministic random stream as MORSE. At K=5 the four random candidates included in MORSE also appear in the matched random-search candidate pool; RandomSearch has a fifth random candidate in place of the Reverse anchor. Candidate selection uses the same `J_B` objective and compressor, so do not compare an average over K random orders with a searched maximum and call it a matched baseline.

## 7. Outputs and evaluation

`MORSEOutput` records the selected candidate ID, the winning **compression order**, canonical retained-unit identities, compressed text, `J_B`, candidate-level results and deterministic provenance. For 1P, the compressed text is reconstructed in canonical input order; for ISD it uses postcompression Reverse presentation. The paper's primary evidence-retention metric is supporting-fact identity recall; helpers are available in `evaluation/sf_recall.py`, with normalized QA F1/EM in `evaluation/qa.py`.

The release intentionally does not redistribute the third-party benchmark data or fixed evaluation manifests. To run a benchmark, prepare its examples and gold supporting-fact identities under the exact protocol defined in the manuscript before invoking this implementation. Preprocessing, cohort identity, downstream generation and aggregation are needed to recreate complete manuscript tables; these are outside this lightweight method archive.

## 8. Scope of validation

`python -m pytest -q tests` checks the included deterministic structural tests. `SCIENTIFIC_PARITY_REPORT.md` and `results/scientific_parity_summary.json` are reports from earlier, model-backed release audits; they are **not** outputs recreated by running the included CPU suite. Paper-specific efficiency reproduction scripts, hardware timing logs, raw model outputs and large caches were excluded from the strict method-only source used to build this public package.
