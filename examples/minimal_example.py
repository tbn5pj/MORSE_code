import json
from pathlib import Path

from morse import MORSE

example = json.loads((Path(__file__).parent / "example_input.json").read_text())
with MORSE(compressor="1p", K=5, seed=42, mode="reference", device="cuda:0") as engine:
    result = engine.compress(example["query"], example["contexts"], example["target_budget"],
                             example_id=example["example_id"])
print(result.compressed_text)

