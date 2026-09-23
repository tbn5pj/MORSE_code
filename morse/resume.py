"""Provenance-checked atomic candidate result store."""
from __future__ import annotations

import json
from pathlib import Path

from .utils import atomic_write_json, sha256_json


class CandidateStore:
    def __init__(self, root):
        self.root = Path(root)

    def path(self, identity):
        return self.root / f"{sha256_json(identity)}.json"

    def load(self, identity):
        path = self.path(identity)
        if not path.exists():
            return None
        value = json.loads(path.read_text())
        if value.get("identity") != identity:
            raise RuntimeError("candidate cache provenance mismatch")
        return value["result"]

    def save(self, identity, result):
        atomic_write_json(self.path(identity), {"identity": identity, "result": result})

