"""Readable run identity and candidate-cache validation."""
from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass

from .constants import COMPRESSION_MODEL, COMPRESSION_REVISION, PROTOCOL_VERSION
from .utils import sha256_json


@dataclass(frozen=True)
class Provenance:
    protocol_version: str
    model: str
    revision: str
    precision: str
    use_cache: bool
    batch_dimension: int
    compressor: str
    method: str
    K: int
    seed: int
    example_hash: str
    config_hash: str

    @classmethod
    def create(cls, *, compressor, method, K, seed, example, config):
        return cls(PROTOCOL_VERSION, COMPRESSION_MODEL, COMPRESSION_REVISION, "bfloat16", False, 1,
                   compressor, method, K, seed, sha256_json(example), sha256_json(config))

    def metadata(self):
        return {**asdict(self), "identity_hash": sha256_json(asdict(self)),
                "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat()}

