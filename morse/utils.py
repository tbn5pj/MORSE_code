"""Stable hashing and restart-safe file operations."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def normalize_contexts(contexts):
    """Validate and normalize dataset-independent context dictionaries."""
    result = []
    for index, source in enumerate(contexts):
        row = dict(source)
        row["id"] = str(row.get("id", f"c{index}"))
        row["title"] = str(row.get("title", ""))
        sentences = row.get("sentences")
        if sentences is None:
            sentences = [str(row.get("text", "")).strip()]
        row["sentences"] = [str(value).strip() for value in sentences if str(value).strip()]
        row["text"] = str(row.get("text") or f"Title: {row['title']}\n" + " ".join(row["sentences"]))
        row["doc_order"] = int(row.get("doc_order", index))
        result.append(row)
    ids = [row["id"] for row in result]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("contexts require at least one unique ID")
    return result
