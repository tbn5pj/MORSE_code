"""Deterministic global candidate streams for MORSE and RandomSearch-K."""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import asdict, dataclass

from .utils import sha256_json


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    permutation: tuple[str, ...]
    permutation_hash: str
    seed: int

    def to_dict(self):
        value = asdict(self)
        value["permutation"] = list(self.permutation)
        return value


def _stream(context_ids, example_id, seed, count, namespace="default"):
    ids = tuple(str(value) for value in context_ids)
    maximum = math.factorial(len(ids))
    count = min(int(count), maximum)
    key = sha256_json({"protocol": "global_uniform_permutation_stream_v1", "panel": namespace,
                       "example_id": str(example_id), "seed": int(seed)})
    rng = random.Random(int(key, 16))
    if maximum <= 100_000 and count == maximum:
        values = list(itertools.permutations(ids))
        rng.shuffle(values)
        return [tuple(value) for value in values]
    seen, values = set(), []
    while len(values) < count:
        order = list(ids)
        rng.shuffle(order)
        order = tuple(order)
        if order not in seen:
            seen.add(order)
            values.append(order)
    return values


def random_permutation_stream(context_ids, example_id, seed, count, namespace="default"):
    return _stream(context_ids, example_id, seed, count, namespace)


def generate_morse_candidates(context_ids, reverse_order, example_id, seed=42, K=5, namespace="default"):
    if K < 1:
        raise ValueError("K must be at least one")
    ids, reverse = tuple(map(str, context_ids)), tuple(map(str, reverse_order))
    if len(reverse) != len(ids) or set(reverse) != set(ids):
        raise ValueError("Reverse anchor is not a full context permutation")
    target = min(K - 1, math.factorial(len(ids)) - 1)
    raw = _stream(ids, example_id, seed, min(math.factorial(len(ids)), target + 1), namespace)
    randoms = [order for order in raw if order != reverse][:target]
    if len(randoms) < target:
        randoms = [order for order in _stream(ids, example_id, seed, math.factorial(len(ids)), namespace)
                   if order != reverse][:target]
    orders = [reverse] + randoms
    return [Candidate("Reverse" if i == 0 else f"R{i}", order, sha256_json(list(order)), seed)
            for i, order in enumerate(orders)]


def generate_randomsearch_candidates(context_ids, example_id, seed=42, K=5, namespace="default", reverse_order=None):
    if K < 1:
        raise ValueError("K must be at least one")
    if reverse_order is None:
        raise ValueError("reverse_order is required to align the shared R stream with MORSE")
    maximum = math.factorial(len(context_ids))
    reverse = None if reverse_order is None else tuple(map(str, reverse_order))
    target = min(K, maximum)
    raw = _stream(context_ids, example_id, seed, min(maximum, target + (1 if reverse is not None else 0)), namespace)
    orders = [order for order in raw if order != reverse][:target]
    if len(orders) < target:
        complete = _stream(context_ids, example_id, seed, maximum, namespace)
        orders = [order for order in complete if order != reverse][:target]
        if reverse is not None and len(orders) < target:
            orders.append(reverse)
    return [Candidate(f"R{i + 1}", order, sha256_json(list(order)), seed) for i, order in enumerate(orders)]
