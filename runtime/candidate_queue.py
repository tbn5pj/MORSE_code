"""Work-conserving persistent process queue for independent candidates."""
from __future__ import annotations

import multiprocessing as mp
import time

from .worker import worker_main


class PersistentCandidateQueue:
    def __init__(self, gpus, local_files_only=True):
        self.gpus = tuple(int(value) for value in gpus)
        if not self.gpus:
            raise ValueError("fast mode needs at least one GPU")
        context = mp.get_context("spawn")
        self.results = context.Queue()
        self.commands, self.processes = {}, {}
        for gpu in self.gpus:
            queue = context.Queue()
            process = context.Process(target=worker_main, args=(gpu, queue, self.results, local_files_only))
            process.start()
            self.commands[gpu], self.processes[gpu] = queue, process
        ready = set()
        while len(ready) < len(self.gpus):
            kind, gpu, payload = self.results.get(timeout=900)
            if kind != "ready":
                raise RuntimeError((kind, gpu, payload))
            ready.add(gpu)

    def _collect(self, prefix, count, timeout=7200):
        values = []
        while len(values) < count:
            kind, gpu, payload = self.results.get(timeout=timeout)
            if kind == "error":
                raise RuntimeError((gpu, payload))
            if not payload["task_id"].startswith(prefix):
                raise RuntimeError("candidate result identity mismatch")
            payload["gpu"] = gpu
            values.append(payload)
        return values

    def reverse(self, example):
        prefix = f"reverse:{example['example_id']}:{time.perf_counter_ns()}"
        self.commands[self.gpus[0]].put((prefix, "reverse", {"example": example}))
        return self._collect(prefix, 1)[0]["value"]

    def presentation(self, example, selected):
        prefix = f"presentation:{example['example_id']}:{time.perf_counter_ns()}"
        self.commands[self.gpus[0]].put((prefix, "presentation", {
            "example": example, "selected": selected, "search_id": prefix}))
        return self._collect(prefix, 1)[0]["value"]

    def candidates(self, example, candidates, budget, compressor):
        prefix = f"candidates:{example['example_id']}:{time.perf_counter_ns()}"
        for index, candidate in enumerate(candidates):
            gpu = self.gpus[index % len(self.gpus)]
            self.commands[gpu].put((f"{prefix}:{index}", "candidate", {
                "example": example, "order": list(candidate.permutation), "budget": int(budget),
                "compressor": compressor}))
        rows = self._collect(prefix, len(candidates))
        rows.sort(key=lambda row: int(row["task_id"].rsplit(":", 1)[1]))
        return [row["value"] for row in rows]

    def search(self, example, K, seed, namespace, budget, compressor):
        """Overlap Reverse construction with independent random candidates."""
        import math
        from morse.candidates import generate_morse_candidates, random_permutation_stream
        ids = [str(row["id"]) for row in example["contexts"]]
        maximum = math.factorial(len(ids))
        target = min(maximum - 1, K - 1)
        raw = random_permutation_stream(ids, example["example_id"], seed, target, namespace)
        prefix = f"search:{example['example_id']}:{time.perf_counter_ns()}"
        self.commands[self.gpus[0]].put((f"{prefix}:anchor", "anchor", {
            "example": example, "budget": budget if isinstance(budget, dict) else int(budget),
            "compressor": compressor, "search_id": prefix}))
        for index, order in enumerate(raw):
            gpu = self.gpus[(index + 1) % len(self.gpus)]
            self.commands[gpu].put((f"{prefix}:raw:{index}", "candidate", {
                "example": example, "order": list(order),
                "budget": budget if isinstance(budget, dict) else int(budget), "compressor": compressor,
                "search_id": prefix}))
        rows = self._collect(prefix, 1 + len(raw))
        anchor = next(row for row in rows if row["task_id"].endswith(":anchor"))["value"]
        random_rows = {int(row["task_id"].rsplit(":", 1)[1]): row["value"] for row in rows
                       if ":raw:" in row["task_id"]}
        reverse = tuple(anchor["order"])
        candidates = generate_morse_candidates(ids, reverse, example["example_id"], seed, K, namespace)
        value_by_order = {tuple(raw[index]): value for index, value in random_rows.items()
                          if tuple(raw[index]) != reverse}
        missing = [candidate for candidate in candidates[1:] if candidate.permutation not in value_by_order]
        repair_prefix = prefix + ":repair"
        for index, candidate in enumerate(missing):
            gpu = self.gpus[index % len(self.gpus)]
            self.commands[gpu].put((f"{repair_prefix}:{index}", "candidate", {
                "example": example, "order": list(candidate.permutation),
                "budget": budget if isinstance(budget, dict) else int(budget),
                "compressor": compressor, "search_id": prefix}))
        for row in self._collect(repair_prefix, len(missing)) if missing else []:
            index = int(row["task_id"].rsplit(":", 1)[1])
            value_by_order[missing[index].permutation] = row["value"]
        values = [anchor["candidate"]] + [value_by_order[candidate.permutation] for candidate in candidates[1:]]
        return candidates, values

    def close(self):
        for queue in self.commands.values():
            queue.put(None)
        for process in self.processes.values():
            process.join(timeout=120)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
