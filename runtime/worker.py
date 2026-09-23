"""One persistent, model-resident candidate worker."""
from __future__ import annotations

import os
import time


def worker_main(gpu, commands, results, local_files_only=True):
    from morse.compression import compress_isd_candidate, compress_one_candidate
    from morse.compression.one_pass import compress_one_candidate_many
    from morse.ordering import reverse_order
    from morse.presentation import reconstruct_postcompression_reverse
    from morse.scoring import ScalarNLL, load_compression_model
    tokenizer, model = load_compression_model(f"cuda:{gpu}", local_files_only)
    nll = ScalarNLL(tokenizer, model)
    results.put(("ready", gpu, {"pid": os.getpid()}))
    current_search_id = None
    while True:
        task = commands.get()
        if task is None:
            return
        task_id, operation, payload = task
        try:
            search_id = payload.get("search_id")
            if search_id is not None and search_id != current_search_id:
                nll.values.clear()
                current_search_id = search_id
            started = time.perf_counter()
            if operation == "presentation":
                value = reconstruct_postcompression_reverse(payload["example"], payload["selected"], nll)
            elif operation == "reverse":
                value = reverse_order(payload["example"]["question"], payload["example"]["contexts"], nll)
            elif operation == "anchor":
                order = reverse_order(payload["example"]["question"], payload["example"]["contexts"], nll)
                function = compress_one_candidate if payload["compressor"] == "1p" else compress_isd_candidate
                candidate = (compress_one_candidate_many(payload["example"], order, payload["budget"], tokenizer, nll)
                             if payload["compressor"] == "1p" and isinstance(payload["budget"], dict)
                             else function(payload["example"], order, payload["budget"], tokenizer, nll))
                value = {"order": order, "candidate": candidate}
            else:
                function = compress_one_candidate if payload["compressor"] == "1p" else compress_isd_candidate
                value = (compress_one_candidate_many(payload["example"], payload["order"], payload["budget"], tokenizer, nll)
                         if payload["compressor"] == "1p" and isinstance(payload["budget"], dict)
                         else function(payload["example"], payload["order"], payload["budget"], tokenizer, nll))
            results.put(("result", gpu, {"task_id": task_id, "value": value,
                                          "seconds": time.perf_counter() - started}))
        except Exception as error:
            results.put(("error", gpu, {"task_id": task_id, "error": repr(error)}))
