"""Exact scalar BF16 likelihood scoring (scientific batch dimension one)."""
from __future__ import annotations

import time

from .constants import COMPRESSION_MODEL, COMPRESSION_REVISION


def load_compression_model(device="cuda:0", local_files_only=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(COMPRESSION_MODEL, revision=COMPRESSION_REVISION,
                                               trust_remote_code=True, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        COMPRESSION_MODEL, revision=COMPRESSION_REVISION, torch_dtype=torch.bfloat16,
        trust_remote_code=True, local_files_only=local_files_only, low_cpu_mem_usage=True)
    model.to(device)
    model.eval()
    model.config.use_cache = False
    if {str(parameter.dtype) for parameter in model.parameters()} != {"torch.bfloat16"}:
        raise RuntimeError("compression model must be entirely BF16")
    return tokenizer, model


def target_nll(prefix, target, tokenizer, model):
    import torch
    prefix_ids = tokenizer(str(prefix), add_special_tokens=False).input_ids
    target_ids = tokenizer(str(target), add_special_tokens=False).input_ids
    if not target_ids:
        return {"avg_nll": 0.0, "total_nll": 0.0, "num_tokens": 0}
    input_ids = torch.tensor([prefix_ids + target_ids], device=next(model.parameters()).device)
    labels = input_ids.clone()
    labels[:, :len(prefix_ids)] = -100
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
    avg = float(outputs.loss.item())
    return {"avg_nll": avg, "total_nll": avg * len(target_ids), "num_tokens": len(target_ids)}


class ScalarNLL:
    def __init__(self, tokenizer, model):
        self.tokenizer, self.model, self.values = tokenizer, model, {}
        self.calls = self.hits = 0
        self.model_seconds = 0.0

    def __call__(self, prefix, target):
        key = (str(prefix), str(target))
        if key in self.values:
            self.hits += 1
            return self.values[key]
        started = time.perf_counter()
        value = target_nll(*key, self.tokenizer, self.model)
        self.model_seconds += time.perf_counter() - started
        self.calls += 1
        self.values[key] = value
        return value

