"""Pure aggregation helpers."""
from __future__ import annotations

import statistics


def mean_and_sample_sd(values):
    values = list(map(float, values))
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0

