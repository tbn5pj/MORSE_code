# Release scope and provenance

## Source basis

This repository is based on the finalized **MORSE method-only source package** prepared for anonymous submission. Core `morse/`, `runtime/`, `evaluation/` and scientific CLI implementations have not been modified for this public-facing packaging. Documentation, an illustrative figure cropped from the manuscript, a no-model CPU example and structural CI have been added.

## Included

- MORSE global search: Reverse anchor + K-1 distinct deterministic random permutations; maximum `J_B` selection after actual compression.
- Both approved complete-sentence compressors: 1P and no-cache ISD.
- Reference single-device and candidate-parallel multi-GPU paths; optional atomic candidate checkpoints.
- Pinned public compression checkpoint metadata, generic JSON CLI, metric helpers and CPU structural tests.
- Historical scientific parity report and compact historical parity summary, labeled as earlier validation evidence.

## Not included

- Downloaded model weights, copyrighted or licensed dataset contents, private research-directory paths, raw model outputs or hardware-specific research caches.
- Full paper-specific cohort manifests, dataset preprocessing scripts and downstream QA experiment runners.
- Dedicated paper-efficiency reproduction/timing artifacts, which were intentionally excluded from this method-only package.
- Any newly re-run GPU experiment: the packaging pass checks the local structural suite only, unless explicitly documented otherwise.

For the full experimental protocol consult the paper. An arXiv URL and finalized author citation should be inserted after public release rather than guessing metadata.
