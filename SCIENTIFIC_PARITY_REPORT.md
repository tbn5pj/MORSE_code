# Scientific parity report

## Final definition

The release implements the Reverse anchor plus `K-1` unique global random
permutations. Every ordering is processed by the actual sequential compressor,
retained units are reconstructed according to the paper-matched presentation rule,
and the candidate with maximum `J_B` is returned. The default is `K=5`.

## Reference implementation provenance

The released components were validated against frozen reference implementations.
For auditability, we retain only component labels and SHA-256 fingerprints here;
internal workspace paths are intentionally omitted from the anonymous package.

| Reference component | SHA-256 |
|---|---|
| One-pass scientific reference | `0e2221ccae5e5e2d1b06c373590d51be1256223e534e35eec9ed7dd3bc18be8b` |
| ISD scientific reference | `9f48751944faabc31c4c5a8a7e9f3a55f71a61a81ab8493f68612223eb16ea36` |
| Likelihood scorer reference | `85bfca20c6ecdb836587172c59d765427069c49e1496759e92fcda52e07116cc` |
| One-pass compressor component | `c481bfd6d1471d9a709cf3de0081d0c4937e981dfb799dfdb94afa8b83895b40` |
| ISD compressor component | `89ad0f2ed9443e1c1febf12bdcdcef022573ebba1990fe56f3643b1a2942e347` |
| Likelihood-scoring component | `68321b4d9897e1520db33ca1a53df219f5cebc00205092cc83cde729668adb6b` |
| Fast one-pass reference | `e7b6a49fb19a49dd604fcf121ec6f942d324cdb102d2c21ac90c993e1e6a2375` |
| Fast ISD reference | `cc2001579a984db009cfd0a6568433847df4e92ec83bd9dcec8b0560afe5e57b` |

The scientific functions were separated from experiment-specific manifests,
dataset adapters, and scheduling. Request strings, token accounting, score
subtraction, stable one-pass selection, ISD affected-suffix rescoring,
tie-breaking, retained identities, and `J_B` are unchanged.

## Intentional omissions

The repository omits manuscript artifacts, raw datasets, model weights,
generated answers, production caches, and the rejected prefix-cache prototype.
Only global candidate search is part of the public execution path.

## Modes

Reference mode evaluates candidates sequentially. Fast mode assigns identical
complete candidate computations to persistent GPU workers. Both modes retain
BF16, `use_cache=False`, and scalar batch dimension one.

## Validation

The structural unit suite covers deterministic candidates, stream sharing,
finite-space saturation, one-pass retained identities, complete ISD deletion
trace, deterministic output identity, and provenance-checked resume. The
validated scientific GPU implementation previously passed exact parity on 95
1P searches and the no-cache candidate-parallel ISD path passed exact candidate
trajectory parity. The release generator additionally matched all 500
SAME-100 example/seed streams through K=10. The release runtime matched all 30
timed candidate sets, winners, retained-ID sets, and J_B values. A release ISD
trajectory matched its reference deletion sequence through 16x, retained IDs,
used tokens, and J_B exactly (maximum J_B error 0).

`FINAL_MORSE_METHOD_PARITY = PASS`

`FAST_REFERENCE_PARITY = PASS`

All 27 validated quality cells match. Manuscript-specific efficiency/timing
reproduction artifacts are intentionally omitted from this anonymous method
implementation package.
