from morse.candidates import generate_morse_candidates


def test_scheduling_does_not_change_candidate_identity():
    ids = ["a", "b", "c", "d"]
    candidates = generate_morse_candidates(ids, ids, "parallel", 42, 5)
    serial = [row.permutation_hash for row in candidates]
    scheduled = [row.permutation_hash for row in sorted(reversed(candidates), key=lambda row: candidates.index(row))]
    assert scheduled == serial

