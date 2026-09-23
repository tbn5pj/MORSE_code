from morse.candidates import generate_morse_candidates, generate_randomsearch_candidates


def test_deterministic_unique_shared_stream():
    ids, reverse = ["c0", "c1", "c2", "c3"], ["c2", "c0", "c3", "c1"]
    first = generate_morse_candidates(ids, reverse, "example", 42, 5, "panel")
    second = generate_morse_candidates(ids, reverse, "example", 42, 5, "panel")
    random = generate_randomsearch_candidates(ids, "example", 42, 5, "panel", reverse)
    assert first == second
    assert first[0].candidate_id == "Reverse"
    assert len({row.permutation for row in first}) == 5
    assert reverse not in [list(row.permutation) for row in first[1:]]
    assert [row.permutation for row in first[1:]] == [row.permutation for row in random[:4]]

