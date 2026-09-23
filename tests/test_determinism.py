from morse.candidates import generate_morse_candidates


def test_seed_changes_stream_but_repeated_seed_is_stable():
    ids = [f"c{i}" for i in range(6)]
    reverse = list(reversed(ids))
    a = generate_morse_candidates(ids, reverse, "x", 42, 5)
    b = generate_morse_candidates(ids, reverse, "x", 42, 5)
    c = generate_morse_candidates(ids, reverse, "x", 43, 5)
    assert a == b and a != c

