from morse.candidates import generate_morse_candidates, generate_randomsearch_candidates


def test_two_context_space_saturates_without_duplicates():
    morse = generate_morse_candidates(["a", "b"], ["a", "b"], "small", K=40)
    random = generate_randomsearch_candidates(["a", "b"], "small", K=40, reverse_order=["a", "b"])
    assert len(morse) == len(random) == 2
    assert len({row.permutation for row in morse}) == len({row.permutation for row in random}) == 2


def test_one_context_space_has_one_candidate():
    assert len(generate_morse_candidates(["a"], ["a"], "one", K=40)) == 1
    assert len(generate_randomsearch_candidates(["a"], "one", K=40, reverse_order=["a"])) == 1
