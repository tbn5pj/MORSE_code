from morse.ordering import reverse_order, reverse_scores
from conftest import DeterministicNLL, tiny_example


def test_reverse_order_frozen_tie_semantics():
    example, nll = tiny_example(), DeterministicNLL()
    scores = reverse_scores(example["question"], example["contexts"], nll)
    assert reverse_order(example["question"], example["contexts"], nll) == ["c2", "c0", "c1"]
    assert scores == {"c0": -18.941176470588236, "c1": -27.58823529411765,
                      "c2": -2.7058823529411775}
