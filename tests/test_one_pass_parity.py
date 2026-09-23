from morse.compression.one_pass import compress_one_candidate
from conftest import DeterministicNLL, TinyTokenizer, tiny_example


def test_one_pass_frozen_structural_fixture():
    example, tokenizer, nll = tiny_example(), TinyTokenizer(), DeterministicNLL()
    result = compress_one_candidate(example, ["c2", "c0", "c1"], 5, tokenizer, nll)
    assert result["used_tokens"] <= 5
    assert result["retained_unit_ids"] == ["c1::s0", "c2::s1"]
    assert result["j_b"] == -28.705882352941178
