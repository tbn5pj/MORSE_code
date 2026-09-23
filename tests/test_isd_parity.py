from morse.compression.isd import compress_isd_candidate
from conftest import DeterministicNLL, TinyTokenizer, tiny_example


def test_isd_frozen_structural_fixture():
    result = compress_isd_candidate(tiny_example(), ["c2", "c0", "c1"], 5,
                                    TinyTokenizer(), DeterministicNLL())
    assert result["used_tokens"] <= 5
    assert [row["deleted_uid"] for row in result["trajectory"]["deletions"]] == ["c2::s0", "c1::s1", "c0::s1", "c0::s0"]
    assert result["retained_unit_ids"] == ["c1::s0", "c2::s1"]
