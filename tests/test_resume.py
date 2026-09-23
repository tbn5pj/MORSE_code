from morse.resume import CandidateStore


def test_resume_requires_exact_identity(tmp_path):
    store = CandidateStore(tmp_path)
    identity = {"example": "x", "seed": 42, "candidate": "R1"}
    store.save(identity, {"retained": ["u1"]})
    assert store.load(identity) == {"retained": ["u1"]}
    assert store.load({**identity, "seed": 43}) is None

