from paranoid_qa.workflow.outcomes import abstain, accept


def test_accept_sets_answered_status():
    assert accept({}) == {"status": "answered"}


def test_abstain_sets_abstained_and_clears_answer():
    out = abstain({})
    assert out["status"] == "abstained"
    assert out["answer"] is None
    assert out["outcome_reason"]
