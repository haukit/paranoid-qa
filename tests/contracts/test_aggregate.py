from paranoid_qa.contracts.aggregate import AggregateAnswer, AggregateClaim


def test_aggregate_claim_has_no_quote_field():
    assert "quote" not in AggregateClaim.model_fields


def test_aggregate_answer_kind_and_text():
    answer = AggregateAnswer(claims=[AggregateClaim(text="x"), AggregateClaim(text="y")])
    assert answer.kind == "aggregate"
    assert answer.text == "x y"
    assert answer.model_dump()["kind"] == "aggregate"
