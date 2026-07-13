import pytest
from pydantic import ValidationError

from paranoid_qa.contracts.specific import SpecificAnswer, SpecificClaim


def test_specific_claim_requires_quote():
    with pytest.raises(ValidationError):
        SpecificClaim(text="a claim with no quote")  # type: ignore[call-arg]


def test_specific_claim_with_quote_is_valid():
    claim = SpecificClaim(text="the cause was X", quote="the cause was X")
    assert claim.quote == "the cause was X"


def test_specific_answer_kind_and_text():
    answer = SpecificAnswer(
        claims=[SpecificClaim(text="a", quote="qa"), SpecificClaim(text="b", quote="qb")]
    )
    assert answer.kind == "specific"
    assert answer.text == "a b"
    assert answer.model_dump()["kind"] == "specific"
