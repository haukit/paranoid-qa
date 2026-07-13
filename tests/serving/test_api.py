"""Serving-layer payload building and stable public SSE stage names (no LLM calls)."""

from paranoid_qa.contracts.common import SourceRef
from paranoid_qa.contracts.specific import SpecificAnswer, SpecificClaim, SpecificClaimVerdict
from paranoid_qa.serving.api import PUBLIC_STAGE_NAMES, _build_payload, _progress


def test_public_stage_names_stable():
    assert PUBLIC_STAGE_NAMES["specific_retrieve"] == "retrieve"
    assert PUBLIC_STAGE_NAMES["specific_grade"] == "grade"
    assert PUBLIC_STAGE_NAMES["specific_generate"] == "generate"
    assert PUBLIC_STAGE_NAMES["specific_verify"] == "verify"
    assert PUBLIC_STAGE_NAMES["aggregate_answer"] == "aggregate"
    assert PUBLIC_STAGE_NAMES["aggregate_verify"] == "verify_aggregate"


def test_progress_maps_internal_node_and_fields():
    chunks = _progress("specific_retrieve", {"specific_chunks": [{"text": "t", "document": "a", "page": "1"}]})
    assert chunks["node"] == "retrieve"
    assert chunks["chunks"] == 1

    verify = _progress(
        "specific_verify",
        {
            "verification_passed": True,
            "specific_verdicts": [SpecificClaimVerdict(verdict="supported", explanation="e")],
        },
    )
    assert verify["node"] == "verify"
    assert verify["faithful"] is True
    assert verify["verdicts"] == ["supported"]


def test_build_payload_answered():
    src = SourceRef(document_id="a.pdf", filename="a.pdf", page="1")
    answer = SpecificAnswer(claims=[SpecificClaim(text="c", quote="q")])
    verdict = SpecificClaimVerdict(verdict="supported", explanation="e", source=src)
    state = {
        "answer": answer,
        "specific_verdicts": [verdict],
        "verification_passed": True,
        "status": "answered",
        "route": "specific",
    }
    payload = _build_payload(state)
    assert payload["status"] == "answered"
    assert payload["faithful"] is True
    assert payload["kind"] == "specific"
    assert payload["claims"][0]["quote"] == "q"
    assert payload["claims"][0]["document"] == "a.pdf"
    assert payload["claims"][0]["citation"] == "a.pdf p.1"


def test_build_payload_abstained_hides_answer():
    payload = _build_payload({"answer": None, "status": "abstained", "route": "specific"})
    assert payload["status"] == "abstained"
    assert payload["answer"] == ""
    assert payload["claims"] == []
