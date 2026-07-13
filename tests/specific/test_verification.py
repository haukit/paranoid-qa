"""Deterministic parts of specific verification: quote location and the relevance fail-open.

The support/relevance critic LLM calls are not exercised here (they need a live model); the
fabricated path and the relevance fail-open both resolve without an LLM call.
"""

import asyncio

from paranoid_qa.contracts.specific import RetrievedChunk, SpecificClaim
from paranoid_qa.specific.verification import _is_relevant, locate_quote, verify_specific_claim

CHUNK: RetrievedChunk = {
    "text": "The probable cause was the flight crew's failure to maintain airspeed.",
    "document": "a.pdf",
    "page": "1",
}


def test_exact_quote_match():
    assert locate_quote("probable cause was the flight crew's failure", [CHUNK]) is CHUNK


def test_case_and_whitespace_normalization():
    assert locate_quote("PROBABLE   CAUSE   was", [CHUNK]) is CHUNK


def test_ellipsis_segment_matching():
    assert locate_quote("probable cause ... maintain airspeed", [CHUNK]) is CHUNK


def test_fabricated_quote_not_located():
    assert locate_quote("engine fire shortly after takeoff", [CHUNK]) is None


def test_empty_quote_is_not_located():
    assert locate_quote("", [CHUNK]) is None


def test_verify_returns_fabricated_when_quote_missing():
    claim = SpecificClaim(text="an engine fire occurred", quote="engine fire shortly after takeoff")
    verdict = asyncio.run(verify_specific_claim(claim, [CHUNK], "what happened?"))
    assert verdict.verdict == "fabricated"


def test_relevance_fails_open_without_document_identity():
    # An unknown document has no identity to gate on, so relevance must not reject it.
    verdict = asyncio.run(_is_relevant("any question", "document-not-in-store.pdf"))
    assert verdict.relevant is True
