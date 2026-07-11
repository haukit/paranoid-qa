"""FastAPI server

Run: uv run uvicorn paranoid_qa.server:app --reload"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import contextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from pydantic import BaseModel, Field

from paranoid_qa import demo
from paranoid_qa.config import settings
from paranoid_qa.graph import build_graph
from paranoid_qa.tracing import setup_tracing

if os.getenv("PHOENIX_COLLECTOR_ENDPOINT"):
    setup_tracing()

_tracer = trace.get_tracer("paranoid-qa")
app = FastAPI(title="paranoid-qa")
graph = build_graph()


@contextmanager
def _ask_span(question: str):
    """Open the span for one QA run, with the question as its input attribute.

    Yields the span so the caller can attach output attributes before it closes."""
    with _tracer.start_as_current_span("ask") as span:
        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value
        )
        span.set_attribute(SpanAttributes.INPUT_VALUE, question)
        span.set_attribute("question", question)

        yield span


def _record_output(span, payload: dict) -> None:
    """Record the final payload and its faithfulness as output attributes on the span."""
    span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(payload))
    span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json")
    span.set_attribute("faithful", bool(payload.get("faithful", False)))


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=settings.max_query_chars)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    if settings.provider == "stub":
        return {"status": "ready", "mode": "stub"}

    missing = []
    if not Path(settings.storage).exists():
        missing.append(str(settings.storage))
    if not Path(settings.lightrag_dir).exists():
        missing.append(str(settings.lightrag_dir))

    if missing:
        raise HTTPException(
            status_code=503,
            detail={"message": "Index artifacts are missing", "missing": missing},
        )

    return {
        "status": "ready",
        "storage": str(settings.storage),
        "lightrag_dir": str(settings.lightrag_dir),
        "provider": settings.provider,
        "embed_provider": settings.embed_provider,
    }


@app.get("/version")
def version():
    return {
        "app": "paranoid-qa",
        "version": "0.1.0",
        "provider": settings.provider,
        "gen_model": settings.gen_model,
        "critic_model": settings.critic_model,
        "embed_model": settings.embed_model,
        "demo_access_required": settings.demo_require_access,
    }


class DemoSessionRequest(BaseModel):
    token: str


@app.post("/demo/session")
def create_demo_session(req: DemoSessionRequest) -> dict:
    if settings.demo_disabled:
        raise HTTPException(503, "Demo is disabled")
    if not settings.demo_invite_code or req.token != settings.demo_invite_code:
        raise HTTPException(401, "Invalid invite token")
    return {
        "session": demo.start_session(),
        "expires_in_days": settings.demo_session_days,
        "questions": settings.demo_questions_per_session,
    }


@app.get("/demo/session")
def demo_session_status(x_demo_session: str | None = Header(default=None)) -> dict:
    if not x_demo_session:
        raise HTTPException(401, "Demo session required")
    sid = demo.read_session(x_demo_session)
    if sid is None:
        raise HTTPException(401, "Invalid or expired session")
    left = demo.remaining(sid)
    if left is None:
        raise HTTPException(401, "Session expired; start a new session")
    return {"remaining": left}


def require_demo_session(x_demo_session: str | None = Header(default=None)) -> str | None:
    if not settings.demo_require_access:
        return None
    if settings.demo_disabled:
        raise HTTPException(503, "Demo is disabled")
    if not x_demo_session:
        raise HTTPException(401, "Demo session required")
    sid = demo.read_session(x_demo_session)
    if sid is None:
        raise HTTPException(401, "Invalid or expired session")

    try:
        demo.charge(sid)
    except demo.DemoDenied as e:
        raise HTTPException(e.status_code, e.detail) from e
    return sid


def _sse(event: str, data: dict) -> str:
    """Format one SSE frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _progress(node: str, update: dict) -> dict:
    """Return a JSON-safe outcome for a node's state update."""
    data: dict = {"node": node}
    for key in ("route", "grade", "faithful", "attempts"):
        if key in update:
            data[key] = update[key]
    if "chunks" in update:
        data["chunks"] = len(update["chunks"])
    if "verdicts" in update:
        data["verdicts"] = [v.verdict for v in update["verdicts"]]
    if update.get("answer") is not None:
        data["claims"] = len(update["answer"].claims)
    return data


def _build_payload(state: dict) -> dict:
    """Collapse the final graph state into the JSON response."""
    answer = state.get("answer")
    verdicts = state.get("verdicts") or []
    claims = [
        {
            "text": c.text,
            "quote": c.quote,
            "citation": str(v.source) if v.source is not None else None,
        }
        for c, v in zip(answer.claims if answer else [], verdicts)
    ]

    return {
        "answer": answer.text if answer else "",
        "claims": claims,
        "faithful": state.get("faithful", False),
    }


def _stub_payload() -> dict:
    """Deterministic fake answer for stub-mode deployment; no graph or model run."""
    return {
        "answer": "This is a stub answer used for deployment testing.",
        "claims": [],
        "faithful": True,
    }


async def _run(question: str) -> AsyncIterator[str]:
    """Stream one QA run as SSE progress frames followed by a final answer frame."""
    if settings.provider == "stub":
        yield _sse("answer", _stub_payload())
        return

    state = {}
    with _ask_span(question) as span:
        async for chunk in graph.astream({"question": question}, stream_mode="updates"):
            for node, update in chunk.items():
                yield _sse("progress", _progress(node, update))
                state.update(update)

        payload = _build_payload(state)
        _record_output(span, payload)

        yield _sse("answer", payload)


async def _run_to_payload(question: str) -> dict:
    """Run one QA run to completion and return the final JSON payload."""
    if settings.provider == "stub":
        return _stub_payload()

    with _ask_span(question) as span:
        state = await graph.ainvoke({"question": question})
        payload = _build_payload(state)
        _record_output(span, payload)
        return payload


@app.post("/ask", dependencies=[Depends(require_demo_session)])
async def ask(req: AskRequest) -> StreamingResponse:
    return StreamingResponse(_run(req.question), media_type="text/event-stream")


@app.post("/ask_json", dependencies=[Depends(require_demo_session)])
async def ask_json(req: AskRequest) -> dict:
    return await _run_to_payload(req.question)
