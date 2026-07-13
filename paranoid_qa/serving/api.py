"""FastAPI server.

Run: uv run uvicorn paranoid_qa.serving.api:app --reload

Internal graph nodes use path-prefixed names (specific_retrieve, aggregate_verify, ...). Those are
mapped to stable public stage names here so package-internal renaming never becomes a frontend API.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from contextlib import contextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from pydantic import BaseModel, Field

from paranoid_qa.config import settings
from paranoid_qa.corpus.repository import get_document_text, list_documents
from paranoid_qa.serving import demo
from paranoid_qa.serving.telemetry import TokenCostProcessor
from paranoid_qa.serving.tracing import setup_tracing
from paranoid_qa.workflow.graph import build_graph

tracer_provider = setup_tracing()
_token_cost = TokenCostProcessor()
tracer_provider.add_span_processor(_token_cost)

_tracer = trace.get_tracer("paranoid-qa")
app = FastAPI(title="paranoid-qa")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)
graph = build_graph()

# Internal (path-prefixed) node name -> stable public SSE stage name.
PUBLIC_STAGE_NAMES = {
    "specific_retrieve": "retrieve",
    "specific_grade": "grade",
    "specific_rewrite": "rewrite",
    "specific_generate": "generate",
    "specific_verify": "verify",
    "aggregate_answer": "aggregate",
    "aggregate_verify": "verify_aggregate",
}


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
    """Record the final payload, its faithfulness, and its status as output attributes."""
    span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(payload))
    span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json")
    span.set_attribute("faithful", bool(payload.get("faithful", False)))
    span.set_attribute("status", str(payload.get("status", "")))


def _totals(trace_id: int, latency_ms: int) -> dict:
    """Extract the trace's token/cost totals and pair them with the request latency."""
    usage = _token_cost.pop(trace_id)
    return {
        "tokens_in": usage.tokens_in,
        "tokens_out": usage.tokens_out,
        "cost_usd": round(usage.cost, 6),
        "latency_ms": latency_ms,
    }


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


@app.get("/corpus")
def corpus() -> dict:
    """List the documents backing the demo (empty if no index is present, e.g. stub mode)."""
    try:
        return {"documents": [doc.filename for doc in list_documents()]}
    except FileNotFoundError:
        return {"documents": []}


@app.get("/sources/{name}")
def source(name: str) -> dict:
    """Return a corpus document's extracted text for the view-source panel."""
    try:
        text = get_document_text(name)
    except FileNotFoundError:
        text = None
    if text is None:
        raise HTTPException(404, "Document not found")
    return {"document": name, "text": text}


def _sse(event: str, data: dict) -> str:
    """Format one SSE frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _progress(node: str, update: dict) -> dict:
    """Translate a node's internal state update into a public progress frame."""
    data: dict = {"node": PUBLIC_STAGE_NAMES.get(node, node)}
    if "route" in update:
        data["route"] = update["route"]
    if "specific_grade" in update:
        data["grade"] = update["specific_grade"]
    if "verification_passed" in update:
        data["faithful"] = update["verification_passed"]
    if "status" in update:
        data["status"] = update["status"]
    if "specific_chunks" in update:
        data["chunks"] = len(update["specific_chunks"])

    attempts = update.get("specific_retrieval_attempts", update.get("specific_revision_attempts"))
    if attempts is not None:
        data["attempts"] = attempts

    verdicts = update.get("specific_verdicts") or update.get("aggregate_verdicts")
    if verdicts is not None:
        data["verdicts"] = [v.verdict for v in verdicts]

    answer = update.get("answer")
    if answer is not None:
        data["claims"] = len(answer.claims)
    return data


def _build_payload(state: dict) -> dict:
    """Collapse the final graph state into the JSON response."""
    answer = state.get("answer")
    verdicts = state.get("specific_verdicts") or state.get("aggregate_verdicts") or []
    answer_claims = answer.claims if answer is not None else []

    claims = []
    for c, v in zip(answer_claims, verdicts):
        src = getattr(v, "source", None)
        claims.append(
            {
                "text": c.text,
                "quote": getattr(c, "quote", None),
                "citation": str(src) if src is not None else None,
                "document": src.filename if src is not None else None,
                "verdict": v.verdict,
                "explanation": v.explanation,
            }
        )

    return {
        "answer": answer.text if answer is not None else "",
        "kind": answer.kind if answer is not None else state.get("route"),
        "claims": claims,
        "faithful": state.get("verification_passed", False),
        "status": state.get("status", "answered"),
        "route": state.get("route"),
        "attempts": state.get("specific_revision_attempts", 0),
    }


def _stub_payload() -> dict:
    """Deterministic fake answer for stub-mode deployment; no graph or model run."""
    return {
        "answer": "This is a stub answer used for deployment testing.",
        "kind": "specific",
        "claims": [],
        "faithful": True,
        "status": "answered",
        "telemetry": {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "latency_ms": 0},
    }


async def _run(question: str) -> AsyncIterator[str]:
    """Stream one QA run as SSE progress frames followed by a final answer frame."""
    if settings.provider == "stub":
        yield _sse("answer", _stub_payload())
        return

    state: dict = {}
    started = time.perf_counter()
    with _ask_span(question) as span:
        trace_id = span.get_span_context().trace_id
        prev_usage = _token_cost.snapshot(trace_id)
        prev_time = started

        async for chunk in graph.astream({"question": question}, stream_mode="updates"):
            for node, update in chunk.items():
                current_usage = _token_cost.snapshot(trace_id)
                now = time.perf_counter()
                metrics = {
                    "tokens_in": current_usage.tokens_in - prev_usage.tokens_in,
                    "tokens_out": current_usage.tokens_out - prev_usage.tokens_out,
                    "cost_usd": round(current_usage.cost - prev_usage.cost, 6),
                    "ms": int((now - prev_time) * 1000),
                    "models": current_usage.models[len(prev_usage.models) :],
                }
                prev_usage, prev_time = current_usage, now

                yield _sse("progress", {**_progress(node, update), "metrics": metrics})
                state.update(update)

        payload = _build_payload(state)
        _record_output(span, payload)
        payload["telemetry"] = _totals(trace_id, int((time.perf_counter() - started) * 1000))
        yield _sse("answer", payload)


async def _run_to_payload(question: str) -> dict:
    """Run one QA run to completion and return the final JSON payload."""
    if settings.provider == "stub":
        return _stub_payload()

    started = time.perf_counter()
    with _ask_span(question) as span:
        trace_id = span.get_span_context().trace_id
        state = await graph.ainvoke({"question": question})
        payload = _build_payload(state)
        _record_output(span, payload)
        payload["telemetry"] = _totals(trace_id, int((time.perf_counter() - started) * 1000))
        return payload


@app.post("/ask", dependencies=[Depends(require_demo_session)])
async def ask(req: AskRequest) -> StreamingResponse:
    return StreamingResponse(_run(req.question), media_type="text/event-stream")


@app.post("/ask_json", dependencies=[Depends(require_demo_session)])
async def ask_json(req: AskRequest) -> dict:
    return await _run_to_payload(req.question)
