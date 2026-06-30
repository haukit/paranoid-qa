"""FastAPI server

Run: uv run uvicorn paranoid_qa.server:app --reload"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from pydantic import BaseModel

from paranoid_qa.graph import build_graph
from paranoid_qa.tracing import setup_tracing

setup_tracing()
_tracer = trace.get_tracer("paranoid-qa")
app = FastAPI(title="paranoid-qa")
graph = build_graph()


class AskRequest(BaseModel):
    question: str


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


async def _run(question: str) -> AsyncIterator[str]:
    state = {}
    with _tracer.start_as_current_span("ask") as span:
        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value
        )
        span.set_attribute(SpanAttributes.INPUT_VALUE, question)
        span.set_attribute("question", question)

        async for chunk in graph.astream({"question": question}, stream_mode="updates"):
            for node, update in chunk.items():
                yield _sse("progress", _progress(node, update))
                state.update(update)

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

        payload = {
            "answer": answer.text if answer else "",
            "claims": claims,
            "faithful": state.get("faithful", False),
        }
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(payload))
        span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json")
        span.set_attribute("faithful", bool(state.get("faithful", False)))

        yield _sse("answer", payload)


@app.post("/ask")
async def ask(req: AskRequest) -> StreamingResponse:
    return StreamingResponse(_run(req.question), media_type="text/event-stream")
