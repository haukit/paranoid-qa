"""Load test for /ask."""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

URL = "http://localhost:8000/ask"
GOLD = Path("evals/data/retrieval_gold.jsonl")


def load_questions(n: int) -> list[str]:
    rows = [json.loads(line) for line in GOLD.read_text().splitlines() if line.strip()]
    qs = [r["question"] for r in rows if r.get("path", "specific") == "specific"]
    return (qs * (n // len(qs) + 1))[:n]  # repeat to fill n


async def one_request(client: httpx.AsyncClient, question: str) -> dict:
    t0 = time.perf_counter()
    ok, status = False, None
    try:
        async with client.stream("POST", URL, json={"question": question}, timeout=180) as resp:
            status = resp.status_code
            async for line in resp.aiter_lines():
                if line.startswith("event: answer"):
                    ok = True  # got a complete grounded answer
    except Exception as e:
        status = type(e).__name__
    return {"latency": time.perf_counter() - t0, "ok": ok, "status": status}


async def run_level(concurrency: int, questions: list[str]) -> dict:
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:

        async def worker(q: str) -> dict:
            async with sem:
                return await one_request(client, q)

        t0 = time.perf_counter()
        results = await asyncio.gather(*(worker(q) for q in questions))
        wall = time.perf_counter() - t0

    lat = sorted(r["latency"] for r in results if r["ok"])

    def pct(p: int) -> float:
        if not lat:
            return 0.0
        if len(lat) == 1:
            return lat[0]
        return statistics.quantiles(lat, n=100)[p - 1]

    return {
        "concurrency": concurrency,
        "p50": pct(50),
        "p95": pct(95),
        "throughput": len(results) / wall,
        "errors": sum(1 for r in results if not r["ok"]),
    }


async def main() -> None:
    n = 40
    questions = load_questions(n)
    print(f"{'conc':>4} {'p50(s)':>7} {'p95(s)':>7} {'req/s':>6} {'err':>4}")
    for c in (1, 2, 4, 8):
        r = await run_level(c, questions)
        print(
            f"{r['concurrency']:>4} {r['p50']:>7.2f} {r['p95']:>7.2f} {r['throughput']:>6.2f} {r['errors']:>4}"
        )


if __name__ == "__main__":
    asyncio.run(main())
