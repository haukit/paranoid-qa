"""Generate critic_gold.jsonl: labelled triples (claims, quote, source) for the critic eval.

For each sampled chunk, we ask a generator model to write a claim plus a verbatim
supporting quote; we keep only triples whose quote are actually in the chunk (using
the locator function in the critic).

We then manufacture negatives from these."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Literal, cast

from llama_index.core import StorageContext
from llama_index.core.schema import BaseNode
from pydantic import BaseModel, Field

from paranoid_qa.config import STORAGE_DIR
from paranoid_qa.contracts.specific import RetrievedChunk
from paranoid_qa.llm.factory import make_structured
from paranoid_qa.specific.verification import locate_quote

SEED = 23
PER_DOC = 4  # questions per document; sparse reports are capped at what they have
MIN_CHARS = 500  # skip covers, toc since those make poor questions
GEN_MODEL = "gpt-4o"  # pick a stronger model than the actual judge model
OUT_PATH = Path("evals/data/critic_gold.jsonl")
POSITIVES_PATH = Path("evals/data/critic_positives.jsonl")  # cache of LLM-generated seeds
CONTRADICTED_PATH = Path("evals/data/critic_contradicted.jsonl")  # cache of LLM mutations


class TrueTriple(BaseModel):
    claim: str = Field(description="One atomic factual statement the passage clearly supports.")
    quote: str = Field(
        description="A span copied VERBATIM from the passage that supports the claim."
    )


class Mutation(BaseModel):
    mutated_claim: str = Field(
        description="The claim with ONE concrete fact changed to a plausible but wrong value the source contradicts; everything else identical."
    )


class Check(BaseModel):
    verdict: Literal["supported", "unsupported", "contradicted"]


def chunk_id(node: BaseNode) -> str:
    m = node.metadata
    return f"{m['file_name']}#{m.get('page_label')}"


def load_chunks() -> list[BaseNode]:
    storage = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    return list(storage.docstore.docs.values())


def as_chunk(node: BaseNode) -> RetrievedChunk:
    return {
        "text": node.get_content(),
        "document": node.metadata["file_name"],
        "page": node.metadata.get("page_label"),
    }


def sample_chunks(nodes: list[BaseNode], per_doc: int, seed: int, min_chars: int) -> list[BaseNode]:
    rng = random.Random(seed)
    by_doc: defaultdict[str, list[BaseNode]] = defaultdict(list)
    for n in nodes:
        if len(n.get_content()) >= min_chars:
            by_doc[n.metadata["file_name"]].append(n)
    out: list[BaseNode] = []
    for doc in sorted(by_doc):
        chunks = by_doc[doc]
        out.extend(rng.sample(chunks, k=min(per_doc, len(chunks))))

    return out


GENERATE_SYS = """You are given a PASSAGE from an NTSB accident report. Produce two things:
- claim: one atomic fact the passage clearly supports, stated IN YOUR OWN WORDS. Rephrase it so
  that confirming it takes understanding, not word-matching; do not copy or lightly trim a span
  of the passage. State a fact about the accident or its investigation, never about the
  document's structure (no "the report contains a section on ...").
- quote: a span copied VERBATIM (word for word) from the passage that supports the claim. It must
  appear in the passage exactly; never paraphrase or shorten it with ellipses.
Rephrase the claim, but keep it fully supported by the quote: do not add any detail the quote
does not establish."""

MUTATE_SYS = """You are given a SOURCE passage, a CLAIM it supports, and the QUOTE that supports it.
Rewrite the CLAIM into a near-miss the SOURCE now CONTRADICTS: change ONE concrete fact (a number,
name, direction, cause, date, or part) to a plausible but incorrect value that conflicts with the
source. Keep the rest of the wording identical. Do not make it merely vague or unsupported; the
source must actively contradict it."""

CHECK_SYS = """Given a SOURCE and a CLAIM, does the source support the claim, contradict it, or
neither? Answer with one verdict: supported, contradicted, or unsupported."""


def cached(path: Path, build: Callable[[], list[dict]]) -> list[dict]:
    """Run build() once and cache its rows to path; reuse on later runs. rm the file to refresh."""
    if path.exists():
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def make_row(i: int, node: BaseNode, t: TrueTriple) -> dict:
    return {
        "id": f"cg_{i:04d}",
        "gold": "supported",
        "claim": t.claim,
        "quote": t.quote,
        "source_text": node.get_content(),
        "document": node.metadata["file_name"],
        "page": node.metadata.get("page_label"),
        "origin": "true",
    }


def generate_positives() -> list[dict]:
    """Generate true triples.

    Cached to POSITIVES_PATH; delete that file to force regeneration."""
    if POSITIVES_PATH.exists():
        return [
            json.loads(line) for line in POSITIVES_PATH.read_text().splitlines() if line.strip()
        ]

    sample = sample_chunks(load_chunks(), PER_DOC, SEED, MIN_CHARS)
    gen = make_structured(TrueTriple, model=GEN_MODEL, temperature=0)

    rows = []
    for i, node in enumerate(sample, 1):
        t = cast(TrueTriple, gen.invoke([("system", GENERATE_SYS), ("human", node.get_content())]))
        if locate_quote(t.quote, [as_chunk(node)]) is None:
            print(f"  skip {chunk_id(node)}: quote not verbatim in chunk")
            continue
        rows.append(make_row(i, node, t))

    POSITIVES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with POSITIVES_PATH.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    return rows


def make_unsupported(positives: list[dict], seed: int) -> list[dict]:
    """Make an unsupported claim for each positive claim.

    Each positive's claim keeps its words but gets a real quote and source lifted
    from a different document, so that the source is irrelevant to the claim."""
    rng = random.Random(seed)
    rows = []
    for p in positives:
        partners = [q for q in positives if q["document"] != p["document"]]
        if not partners:
            continue
        partner = rng.choice(partners)
        rows.append(
            {
                "id": f"{p['id']}_unsup",
                "gold": "unsupported",
                "claim": p["claim"],
                "quote": partner["quote"],
                "source_text": partner["source_text"],
                "document": partner["document"],
                "page": partner["page"],
                "origin": "synthetic_unsupported_swap",
                "seed_id": p["id"],  # so deleting a junk seed's derivatives is one filter
                "partner_id": partner["id"],
            }
        )
    return rows


def make_contradicted(positives: list[dict], model: str) -> list[dict]:
    mut = make_structured(Mutation, model=model, temperature=0)
    chk = make_structured(Check, model=model, temperature=0)
    rows: list[dict] = []
    for p in positives:
        prompt = f"SOURCE:\n{p['source_text']}\n\nCLAIM: {p['claim']}\n\nQUOTE: {p['quote']}"
        m = cast(Mutation, mut.invoke([("system", MUTATE_SYS), ("human", prompt)]))
        v = cast(
            Check,
            chk.invoke(
                [
                    ("system", CHECK_SYS),
                    ("human", f"SOURCE:\n{p['source_text']}\n\nCLAIM: {m.mutated_claim}"),
                ]
            ),
        )
        if v.verdict != "contradicted":
            print(f"  skip {p['id']}: mutation came out {v.verdict}, not contradicted")
            continue
        rows.append(
            {
                "id": f"{p['id']}_contra",
                "gold": "contradicted",
                "claim": m.mutated_claim,
                "quote": p["quote"],
                "source_text": p["source_text"],
                "document": p["document"],
                "page": p["page"],
                "origin": "synthetic_contradicted_mutation",
                "seed_id": p["id"],
            }
        )
    return rows


def row_chunk(r: dict) -> RetrievedChunk:
    return {"text": r["source_text"], "document": r["document"], "page": r["page"]}


def make_fabricated(positives: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for p in positives:
        partners = [q for q in positives if q["document"] != p["document"]]
        if not partners:
            continue
        partner = rng.choice(partners)
        if locate_quote(partner["quote"], [row_chunk(p)]) is not None:
            print(f"  skip {p['id']}: foreign quote coincidentally present in source")
            continue
        rows.append(
            {
                "id": f"{p['id']}_fab",
                "gold": "fabricated",
                "claim": p["claim"],
                "quote": partner["quote"],
                "source_text": p["source_text"],
                "document": p["document"],
                "page": p["page"],
                "origin": "synthetic_fabricated_foreign_quote",
                "seed_id": p["id"],
                "partner_id": partner["id"],
            }
        )
    return rows


def main() -> None:
    positives = generate_positives()
    unsupported = make_unsupported(positives, SEED)
    contradicted = cached(CONTRADICTED_PATH, lambda: make_contradicted(positives, GEN_MODEL))
    fabricated = make_fabricated(
        positives, SEED + 1
    )  # +1 so partners differ from the unsupported swaps
    rows = positives + unsupported + contradicted + fabricated
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(
        f"wrote {len(rows)}: {len(positives)} supported, {len(unsupported)} unsupported, "
        f"{len(contradicted)} contradicted, {len(fabricated)} fabricated"
    )


if __name__ == "__main__":
    main()
