"""Typed contracts shared across the workflow and both QA paths.

Split by concern:
- `common`: stable cross-path primitives (route kind, run status, source reference).
- `specific`: quote-grounded specific-path claims, answers, and verdicts.
- `aggregate`: reference-grounded aggregate-path claims, answers, and verdicts.
- `state`: the single LangGraph state passed between all nodes.
"""
