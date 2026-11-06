"""Citation checks. You write this file."""
from __future__ import annotations

from bayyina.types import Hit


def verify_citation(quote: str, document: str, start: int, end: int) -> bool:
    raise NotImplementedError


def format_citation(hit: Hit, max_quote: int = 160) -> str:
    """[doc_id:start-end] "quote" """
    raise NotImplementedError
