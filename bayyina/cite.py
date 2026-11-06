"""Citation checks."""
from __future__ import annotations

from bayyina.types import Hit


def verify_citation(quote: str, document: str, start: int, end: int) -> bool:
    """Verify that quote occurs inside document[start:end] (whitespace collapsed).

    True only when 0 <= start < end <= len(document), the quote is not blank,
    and the quote (whitespace collapsed) occurs inside document[start:end] (whitespace collapsed).
    """
    if not (0 <= start < end <= len(document)):
        return False
    if not quote or not quote.strip():
        return False
    clean_quote = " ".join(quote.split())
    clean_span = " ".join(document[start:end].split())
    return clean_quote in clean_span


def format_citation(hit: Hit, max_quote: int = 160) -> str:
    """Format a hit as [doc_id:start-end] "quote".

    The quote is the hit's text with whitespace collapsed.
    When it is longer than max_quote, take its first max_quote characters,
    cut back to the last space inside them (dropping the space), and append …
    """
    quote = " ".join(hit.text.split())
    if len(quote) > max_quote:
        prefix = quote[:max_quote]
        last_space = prefix.rfind(" ")
        if last_space != -1:
            prefix = prefix[:last_space]
        quote = prefix + "…"
    return f'[{hit.doc_id}:{hit.start}-{hit.end}] "{quote}"'
