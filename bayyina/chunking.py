"""Offset-preserving chunker. See README.md for the seven rules."""
from __future__ import annotations

from bayyina.types import Chunk


def chunk_text(doc_id: str, text: str, max_chars: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split `text` into chunks whose offsets point into the ORIGINAL text.

    ValueError unless max_chars > 0 and 0 <= overlap < max_chars.
    """
    if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
        raise ValueError(f"Invalid max_chars={max_chars} or overlap={overlap}")

    if not text or text.isspace():
        return []

    # Find the first non-whitespace character in text
    s = 0
    while s < len(text) and text[s].isspace():
        s += 1
    if s == len(text):
        return []

    sentence_markers = {".", "!", "?", "؟", "۔", "\n"}
    chunks: list[Chunk] = []

    def is_word_start(pos: int) -> bool:
        if pos < 0 or pos >= len(text) or text[pos].isspace():
            return False
        return pos == 0 or text[pos - 1].isspace()

    while s < len(text):
        if len(text) - s <= max_chars:
            raw = text[s:]
            stripped = raw.rstrip()
            c_end = s + len(stripped)
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}#{len(chunks)}",
                    start=s,
                    end=c_end,
                    text=stripped,
                )
            )
            break

        window = text[s : s + max_chars]
        chosen_c = None

        # Preference 1: Paragraph break \n\n wholly inside window at window positions c-2, c-1
        for c in range(max_chars, max_chars // 2 - 1, -1):
            if c >= 2 and window[c - 2 : c] == "\n\n":
                chosen_c = c
                break

        # Preference 2: Sentence end (. ! ? ؟ ۔ or \n) wholly inside window at window position c-1
        if chosen_c is None:
            for c in range(max_chars, max_chars // 2 - 1, -1):
                if c >= 1 and window[c - 1] in sentence_markers:
                    chosen_c = c
                    break

        # Preference 3: Space (any whitespace character) wholly inside window at window position c-1
        if chosen_c is None:
            for c in range(max_chars, max_chars // 2 - 1, -1):
                if c >= 1 and window[c - 1].isspace():
                    chosen_c = c
                    break

        # Preference 4: Hard cut
        if chosen_c is None:
            chosen_c = max_chars

        raw = text[s : s + chosen_c]
        stripped = raw.rstrip()
        c_end = s + len(stripped)

        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}#{len(chunks)}",
                start=s,
                end=c_end,
                text=stripped,
            )
        )

        ended_mid_word = (
            c_end < len(text)
            and not text[c_end - 1].isspace()
            and not text[c_end].isspace()
        )

        candidate_start = max(s + 1, c_end - overlap)
        word_starts = [w for w in range(candidate_start, c_end) if is_word_start(w)]

        if word_starts:
            next_s = word_starts[0]
        elif ended_mid_word:
            next_s = candidate_start
        else:
            pos = c_end
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos >= len(text):
                break
            next_s = pos

        s = next_s

    return chunks
