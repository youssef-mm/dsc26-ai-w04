"""Arabic-aware normalisation and tokenisation. You write this file."""
from __future__ import annotations


def normalize_ar(text: str) -> str:
    """Remove U+064B-U+0652 and U+0670; remove tatweel U+0640; map أ إ آ ٱ -> ا, ى -> ي, ة -> ه;
    map ٠-٩ and ۰-۹ to 0-9; lowercase; collapse runs of whitespace to one space and strip."""
    raise NotImplementedError


def tokenize(text: str) -> list[str]:
    """normalize_ar, then every match of the regex \\w+ (Unicode)."""
    raise NotImplementedError
