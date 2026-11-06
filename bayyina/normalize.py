"""Arabic-aware normalisation and tokenisation."""
from __future__ import annotations

import re

_TRANS = {}

# Remove U+064B–U+0652 (Arabic diacritics / tashkeel) and U+0670 (superscript alef)
for _cp in range(0x064B, 0x0653):
    _TRANS[_cp] = None
_TRANS[0x0670] = None

# Remove tatweel U+0640
_TRANS[0x0640] = None

# Map أ إ آ ٱ -> ا
for _ch in "أإآٱ":
    _TRANS[ord(_ch)] = "ا"

# Map ى -> ي
_TRANS[ord("ى")] = "ي"

# Map ة -> ه
_TRANS[ord("ة")] = "ه"

# Map ٠–٩ (Arabic-Indic) and ۰–۹ (Eastern Arabic-Indic / Persian) to 0–9
for _i in range(10):
    _TRANS[0x0660 + _i] = str(_i)
    _TRANS[0x06F0 + _i] = str(_i)


def normalize_ar(text: str) -> str:
    """Remove U+064B-U+0652 and U+0670; remove tatweel U+0640; map أ إ آ ٱ -> ا, ى -> ي, ة -> ه;
    map ٠-٩ and ۰-۹ to 0-9; lowercase; collapse runs of whitespace to one space and strip.
    Nothing else: in particular ؤ and ئ stay as they are."""
    cleaned = text.translate(_TRANS).lower()
    return " ".join(cleaned.split())


def tokenize(text: str) -> list[str]:
    """normalize_ar, then every match of the regex \\w+ (Unicode). Franco tokens such as sha7en stay intact."""
    norm = normalize_ar(text)
    return re.findall(r"\w+", norm)
