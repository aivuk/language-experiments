"""Text loading, tokenization, and windowing utilities."""

from __future__ import annotations

import re
from pathlib import Path

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def is_punctuation(token: str) -> bool:
    return not any(ch.isalnum() for ch in token)


def apply_filters(
    tokens: list[str],
    *,
    ignore_case: bool = False,
    ignore_punctuation: bool = False,
    ignore_numbers: bool = False,
) -> list[str]:
    filtered = [token.lower() for token in tokens] if ignore_case else list(tokens)
    if ignore_punctuation:
        filtered = [token for token in filtered if not is_punctuation(token)]
    if ignore_numbers:
        filtered = [token for token in filtered if not token.isnumeric()]
    return filtered


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "untitled"


def windows(tokens: list[str], size: int, step: int | None = None) -> list[list[str]]:
    if size <= 0:
        raise ValueError("window size must be greater than zero")
    step = step or size
    if step <= 0:
        raise ValueError("window step must be greater than zero")
    return [tokens[start : start + size] for start in range(0, len(tokens), step) if tokens[start : start + size]]

