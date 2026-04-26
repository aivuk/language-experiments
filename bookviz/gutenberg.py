"""Project Gutenberg download support."""

from __future__ import annotations

import json
import re
from pathlib import Path

import requests

from .text import slugify

GUTENBERG_CACHE_URLS = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
]


def fetch_gutenberg(book_id: int, output_dir: Path, *, title: str | None = None, force: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    guessed_title = title or f"gutenberg-{book_id}"
    path = output_dir / f"{book_id}-{slugify(guessed_title)}.txt"
    metadata_path = path.with_suffix(".json")
    if path.exists() and not force:
        return path
    response = None
    source_url = ""
    for template in GUTENBERG_CACHE_URLS:
        source_url = template.format(id=book_id)
        response = requests.get(source_url, timeout=30)
        if response.ok and response.text.strip():
            break
    if response is None or not response.ok:
        raise RuntimeError(f"Could not download Gutenberg book {book_id}")
    text = strip_gutenberg_boilerplate(response.text)
    detected_title = title or detect_title(response.text) or f"gutenberg-{book_id}"
    final_path = output_dir / f"{book_id}-{slugify(detected_title)}.txt"
    final_path.write_text(text, encoding="utf-8")
    metadata_path = final_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps({"id": book_id, "title": detected_title, "source_url": source_url}, indent=2),
        encoding="utf-8",
    )
    return final_path


def detect_title(text: str) -> str | None:
    for line in text.splitlines()[:80]:
        match = re.match(r"\s*Title:\s*(.+)", line)
        if match:
            return match.group(1).strip()
    return None


def strip_gutenberg_boilerplate(text: str) -> str:
    start_patterns = [
        r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*",
        r"\*\*\* START OF .*?\*\*\*",
    ]
    end_patterns = [
        r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*",
        r"\*\*\* END OF .*?\*\*\*",
    ]
    stripped = text
    for pattern in start_patterns:
        match = re.search(pattern, stripped, re.IGNORECASE | re.DOTALL)
        if match:
            stripped = stripped[match.end() :]
            break
    for pattern in end_patterns:
        match = re.search(pattern, stripped, re.IGNORECASE | re.DOTALL)
        if match:
            stripped = stripped[: match.start()]
            break
    return stripped.strip() + "\n"

