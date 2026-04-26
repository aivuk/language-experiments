"""High-level rendering pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .html_viewer import write_viewer
from .metrics import metric_values, normalize
from .render import render_values
from .text import apply_filters, load_text, tokenize


@dataclass
class RenderOptions:
    metric: str = "word-freq"
    color: str = "red-blue"
    window_size: int | None = None
    window_step: int | None = None
    ignore_case: bool = False
    ignore_punctuation: bool = False
    ignore_numbers: bool = False
    normalize_values: bool = True
    domain: tuple[float, float] | None = None


@dataclass
class RenderResult:
    source: Path
    output: Path
    image: Image.Image
    size: int
    values: list[float]
    labels: list[str]
    raw_values: list[float]
    token_count: int


def prepare_values(source: Path, options: RenderOptions) -> tuple[list[float], list[str], int]:
    tokens = tokenize(load_text(source))
    tokens = apply_filters(
        tokens,
        ignore_case=options.ignore_case,
        ignore_punctuation=options.ignore_punctuation,
        ignore_numbers=options.ignore_numbers,
    )
    values, labels = metric_values(
        tokens,
        options.metric,
        window_size=options.window_size,
        window_step=options.window_step,
    )
    return values, labels, len(tokens)


def render_book(source: Path, output: Path, options: RenderOptions, *, html: bool = False) -> RenderResult:
    raw_values, labels, token_count = prepare_values(source, options)
    values = normalize(raw_values, domain=options.domain) if options.normalize_values else raw_values
    image, size, label_values = render_values(values, output, color=options.color, labels=labels)
    if html:
        html_output = output.with_suffix(".html")
        write_viewer(
            image,
            html_output,
            title=source.stem,
            subtitle=subtitle(options, token_count, len(values)),
            labels=[label for label, _value in label_values],
            values=[value for _label, value in label_values],
        )
    return RenderResult(
        source=source,
        output=output,
        image=image,
        size=size,
        values=values,
        labels=labels,
        raw_values=raw_values,
        token_count=token_count,
    )


def subtitle(options: RenderOptions, token_count: int, value_count: int) -> str:
    mode = f"window {options.window_size}" if options.window_size else "per token"
    return f"{options.metric} · {options.color} · {mode} · {token_count:,} tokens · {value_count:,} values"

