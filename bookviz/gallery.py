"""Static gallery site generation."""

from __future__ import annotations

import html
import shutil
from pathlib import Path

from .metrics import ALL_METRICS
from .pipeline import RenderOptions, render_book, subtitle
from .text import slugify


def generate_gallery(
    input_dir: Path,
    output_dir: Path,
    *,
    metrics: list[str],
    color: str,
    window_size: int | None,
    window_step: int | None,
    limit: int | None = None,
) -> list[Path]:
    books = sorted(input_dir.glob("*.txt"))
    if limit:
        books = books[:limit]
    if not books:
        raise ValueError(f"No .txt files found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(exist_ok=True)
    pages: list[dict[str, str]] = []
    generated: list[Path] = []
    for book in books:
        for metric in metrics:
            if metric not in ALL_METRICS:
                raise ValueError(f"Unknown metric: {metric}")
            options = RenderOptions(metric=metric, color=color, window_size=window_size, window_step=window_step)
            slug = f"{slugify(book.stem)}-{slugify(metric)}"
            png_path = assets_dir / f"{slug}.png"
            result = render_book(book, png_path, options, html=True)
            html_name = f"{slug}.html"
            shutil.move(str(png_path.with_suffix(".html")), output_dir / html_name)
            pages.append(
                {
                    "title": book.stem,
                    "metric": metric,
                    "image": f"assets/{png_path.name}",
                    "viewer": html_name,
                    "subtitle": subtitle(options, result.token_count, len(result.values)),
                }
            )
            generated.extend([png_path, output_dir / html_name])
    index_path = output_dir / "index.html"
    index_path.write_text(index_html(pages), encoding="utf-8")
    generated.append(index_path)
    return generated


def index_html(pages: list[dict[str, str]]) -> str:
    cards = "\n".join(
        f"""<article>
      <a href="{html.escape(page['viewer'])}"><img src="{html.escape(page['image'])}" alt=""></a>
      <div>
        <h2>{html.escape(page['title'])}</h2>
        <p>{html.escape(page['metric'])}</p>
        <small>{html.escape(page['subtitle'])}</small>
      </div>
    </article>"""
        for page in pages
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Language Experiments Gallery</title>
  <style>
    body {{ margin: 0; background: #f5f3ef; color: #202020; font-family: system-ui, sans-serif; }}
    header {{ padding: 28px clamp(18px, 4vw, 48px) 18px; border-bottom: 1px solid #d8d3c8; }}
    h1 {{ margin: 0; font-size: clamp(28px, 5vw, 56px); letter-spacing: 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 18px; padding: clamp(18px, 4vw, 48px); }}
    article {{ background: #fff; border: 1px solid #d9d9d9; border-radius: 8px; overflow: hidden; }}
    img {{ width: 100%; aspect-ratio: 1; object-fit: contain; image-rendering: pixelated; background: #181818; display: block; }}
    article div {{ padding: 12px; }}
    h2 {{ margin: 0 0 4px; font-size: 16px; }}
    p {{ margin: 0 0 8px; color: #5c3525; font-weight: 650; }}
    small {{ color: #666; line-height: 1.35; display: block; }}
  </style>
</head>
<body>
  <header>
    <h1>Language Experiments Gallery</h1>
  </header>
  <main class="grid">
    {cards}
  </main>
</body>
</html>
"""

