"""PNG rendering helpers."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

from .colors import COLOR_MAPPERS


def render_values(
    values: list[float],
    output_path: Path,
    *,
    color: str = "red-blue",
    labels: list[str] | None = None,
) -> tuple[Image.Image, int, list[tuple[str, float]]]:
    if not values:
        raise ValueError("No values to render")
    color_fn = COLOR_MAPPERS[color]
    size = int(math.ceil(math.sqrt(len(values))))
    image = Image.new("RGB", (size, size), color=(0, 0, 0))
    for index, value in enumerate(values):
        image.putpixel((index % size, index // size), color_fn(value))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    label_values = list(zip(labels or [str(index) for index in range(len(values))], values))
    return image, size, label_values


def render_comparison(
    rendered: list[tuple[str, Image.Image]],
    output_path: Path,
    *,
    cell_padding: int = 18,
    label_height: int = 26,
) -> Image.Image:
    if not rendered:
        raise ValueError("No images to compare")
    max_width = max(image.width for _title, image in rendered)
    max_height = max(image.height for _title, image in rendered)
    width = len(rendered) * (max_width + cell_padding) + cell_padding
    height = max_height + label_height + cell_padding * 2
    canvas = Image.new("RGB", (width, height), color=(24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    x = cell_padding
    for title, image in rendered:
        draw.text((x, cell_padding), title[:42], fill=(235, 235, 235))
        canvas.paste(image, (x, cell_padding + label_height))
        x += max_width + cell_padding
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return canvas

