"""Color maps for normalized metric values."""

from __future__ import annotations

import colorsys
import random
from collections.abc import Callable

Color = tuple[int, int, int]
ColorMapper = Callable[[float], Color]


def red_blue(value: float) -> Color:
    value = clamp(value)
    v = int(value * 255)
    return (v, 0, 255 - v)


def blue_red(value: float) -> Color:
    value = clamp(value)
    v = int(value * 255)
    return (255 - v, 0, v)


def heat(value: float) -> Color:
    value = clamp(value)
    if value < 0.33:
        return (int(value * 3 * 255), 0, 0)
    if value < 0.66:
        return (255, int((value - 0.33) * 3 * 255), 0)
    return (255, 255, int((value - 0.66) * 3 * 255))


def grayscale(value: float) -> Color:
    value = clamp(value)
    v = int(value * 255)
    return (v, v, v)


def green_purple(value: float) -> Color:
    value = clamp(value)
    v = int(value * 255)
    return (v, 255 - v, v)


def rainbow(value: float) -> Color:
    value = clamp(value)
    r, g, b = colorsys.hsv_to_rgb(value, 1.0, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def random_per_value(value: float) -> Color:
    value = clamp(value)
    rng = random.Random(int(value * 10000))
    return (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


COLOR_MAPPERS: dict[str, ColorMapper] = {
    "red-blue": red_blue,
    "blue-red": blue_red,
    "heat": heat,
    "grayscale": grayscale,
    "green-purple": green_purple,
    "rainbow": rainbow,
    "random": random_per_value,
}

