#!/usr/bin/env python3
"""
Book visualization tool - converts text into images based on linguistic analysis.

Architecture:
- Color mappers: functions that convert a 0-1 value to RGB tuple
- Metrics: generators that yield 0-1 values from text analysis
- Renderer: single function that combines metric + color mapper into an image

Usage:
    python book_png.py <text_file> [--metric METRIC] [--color COLOR] [--output FILE]
    python book_png.py --list  # show available metrics and color schemes
"""

import argparse
import math
import random
import sys
from pathlib import Path

import nltk
from PIL import Image


# =============================================================================
# COLOR MAPPERS
# Each takes a float (0.0-1.0) and returns an RGB tuple (0-255, 0-255, 0-255)
# =============================================================================

def red_blue(value):
    """High values = red, low values = blue."""
    v = int(value * 255)
    return (v, 0, 255 - v)


def blue_red(value):
    """High values = blue, low values = red."""
    v = int(value * 255)
    return (255 - v, 0, v)


def heat(value):
    """Heat map: black -> red -> yellow -> white."""
    if value < 0.33:
        return (int(value * 3 * 255), 0, 0)
    elif value < 0.66:
        return (255, int((value - 0.33) * 3 * 255), 0)
    else:
        return (255, 255, int((value - 0.66) * 3 * 255))


def grayscale(value):
    """Simple grayscale gradient."""
    v = int(value * 255)
    return (v, v, v)


def green_purple(value):
    """Low = green, high = purple."""
    v = int(value * 255)
    return (v, 255 - v, v)


def rainbow(value):
    """Cycle through hue spectrum."""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(value, 1.0, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def random_per_value(value):
    """Consistent random color per unique value (seeded by value)."""
    random.seed(int(value * 10000))
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


COLOR_MAPPERS = {
    'red-blue': red_blue,
    'blue-red': blue_red,
    'heat': heat,
    'grayscale': grayscale,
    'green-purple': green_purple,
    'rainbow': rainbow,
    'random': random_per_value,
}


# =============================================================================
# METRICS (VALUE EXTRACTORS)
# Each takes a word list and yields normalized floats (0.0-1.0)
# =============================================================================

def word_frequency(words):
    """Color by word frequency (log scale). Common words = high value."""
    freq = nltk.FreqDist(words)
    max_log = math.log(freq[freq.max()])
    for word in words:
        yield math.log(freq[word]) / max_log


def word_frequency_linear(words):
    """Color by word frequency (linear scale). Common words = high value."""
    freq = nltk.FreqDist(words)
    max_freq = freq[freq.max()]
    for word in words:
        yield freq[word] / max_freq


def bigram_probability(words):
    """Color by conditional probability P(word2|word1) for each bigram."""
    bigrams = list(nltk.bigrams(words))
    if not bigrams:
        return
    cfd = nltk.ConditionalFreqDist(bigrams)
    for w1, w2 in bigrams:
        yield cfd[w1].freq(w2)


def bigram_diversity(words):
    """Color by how many different words can follow each word."""
    bigrams = list(nltk.bigrams(words))
    if not bigrams:
        return
    cfd = nltk.ConditionalFreqDist(bigrams)
    max_count = max(len(cfd[c]) for c in cfd.conditions())
    for w1, w2 in bigrams:
        yield len(cfd[w1]) / max_count


def word_length(words):
    """Color by word length (normalized)."""
    max_len = max(len(w) for w in words) if words else 1
    for word in words:
        yield len(word) / max_len


def word_position(words):
    """Color by position in text (gradient from start to end)."""
    total = len(words)
    for i, _ in enumerate(words):
        yield i / total if total > 0 else 0


def unique_word_id(words):
    """Assign consistent value to each unique word (for random coloring)."""
    word_ids = {}
    unique_count = 0
    # First pass: assign IDs
    for word in words:
        if word not in word_ids:
            word_ids[word] = unique_count
            unique_count += 1
    # Second pass: yield normalized values
    max_id = unique_count - 1 if unique_count > 1 else 1
    for word in words:
        yield word_ids[word] / max_id


METRICS = {
    'word-freq': word_frequency,
    'word-freq-linear': word_frequency_linear,
    'bigram-prob': bigram_probability,
    'bigram-diversity': bigram_diversity,
    'word-length': word_length,
    'word-position': word_position,
    'unique-word': unique_word_id,
}


# =============================================================================
# RENDERER
# =============================================================================

def render(values, color_fn, output_path):
    """
    Render a sequence of values as a square image.

    Args:
        values: iterable of floats (0.0-1.0)
        color_fn: function that maps float -> RGB tuple
        output_path: where to save the PNG
    """
    values = list(values)
    if not values:
        print("No values to render", file=sys.stderr)
        return

    size = int(math.ceil(math.sqrt(len(values))))
    img = Image.new("RGB", (size, size), color=(0, 0, 0))

    for i, val in enumerate(values):
        x = i % size
        y = i // size
        # Clamp value to 0-1 range
        val = max(0.0, min(1.0, val))
        img.putpixel((x, y), color_fn(val))

    img.save(output_path)
    print(f"Saved: {output_path} ({size}x{size} pixels, {len(values)} values)")


# =============================================================================
# CLI
# =============================================================================

def list_options():
    """Print available metrics and color schemes."""
    print("Available metrics:")
    for name, fn in METRICS.items():
        doc = fn.__doc__.split('\n')[0] if fn.__doc__ else ''
        print(f"  {name:20} {doc}")

    print("\nAvailable color schemes:")
    for name, fn in COLOR_MAPPERS.items():
        doc = fn.__doc__.split('\n')[0] if fn.__doc__ else ''
        print(f"  {name:20} {doc}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert text files into visual representations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python book_png.py book.txt
  python book_png.py book.txt --metric word-freq --color heat
  python book_png.py book.txt -m bigram-diversity -c rainbow -o output.png
  python book_png.py --list
        """
    )
    parser.add_argument('file', nargs='?', help='Text file to visualize')
    parser.add_argument('--list', action='store_true', help='List available metrics and colors')
    parser.add_argument('-m', '--metric', default='word-freq',
                        choices=list(METRICS.keys()),
                        help='Metric to visualize (default: word-freq)')
    parser.add_argument('-c', '--color', default='red-blue',
                        choices=list(COLOR_MAPPERS.keys()),
                        help='Color scheme (default: red-blue)')
    parser.add_argument('-o', '--output', help='Output filename (default: <input>-<metric>.png)')

    args = parser.parse_args()

    if args.list:
        list_options()
        return

    if not args.file:
        parser.print_help()
        return

    input_path = Path(args.file)
    if not input_path.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    # Read and tokenize
    text = input_path.read_text(encoding='utf-8', errors='replace')
    words = nltk.word_tokenize(text)
    print(f"Loaded {len(words)} words from {args.file}")

    # Generate output filename
    if args.output:
        output_path = args.output
    else:
        output_path = input_path.stem + f"-{args.metric}.png"

    # Get metric and color functions
    metric_fn = METRICS[args.metric]
    color_fn = COLOR_MAPPERS[args.color]

    # Render
    values = metric_fn(words)
    render(values, color_fn, output_path)


if __name__ == '__main__':
    main()
