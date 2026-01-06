#!/usr/bin/env python3
"""
Book visualization tool - converts text into images based on linguistic analysis.

Architecture:
- Color mappers: functions that convert a 0-1 value to RGB tuple
- Metrics: generators that yield 0-1 values from text analysis
- Renderer: single function that combines metric + color mapper into an image

Usage:
    python book_png.py <text_file> [--metric METRIC] [--color COLOR] [--output FILE]
    python book_png.py <text_file> --html  # generate interactive HTML viewer
    python book_png.py --list  # show available metrics and color schemes
"""

import argparse
import base64
import io
import json
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

def render(values, color_fn, output_path, words=None):
    """
    Render a sequence of values as a square image.

    Args:
        values: iterable of floats (0.0-1.0)
        color_fn: function that maps float -> RGB tuple
        output_path: where to save the PNG
        words: optional list of words (for JSON export)

    Returns:
        tuple: (PIL.Image, size, list of (word, value) pairs)
    """
    values = list(values)
    if not values:
        print("No values to render", file=sys.stderr)
        return None, 0, []

    size = int(math.ceil(math.sqrt(len(values))))
    img = Image.new("RGB", (size, size), color=(0, 0, 0))

    word_data = []
    for i, val in enumerate(values):
        x = i % size
        y = i // size
        # Clamp value to 0-1 range
        clamped = max(0.0, min(1.0, val))
        img.putpixel((x, y), color_fn(clamped))
        if words and i < len(words):
            word_data.append((words[i], val))

    img.save(output_path)
    print(f"Saved: {output_path} ({size}x{size} pixels, {len(values)} values)")
    return img, size, word_data


def render_html(img, size, word_data, output_path, metric_name, color_name, source_file):
    """
    Generate an interactive HTML viewer with zoom/pan and word tooltips.

    Args:
        img: PIL.Image object
        size: image dimension (size x size)
        word_data: list of (word, value) tuples
        output_path: where to save the HTML file
        metric_name: name of the metric used
        color_name: name of the color scheme used
        source_file: original text file name
    """
    # Convert image to base64
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

    # Create word lookup (just the words array - position is implicit)
    words_json = json.dumps([w for w, v in word_data])
    values_json = json.dumps([round(v, 4) for w, v in word_data])

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{source_file} - {metric_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #1a1a1a;
            color: #fff;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
            overflow: hidden;
            height: 100vh;
        }}
        #header {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            padding: 10px 20px;
            background: rgba(0,0,0,0.8);
            z-index: 100;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        #header h1 {{
            font-size: 14px;
            font-weight: normal;
        }}
        #header .info {{
            font-size: 12px;
            color: #888;
        }}
        #controls {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        #controls button {{
            background: #333;
            border: 1px solid #555;
            color: #fff;
            padding: 5px 12px;
            cursor: pointer;
            border-radius: 3px;
        }}
        #controls button:hover {{
            background: #444;
        }}
        #zoom-level {{
            font-size: 12px;
            color: #888;
            min-width: 60px;
        }}
        #container {{
            position: absolute;
            top: 50px;
            left: 0;
            right: 0;
            bottom: 0;
            overflow: hidden;
            cursor: grab;
        }}
        #container.dragging {{
            cursor: grabbing;
        }}
        #canvas-wrapper {{
            position: absolute;
            transform-origin: 0 0;
        }}
        #image {{
            display: block;
            image-rendering: pixelated;
            image-rendering: crisp-edges;
        }}
        #tooltip {{
            position: fixed;
            background: rgba(0,0,0,0.9);
            border: 1px solid #444;
            padding: 8px 12px;
            border-radius: 4px;
            pointer-events: none;
            z-index: 200;
            display: none;
            font-size: 13px;
            max-width: 300px;
        }}
        #tooltip .word {{
            font-size: 16px;
            font-weight: bold;
            color: #fff;
            margin-bottom: 4px;
        }}
        #tooltip .details {{
            color: #aaa;
            font-size: 11px;
        }}
        #help {{
            position: fixed;
            bottom: 10px;
            left: 10px;
            font-size: 11px;
            color: #555;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>{source_file}</h1>
        <div class="info">{metric_name} · {color_name} · {size}×{size} px · {len(word_data):,} words</div>
        <div id="controls">
            <button onclick="zoomIn()">+ Zoom</button>
            <button onclick="zoomOut()">− Zoom</button>
            <button onclick="resetView()">Reset</button>
            <span id="zoom-level">100%</span>
        </div>
    </div>
    <div id="container">
        <div id="canvas-wrapper">
            <img id="image" src="data:image/png;base64,{img_base64}" width="{size}" height="{size}">
        </div>
    </div>
    <div id="tooltip">
        <div class="word"></div>
        <div class="details"></div>
    </div>
    <div id="help">Scroll to zoom · Drag to pan · Hover for words</div>

    <script>
        const SIZE = {size};
        const WORDS = {words_json};
        const VALUES = {values_json};

        const container = document.getElementById('container');
        const wrapper = document.getElementById('canvas-wrapper');
        const image = document.getElementById('image');
        const tooltip = document.getElementById('tooltip');
        const zoomLabel = document.getElementById('zoom-level');

        let scale = 1;
        let panX = 0;
        let panY = 0;
        let isDragging = false;
        let dragStartX = 0;
        let dragStartY = 0;
        let dragStartPanX = 0;
        let dragStartPanY = 0;

        function updateTransform() {{
            wrapper.style.transform = `translate(${{panX}}px, ${{panY}}px) scale(${{scale}})`;
            zoomLabel.textContent = Math.round(scale * 100) + '%';
        }}

        function centerImage() {{
            const rect = container.getBoundingClientRect();
            panX = (rect.width - SIZE * scale) / 2;
            panY = (rect.height - SIZE * scale) / 2;
            updateTransform();
        }}

        function zoomIn() {{
            scale = Math.min(scale * 1.5, 200);
            centerImage();
        }}

        function zoomOut() {{
            scale = Math.max(scale / 1.5, 0.1);
            centerImage();
        }}

        function resetView() {{
            scale = 1;
            centerImage();
        }}

        // Mouse wheel zoom
        container.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const rect = container.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            // Position relative to image before zoom
            const imgX = (mouseX - panX) / scale;
            const imgY = (mouseY - panY) / scale;

            // Apply zoom
            const zoomFactor = e.deltaY < 0 ? 1.2 : 0.8;
            scale = Math.max(0.1, Math.min(200, scale * zoomFactor));

            // Adjust pan to keep mouse position stable
            panX = mouseX - imgX * scale;
            panY = mouseY - imgY * scale;

            updateTransform();
        }});

        // Pan with mouse drag
        container.addEventListener('mousedown', (e) => {{
            isDragging = true;
            container.classList.add('dragging');
            dragStartX = e.clientX;
            dragStartY = e.clientY;
            dragStartPanX = panX;
            dragStartPanY = panY;
        }});

        window.addEventListener('mousemove', (e) => {{
            if (isDragging) {{
                panX = dragStartPanX + (e.clientX - dragStartX);
                panY = dragStartPanY + (e.clientY - dragStartY);
                updateTransform();
            }}

            // Tooltip
            const rect = image.getBoundingClientRect();
            const x = Math.floor((e.clientX - rect.left) / (rect.width / SIZE));
            const y = Math.floor((e.clientY - rect.top) / (rect.height / SIZE));
            const idx = y * SIZE + x;

            if (x >= 0 && x < SIZE && y >= 0 && y < SIZE && idx < WORDS.length) {{
                const word = WORDS[idx];
                const value = VALUES[idx];
                tooltip.querySelector('.word').textContent = word;
                tooltip.querySelector('.details').textContent = `Position: ${{idx.toLocaleString()}} · Value: ${{value.toFixed(4)}}`;
                tooltip.style.display = 'block';
                tooltip.style.left = (e.clientX + 15) + 'px';
                tooltip.style.top = (e.clientY + 15) + 'px';
            }} else {{
                tooltip.style.display = 'none';
            }}
        }});

        window.addEventListener('mouseup', () => {{
            isDragging = false;
            container.classList.remove('dragging');
        }});

        // Initial centering
        window.addEventListener('load', centerImage);
        window.addEventListener('resize', centerImage);
    </script>
</body>
</html>'''

    Path(output_path).write_text(html)
    print(f"Saved: {output_path} (interactive HTML viewer)")


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
    parser.add_argument('--html', action='store_true',
                        help='Generate interactive HTML viewer with zoom and word tooltips')
    parser.add_argument('-i', '--ignore-case', action='store_true',
                        help='Treat words case-insensitively (lowercase all)')
    parser.add_argument('--ignore-punctuation', action='store_true',
                        help='Filter out punctuation tokens')
    parser.add_argument('--ignore-numbers', action='store_true',
                        help='Filter out numeric tokens')

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
    original_count = len(words)

    # Apply filters
    if args.ignore_case:
        words = [w.lower() for w in words]

    if args.ignore_punctuation:
        words = [w for w in words if w.isalnum()]

    if args.ignore_numbers:
        words = [w for w in words if not w.isnumeric()]

    print(f"Loaded {original_count} tokens from {args.file}", end='')
    if len(words) != original_count:
        print(f" ({len(words)} after filtering)")
    else:
        print()

    # Generate output filename
    if args.output:
        output_path = args.output
    else:
        output_path = input_path.stem + f"-{args.metric}.png"

    # Get metric and color functions
    metric_fn = METRICS[args.metric]
    color_fn = COLOR_MAPPERS[args.color]

    # Determine which words to pass (for bigram metrics, we need the bigram pairs)
    if args.metric in ('bigram-prob', 'bigram-diversity'):
        display_words = [f"{w1} → {w2}" for w1, w2 in nltk.bigrams(words)]
    else:
        display_words = words

    # Render
    values = metric_fn(words)
    img, size, word_data = render(values, color_fn, output_path, words=display_words)

    # Generate HTML viewer if requested
    if args.html and img:
        html_path = Path(output_path).stem + '.html'
        render_html(img, size, word_data, html_path, args.metric, args.color, input_path.name)


if __name__ == '__main__':
    main()
