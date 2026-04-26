# Language Experiments

Language Experiments turns books into visual fingerprints. It tokenizes a text,
computes a linguistic metric, maps the metric to colors, and writes PNG and
HTML viewers where each pixel represents a token or a sliding window of tokens.

## Setup

```bash
uv sync
```

## Commands

List available metrics and color maps:

```bash
uv run bookviz list
```

Render one book:

```bash
uv run bookviz render books/dubliners.txt --metric word-freq --color heat --html
```

The legacy form still works:

```bash
uv run python book_png.py books/dubliners.txt --metric word-freq --html
```

Render a sliding-window view:

```bash
uv run bookviz render books/ulysses.txt \
  --metric lexical-diversity \
  --window-size 200 \
  --window-step 50 \
  --html
```

Compare books with shared color normalization:

```bash
uv run bookviz compare \
  books/dubliners.txt books/ulysses.txt books/moby-dick.txt \
  --metric lexical-diversity \
  --window-size 200 \
  --output outputs/comparison.png \
  --html
```

Download a book from Project Gutenberg:

```bash
uv run bookviz gutenberg 2701 --title moby-dick
```

Generate the GitHub Pages gallery:

```bash
uv run bookviz gallery \
  --input books \
  --output site \
  --metrics word-freq lexical-diversity bigram-diversity \
  --window-size 200
```

The gallery is a static browser app. It publishes the book text files and lets
the browser compute metrics, change window size, change window step, switch
books, and redraw the visualization without regenerating images in CI.

## Metrics

Token metrics:

- `word-freq`
- `word-freq-linear`
- `bigram-prob`
- `bigram-diversity`
- `word-length`
- `word-position`
- `unique-word`

Window metrics:

- `avg-word-length`
- `lexical-diversity`
- `punctuation-density`
- `repetition-density`
- `sentence-length`

Token metrics can also be used with `--window-size`; their values are averaged
inside each window.

## GitHub Pages

The workflow at `.github/workflows/pages.yml` builds the client-side gallery
with `uv` and publishes the generated `site/` directory to GitHub Pages. Enable
Pages in the repository settings and choose GitHub Actions as the source.
