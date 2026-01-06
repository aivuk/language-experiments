# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Language-experiments is a Python project that converts literary texts into visual representations using linguistic analysis. It maps word and bigram frequency data to color values, creating images that represent textual patterns.

## Running the Code

```bash
# Generate visualization from a text file
python book_png.py <path_to_text_file>
# Output: <filename>-lenbig.png
```

No build system or test framework exists. Dependencies: NLTK, PIL/Pillow, Flask, SQLAlchemy.

## Architecture

### Core Visualization Engine (`book_png.py`)

Four visualization methods that convert tokenized text to PNG images:

1. **`blue_red_gradient_words`** - Maps word frequency to blue-red gradient using logarithmic scaling
2. **`random_colored_words`** - Assigns consistent random colors per unique word
3. **`conditional_probabilities_bigrams`** - Colors based on bigram transition probabilities (disabled)
4. **`possible_bigrams`** - Colors based on bigram diversity per word (currently active)

**Processing pipeline:** Text → `nltk.word_tokenize()` → frequency analysis → color mapping → square PNG output

Image dimensions are automatically calculated as √n × √n where n = number of words/bigrams.

### Web Application (`web/textnav.py`)

Skeleton Flask app with SQLAlchemy models (User, Book, Picture, PictureParams) using SQLite. No routes implemented - database schema only.

## Key Patterns

- Functional decomposition: each visualization method is self-contained
- NLTK for all NLP operations (tokenization, bigrams, frequency distributions)
- PIL for pixel-by-pixel image generation
- Square grid layout maps linear sequences to 2D images
- RGB channels encode different frequency metrics

## Code Notes

- Written for Python 2.x (uses deprecated patterns like `has_key()`, old Flask-SQLAlchemy imports)
- No error handling or input validation
- Active function in `book_png.py` is `possible_bigrams()` at the bottom of the file
