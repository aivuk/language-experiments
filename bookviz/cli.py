"""Command line interface for language-experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .colors import COLOR_MAPPERS
from .gallery import generate_gallery
from .gutenberg import fetch_gutenberg
from .metrics import ALL_METRICS, TOKEN_METRICS, WINDOW_METRICS, normalize
from .pipeline import RenderOptions, prepare_values, render_book
from .render import render_comparison
from .text import slugify

COMMANDS = {"render", "compare", "gutenberg", "gallery", "list"}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = ["--help"]
    if argv[0] == "--list":
        argv = ["list"]
    elif argv[0] not in COMMANDS and not argv[0].startswith("-"):
        argv = ["render", *argv]
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize books as linguistic fingerprints.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render", help="Render one book to PNG and optional HTML.")
    render.add_argument("file", type=Path)
    add_render_options(render)
    render.add_argument("-o", "--output", type=Path)
    render.add_argument("--html", action="store_true", help="Generate an interactive HTML viewer.")
    render.set_defaults(func=cmd_render)

    compare = subparsers.add_parser("compare", help="Render books with shared normalization for comparison.")
    compare.add_argument("files", nargs="+", type=Path)
    add_render_options(compare)
    compare.add_argument("-o", "--output", type=Path, default=Path("outputs/comparison.png"))
    compare.add_argument("--html", action="store_true", help="Also create individual HTML viewers.")
    compare.set_defaults(func=cmd_compare)

    gutenberg = subparsers.add_parser("gutenberg", help="Download and cache a Project Gutenberg text.")
    gutenberg.add_argument("id", type=int)
    gutenberg.add_argument("--title")
    gutenberg.add_argument("-o", "--output-dir", type=Path, default=Path("books/gutenberg"))
    gutenberg.add_argument("--force", action="store_true")
    gutenberg.set_defaults(func=cmd_gutenberg)

    gallery = subparsers.add_parser("gallery", help="Generate the static client-side gallery.")
    gallery.add_argument("--input", type=Path, default=Path("books"))
    gallery.add_argument("--output", type=Path, default=Path("site"))
    gallery.add_argument("--metrics", nargs="+", default=["word-freq", "lexical-diversity"])
    gallery.add_argument("-c", "--color", choices=COLOR_MAPPERS.keys(), default="red-blue")
    gallery.add_argument("--window-size", type=int, default=200)
    gallery.add_argument("--window-step", type=int)
    gallery.add_argument("--limit", type=int)
    gallery.set_defaults(func=cmd_gallery)

    list_cmd = subparsers.add_parser("list", help="List metrics and color maps.")
    list_cmd.set_defaults(func=cmd_list)
    return parser


def add_render_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-m", "--metric", choices=ALL_METRICS.keys(), default="word-freq")
    parser.add_argument("-c", "--color", choices=COLOR_MAPPERS.keys(), default="red-blue")
    parser.add_argument("--window-size", type=int, help="Aggregate metric over sliding windows.")
    parser.add_argument("--window-step", type=int, help="Token step between sliding windows.")
    parser.add_argument("-i", "--ignore-case", action="store_true")
    parser.add_argument("--ignore-punctuation", action="store_true")
    parser.add_argument("--ignore-numbers", action="store_true")


def options_from_args(args: argparse.Namespace) -> RenderOptions:
    return RenderOptions(
        metric=args.metric,
        color=args.color,
        window_size=args.window_size,
        window_step=args.window_step,
        ignore_case=args.ignore_case,
        ignore_punctuation=args.ignore_punctuation,
        ignore_numbers=args.ignore_numbers,
    )


def cmd_render(args: argparse.Namespace) -> None:
    output = args.output or Path(f"{args.file.stem}-{args.metric}.png")
    result = render_book(args.file, output, options_from_args(args), html=args.html)
    print(f"Saved {result.output} ({result.size}x{result.size}, {len(result.values)} values)")
    if args.html:
        print(f"Saved {result.output.with_suffix('.html')}")


def cmd_compare(args: argparse.Namespace) -> None:
    options = options_from_args(args)
    prepared = []
    all_values: list[float] = []
    for source in args.files:
        raw_values, labels, token_count = prepare_values(source, options)
        prepared.append((source, raw_values, labels, token_count))
        all_values.extend(raw_values)
    domain = (min(all_values), max(all_values)) if all_values else (0.0, 1.0)
    rendered = []
    comparison_dir = args.output.parent / f"{args.output.stem}-items"
    for source, raw_values, labels, _token_count in prepared:
        values = normalize(raw_values, domain=domain)
        item_output = comparison_dir / f"{slugify(source.stem)}-{options.metric}.png"
        from .render import render_values

        image, _size, _label_values = render_values(values, item_output, color=options.color, labels=labels)
        rendered.append((source.stem, image))
        if args.html:
            html_options = RenderOptions(**{**options.__dict__, "domain": domain})
            render_book(source, item_output, html_options, html=True)
    render_comparison(rendered, args.output)
    print(f"Saved {args.output} with shared normalization domain {domain[0]:.4f}..{domain[1]:.4f}")


def cmd_gutenberg(args: argparse.Namespace) -> None:
    path = fetch_gutenberg(args.id, args.output_dir, title=args.title, force=args.force)
    print(f"Saved {path}")


def cmd_gallery(args: argparse.Namespace) -> None:
    generated = generate_gallery(
        args.input,
        args.output,
        metrics=args.metrics,
        color=args.color,
        window_size=args.window_size,
        window_step=args.window_step,
        limit=args.limit,
    )
    print(f"Generated {len(generated)} files in {args.output}")


def cmd_list(_args: argparse.Namespace) -> None:
    print("Token metrics:")
    for name in TOKEN_METRICS:
        print(f"  {name}")
    print("\nWindow metrics:")
    for name in WINDOW_METRICS:
        print(f"  {name}")
    print("\nColor maps:")
    for name in COLOR_MAPPERS:
        print(f"  {name}")


if __name__ == "__main__":
    main()
