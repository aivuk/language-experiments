from pathlib import Path

from bookviz.gallery import generate_gallery


def test_gallery_generates_client_side_explorer(tmp_path: Path):
    books = tmp_path / "books"
    site = tmp_path / "site"
    books.mkdir()
    (books / "sample.txt").write_text("one two one three", encoding="utf-8")

    generate_gallery(
        books,
        site,
        metrics=["word-freq", "lexical-diversity"],
        color="red-blue",
        window_size=2,
        window_step=None,
    )

    index_html = (site / "index.html").read_text(encoding="utf-8")
    copied_book = (site / "books" / "sample.txt").read_text(encoding="utf-8")
    assert copied_book == "one two one three"
    assert "const BOOKS" in index_html
    assert "windowSize" in index_html
    assert "lockStep" in index_html
    assert "FIXED_SCALE_METRICS" in index_html
    assert "chunk.length < size * 0.5" in index_html
    assert "raw range" in index_html
    assert "lexical-diversity" in index_html
