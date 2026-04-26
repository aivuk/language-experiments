from pathlib import Path

from bookviz.gallery import generate_gallery


def test_gallery_keeps_token_metrics_at_word_level(tmp_path: Path):
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

    word_freq_html = (site / "sample-word-freq.html").read_text(encoding="utf-8")
    lexical_html = (site / "sample-lexical-diversity.html").read_text(encoding="utf-8")
    assert '"one"' in word_freq_html
    assert "tokens 0-1: one two" in lexical_html
