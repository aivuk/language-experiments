from bookviz.gutenberg import detect_title, strip_gutenberg_boilerplate


def test_strip_gutenberg_boilerplate():
    text = """Header
Title: Example Book

*** START OF THE PROJECT GUTENBERG EBOOK EXAMPLE BOOK ***
Body text.
*** END OF THE PROJECT GUTENBERG EBOOK EXAMPLE BOOK ***
Footer"""
    assert strip_gutenberg_boilerplate(text) == "Body text.\n"


def test_detect_title():
    assert detect_title("Project Gutenberg\nTitle: Moby Dick\n") == "Moby Dick"

