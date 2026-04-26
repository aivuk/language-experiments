from bookviz.metrics import metric_values, normalize
from bookviz.text import tokenize, windows


def test_tokenize_keeps_words_and_punctuation():
    assert tokenize("Hello, world!") == ["Hello", ",", "world", "!"]


def test_word_frequency_values_match_token_count():
    values, labels = metric_values(["a", "b", "a"], "word-freq")
    assert len(values) == 3
    assert labels == ["a", "b", "a"]
    assert values[0] == values[2]
    assert values[0] > values[1]


def test_windows_use_size_and_step():
    assert windows(["a", "b", "c", "d"], 3, 2) == [["a", "b", "c"], ["c", "d"]]


def test_window_metric_values():
    values, labels = metric_values(["a", "b", "a", ".", "c"], "lexical-diversity", window_size=4)
    assert values == [2 / 3, 1.0]
    assert len(labels) == 2
    assert labels[0] == "tokens 0-3: a b a"


def test_normalize_with_shared_domain():
    assert normalize([10, 15, 20], domain=(10, 20)) == [0.0, 0.5, 1.0]
