"""Metric functions for token-level and sliding-window visualizations."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable

from .text import is_punctuation, windows

MetricResult = tuple[list[float], list[str]]
MetricFn = Callable[[list[str]], MetricResult]


def word_frequency(tokens: list[str]) -> MetricResult:
    freq = Counter(tokens)
    if not tokens:
        return [], []
    max_log = math.log(max(freq.values())) or 1.0
    return [math.log(freq[token]) / max_log for token in tokens], tokens


def word_frequency_linear(tokens: list[str]) -> MetricResult:
    freq = Counter(tokens)
    if not tokens:
        return [], []
    max_freq = max(freq.values()) or 1
    return [freq[token] / max_freq for token in tokens], tokens


def bigram_probability(tokens: list[str]) -> MetricResult:
    pairs = list(zip(tokens, tokens[1:]))
    if not pairs:
        return [], []
    transitions: dict[str, Counter[str]] = defaultdict(Counter)
    for first, second in pairs:
        transitions[first][second] += 1
    values = [transitions[first][second] / sum(transitions[first].values()) for first, second in pairs]
    labels = [f"{first} -> {second}" for first, second in pairs]
    return values, labels


def bigram_diversity(tokens: list[str]) -> MetricResult:
    pairs = list(zip(tokens, tokens[1:]))
    if not pairs:
        return [], []
    followers: dict[str, set[str]] = defaultdict(set)
    for first, second in pairs:
        followers[first].add(second)
    max_count = max(len(values) for values in followers.values()) or 1
    values = [len(followers[first]) / max_count for first, _second in pairs]
    labels = [f"{first} -> {second}" for first, second in pairs]
    return values, labels


def word_length(tokens: list[str]) -> MetricResult:
    if not tokens:
        return [], []
    max_len = max(len(token) for token in tokens) or 1
    return [len(token) / max_len for token in tokens], tokens


def word_position(tokens: list[str]) -> MetricResult:
    total = len(tokens) or 1
    return [index / total for index, _token in enumerate(tokens)], tokens


def unique_word_id(tokens: list[str]) -> MetricResult:
    ids: dict[str, int] = {}
    for token in tokens:
        ids.setdefault(token, len(ids))
    max_id = max(len(ids) - 1, 1)
    return [ids[token] / max_id for token in tokens], tokens


def average_word_length(tokens: list[str]) -> float:
    words = [token for token in tokens if not is_punctuation(token)]
    return sum(len(token) for token in words) / len(words) if words else 0.0


def lexical_diversity(tokens: list[str]) -> float:
    words = [token.lower() for token in tokens if not is_punctuation(token)]
    return len(set(words)) / len(words) if words else 0.0


def punctuation_density(tokens: list[str]) -> float:
    return sum(1 for token in tokens if is_punctuation(token)) / len(tokens) if tokens else 0.0


def repetition_density(tokens: list[str]) -> float:
    words = [token.lower() for token in tokens if not is_punctuation(token)]
    return 1.0 - (len(set(words)) / len(words)) if words else 0.0


def sentence_length_estimate(tokens: list[str]) -> float:
    words = [token for token in tokens if not is_punctuation(token)]
    sentence_breaks = sum(1 for token in tokens if token in {".", "?", "!"}) or 1
    return len(words) / sentence_breaks


WINDOW_METRICS: dict[str, Callable[[list[str]], float]] = {
    "avg-word-length": average_word_length,
    "lexical-diversity": lexical_diversity,
    "punctuation-density": punctuation_density,
    "repetition-density": repetition_density,
    "sentence-length": sentence_length_estimate,
}

TOKEN_METRICS: dict[str, MetricFn] = {
    "word-freq": word_frequency,
    "word-freq-linear": word_frequency_linear,
    "bigram-prob": bigram_probability,
    "bigram-diversity": bigram_diversity,
    "word-length": word_length,
    "word-position": word_position,
    "unique-word": unique_word_id,
}

ALL_METRICS = {**TOKEN_METRICS, **WINDOW_METRICS}


def metric_values(tokens: list[str], metric: str, *, window_size: int | None = None, window_step: int | None = None) -> MetricResult:
    if window_size:
        return window_metric_values(tokens, metric, window_size=window_size, window_step=window_step)
    if metric not in TOKEN_METRICS:
        raise ValueError(f"{metric!r} requires --window-size")
    return TOKEN_METRICS[metric](tokens)


def window_metric_values(tokens: list[str], metric: str, *, window_size: int, window_step: int | None = None) -> MetricResult:
    chunks = windows(tokens, window_size, window_step)
    labels = [window_label(chunk, index, window_step or window_size) for index, chunk in enumerate(chunks)]
    if metric in WINDOW_METRICS:
        return [WINDOW_METRICS[metric](chunk) for chunk in chunks], labels
    if metric in TOKEN_METRICS:
        values = []
        for chunk in chunks:
            chunk_values, _labels = TOKEN_METRICS[metric](chunk)
            values.append(sum(chunk_values) / len(chunk_values) if chunk_values else 0.0)
        return values, labels
    raise ValueError(f"Unknown metric: {metric}")


def window_label(chunk: list[str], index: int, step: int) -> str:
    start = index * step
    end = start + len(chunk) - 1
    words = [token for token in chunk if not is_punctuation(token)]
    excerpt = " ".join(words[:14])
    if len(words) > 14:
        excerpt += " ..."
    return f"tokens {start}-{end}: {excerpt}" if excerpt else f"tokens {start}-{end}"


def normalize(values: list[float], *, domain: tuple[float, float] | None = None) -> list[float]:
    if not values:
        return []
    low, high = domain if domain else (min(values), max(values))
    if high == low:
        return [0.0 for _value in values]
    return [(value - low) / (high - low) for value in values]
