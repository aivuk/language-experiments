"""Static client-side gallery generation."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

from .text import slugify


def generate_gallery(
    input_dir: Path,
    output_dir: Path,
    *,
    metrics: list[str],
    color: str,
    window_size: int | None,
    window_step: int | None,
    limit: int | None = None,
) -> list[Path]:
    books = sorted(input_dir.glob("*.txt"))
    if limit:
        books = books[:limit]
    if not books:
        raise ValueError(f"No .txt files found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    books_dir = output_dir / "books"
    books_dir.mkdir(exist_ok=True)
    manifest: list[dict[str, str]] = []
    generated: list[Path] = []
    for book in books:
        target = books_dir / f"{slugify(book.stem)}.txt"
        shutil.copyfile(book, target)
        manifest.append({"title": book.stem, "path": f"books/{target.name}"})
        generated.append(target)
    index_path = output_dir / "index.html"
    index_path.write_text(index_html(manifest, metrics, color, window_size, window_step), encoding="utf-8")
    generated.append(index_path)
    return generated


def index_html(
    books: list[dict[str, str]],
    metrics: list[str],
    color: str,
    window_size: int | None,
    window_step: int | None,
) -> str:
    books_json = json.dumps(books)
    metrics_json = json.dumps(metrics or ["word-freq", "lexical-diversity"])
    color_json = json.dumps(color)
    window_size_json = json.dumps(window_size or 200)
    window_step_json = json.dumps(window_step or window_size or 200)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Language Experiments Gallery</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; min-height: 100vh; background: #f4f1ea; color: #202020; font-family: system-ui, sans-serif; }}
    header {{ display: flex; align-items: center; justify-content: space-between; gap: 20px; padding: 16px 22px; border-bottom: 1px solid #d6d0c4; background: #fffaf0; }}
    h1 {{ margin: 0; font-size: 22px; letter-spacing: 0; }}
    .app {{ display: grid; grid-template-columns: 320px minmax(0, 1fr); min-height: calc(100vh - 66px); }}
    aside {{ padding: 18px; border-right: 1px solid #d6d0c4; background: #fbfaf7; }}
    main {{ position: relative; min-width: 0; display: grid; grid-template-rows: minmax(0, 1fr) auto; }}
    label {{ display: grid; gap: 6px; margin-bottom: 14px; font-size: 12px; font-weight: 700; color: #4f4b45; }}
    select, input {{ width: 100%; min-height: 34px; border: 1px solid #bdb7ad; border-radius: 5px; padding: 6px 8px; background: #fff; color: #202020; }}
    input[type="checkbox"] {{ width: auto; min-height: auto; }}
    input[type="range"] {{ padding: 0; }}
    .row {{ display: flex; gap: 10px; align-items: center; }}
    .row label {{ margin: 0; display: flex; align-items: center; gap: 8px; font-weight: 650; }}
    .meta {{ margin-top: 18px; padding-top: 16px; border-top: 1px solid #ddd7cc; font-size: 13px; line-height: 1.45; color: #5b564f; }}
    .stage {{ min-width: 0; overflow: auto; display: grid; place-items: center; padding: 24px; background: #191919; }}
    canvas {{ image-rendering: pixelated; image-rendering: crisp-edges; background: #000; cursor: crosshair; }}
    .details {{ min-height: 84px; padding: 12px 16px; border-top: 1px solid #d6d0c4; background: #fff; font-size: 13px; line-height: 1.4; overflow-wrap: anywhere; }}
    .details strong {{ display: block; margin-bottom: 5px; font-size: 15px; }}
    @media (max-width: 760px) {{
      .app {{ grid-template-columns: 1fr; }}
      aside {{ border-right: 0; border-bottom: 1px solid #d6d0c4; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Language Experiments Gallery</h1>
    <div id="status">Loading</div>
  </header>
  <div class="app">
    <aside>
      <label>Book
        <select id="book"></select>
      </label>
      <label>Metric
        <select id="metric"></select>
      </label>
      <label>Color
        <select id="color">
          <option value="red-blue">red-blue</option>
          <option value="blue-red">blue-red</option>
          <option value="heat">heat</option>
          <option value="grayscale">grayscale</option>
          <option value="green-purple">green-purple</option>
          <option value="rainbow">rainbow</option>
        </select>
      </label>
      <div class="row">
        <label><input id="perToken" type="checkbox" checked> word-level pixels</label>
      </div>
      <label>Window size <span id="windowSizeValue"></span>
        <input id="windowSize" type="range" min="25" max="1000" step="25">
      </label>
      <div class="row">
        <label><input id="lockStep" type="checkbox" checked> lock step to size</label>
      </div>
      <label>Window step <span id="windowStepValue"></span>
        <input id="windowStep" type="range" min="25" max="1000" step="25">
      </label>
      <div class="meta" id="meta"></div>
    </aside>
    <main>
      <div class="stage"><canvas id="canvas"></canvas></div>
      <div class="details" id="details"><strong>Hover over the image</strong><span></span></div>
    </main>
  </div>
  <script>
    const BOOKS = {books_json};
    const DEFAULT_METRICS = {metrics_json};
    const DEFAULT_COLOR = {color_json};
    const DEFAULT_WINDOW_SIZE = {window_size_json};
    const DEFAULT_WINDOW_STEP = {window_step_json};
    const TOKEN_METRICS = ["word-freq", "word-freq-linear", "bigram-prob", "bigram-diversity", "word-length", "word-position", "unique-word"];
    const WINDOW_METRICS = ["avg-word-length", "lexical-diversity", "punctuation-density", "repetition-density", "sentence-length"];
    const FIXED_SCALE_METRICS = ["lexical-diversity", "punctuation-density", "repetition-density", "word-freq-linear", "bigram-prob", "word-length", "word-position", "unique-word"];
    const ALL_METRICS = [...TOKEN_METRICS, ...WINDOW_METRICS];
    const tokenRe = /[\\p{{L}}\\p{{N}}_]+|[^\\p{{L}}\\p{{N}}_\\s]/gu;
    const state = {{ text: "", tokens: [], values: [], labels: [], size: 0 }};
    const els = {{
      book: document.getElementById("book"),
      metric: document.getElementById("metric"),
      color: document.getElementById("color"),
      perToken: document.getElementById("perToken"),
      lockStep: document.getElementById("lockStep"),
      windowSize: document.getElementById("windowSize"),
      windowStep: document.getElementById("windowStep"),
      windowSizeValue: document.getElementById("windowSizeValue"),
      windowStepValue: document.getElementById("windowStepValue"),
      canvas: document.getElementById("canvas"),
      details: document.getElementById("details"),
      meta: document.getElementById("meta"),
      status: document.getElementById("status"),
    }};

    function init() {{
      BOOKS.forEach((book, index) => els.book.add(new Option(book.title, index)));
      ALL_METRICS.forEach(metric => els.metric.add(new Option(metric, metric)));
      els.metric.value = DEFAULT_METRICS.includes("lexical-diversity") ? "lexical-diversity" : DEFAULT_METRICS[0] || "word-freq";
      els.color.value = DEFAULT_COLOR;
      els.windowSize.value = DEFAULT_WINDOW_SIZE;
      els.windowStep.value = DEFAULT_WINDOW_STEP;
      els.lockStep.checked = DEFAULT_WINDOW_SIZE === DEFAULT_WINDOW_STEP;
      els.book.addEventListener("input", loadBook);
      for (const el of [els.metric, els.color, els.perToken, els.lockStep, els.windowSize, els.windowStep]) {{
        el.addEventListener("input", () => {{
          if (el === els.windowSize && els.lockStep.checked) els.windowStep.value = els.windowSize.value;
          if (el === els.lockStep && els.lockStep.checked) els.windowStep.value = els.windowSize.value;
          render();
        }});
      }}
      els.canvas.addEventListener("mousemove", showDetails);
      els.canvas.addEventListener("mouseleave", () => setDetails("Hover over the image", ""));
      loadBook();
    }}

    async function loadBook() {{
      const book = BOOKS[Number(els.book.value)];
      els.status.textContent = "Loading " + book.title;
      const response = await fetch(book.path);
      state.text = await response.text();
      state.tokens = tokenize(state.text);
      els.status.textContent = book.title;
      render();
    }}

    function tokenize(text) {{
      return text.match(tokenRe) || [];
    }}

    function render() {{
      const metric = els.metric.value;
      const canUsePerToken = TOKEN_METRICS.includes(metric);
      els.perToken.disabled = !canUsePerToken;
      const useWindows = !canUsePerToken || !els.perToken.checked;
      if (els.lockStep.checked) els.windowStep.value = els.windowSize.value;
      els.windowSize.disabled = !useWindows;
      els.lockStep.disabled = !useWindows;
      els.windowStep.disabled = !useWindows || els.lockStep.checked;
      els.windowSizeValue.textContent = els.windowSize.value;
      els.windowStepValue.textContent = els.windowStep.value;
      const result = useWindows
        ? windowValues(state.tokens, metric, Number(els.windowSize.value), Number(els.windowStep.value))
        : tokenValues(state.tokens, metric);
      state.values = normalize(result.values, scaleDomain(metric));
      state.labels = result.labels;
      draw(state.values);
      els.meta.innerHTML = [
        `<strong>${{state.tokens.length.toLocaleString()}}</strong> tokens`,
        `<strong>${{state.values.length.toLocaleString()}}</strong> pixels`,
        `raw range ${{formatRange(valueRange(result.values))}}`,
        scaleDomain(metric) ? "color scale 0-1" : "color scale current min-max",
        useWindows ? `windowed, size ${{els.windowSize.value}}, step ${{els.windowStep.value}}` : "word-level pixels",
      ].join("<br>");
      setDetails("Hover over the image", "");
    }}

    function tokenValues(tokens, metric) {{
      if (metric === "word-freq" || metric === "word-freq-linear") {{
        const counts = counter(tokens);
        const max = maxOf(counts.values());
        const values = tokens.map(token => metric === "word-freq" ? Math.log(counts.get(token)) / (Math.log(max) || 1) : counts.get(token) / max);
        return {{ values, labels: tokens }};
      }}
      if (metric === "word-length") {{
        const max = Math.max(maxOf(tokens.map(token => token.length)), 1);
        return {{ values: tokens.map(token => token.length / max), labels: tokens }};
      }}
      if (metric === "word-position") {{
        return {{ values: tokens.map((_token, index) => index / Math.max(tokens.length, 1)), labels: tokens }};
      }}
      if (metric === "unique-word") {{
        const ids = new Map();
        tokens.forEach(token => {{ if (!ids.has(token)) ids.set(token, ids.size); }});
        const max = Math.max(ids.size - 1, 1);
        return {{ values: tokens.map(token => ids.get(token) / max), labels: tokens }};
      }}
      if (metric === "bigram-prob" || metric === "bigram-diversity") {{
        const pairs = tokens.slice(0, -1).map((token, index) => [token, tokens[index + 1]]);
        const followers = new Map();
        for (const [first, second] of pairs) {{
          if (!followers.has(first)) followers.set(first, new Map());
          followers.get(first).set(second, (followers.get(first).get(second) || 0) + 1);
        }}
        const maxDiversity = Math.max(maxOf([...followers.values()].map(map => map.size)), 1);
        const values = pairs.map(([first, second]) => {{
          const map = followers.get(first);
          if (metric === "bigram-prob") return map.get(second) / [...map.values()].reduce((a, b) => a + b, 0);
          return map.size / maxDiversity;
        }});
        return {{ values, labels: pairs.map(pair => pair.join(" -> ")) }};
      }}
      return windowValues(tokens, metric, Number(els.windowSize.value), Number(els.windowStep.value));
    }}

    function windowValues(tokens, metric, size, step) {{
      const values = [];
      const labels = [];
      for (let start = 0; start < tokens.length; start += step) {{
        const chunk = tokens.slice(start, start + size);
        if (!chunk.length) continue;
        if (chunk.length < size * 0.5) continue;
        labels.push(windowLabel(chunk, start));
        if (WINDOW_METRICS.includes(metric)) values.push(windowMetric(chunk, metric));
        else {{
          const inner = tokenValues(chunk, metric).values;
          values.push(inner.reduce((sum, value) => sum + value, 0) / Math.max(inner.length, 1));
        }}
      }}
      return {{ values, labels }};
    }}

    function windowMetric(tokens, metric) {{
      const words = tokens.filter(token => !isPunctuation(token));
      const lower = words.map(token => token.toLocaleLowerCase());
      if (metric === "avg-word-length") return words.reduce((sum, token) => sum + token.length, 0) / Math.max(words.length, 1);
      if (metric === "lexical-diversity") return new Set(lower).size / Math.max(lower.length, 1);
      if (metric === "punctuation-density") return tokens.filter(isPunctuation).length / Math.max(tokens.length, 1);
      if (metric === "repetition-density") return 1 - new Set(lower).size / Math.max(lower.length, 1);
      if (metric === "sentence-length") return words.length / Math.max(tokens.filter(token => [".", "?", "!"].includes(token)).length, 1);
      return 0;
    }}

    function windowLabel(tokens, start) {{
      const words = tokens.filter(token => !isPunctuation(token));
      const excerpt = words.slice(0, 16).join(" ") + (words.length > 16 ? " ..." : "");
      return `tokens ${{start}}-${{start + tokens.length - 1}}: ${{excerpt}}`;
    }}

    function counter(tokens) {{
      const counts = new Map();
      tokens.forEach(token => counts.set(token, (counts.get(token) || 0) + 1));
      return counts;
    }}

    function isPunctuation(token) {{
      return !/[\\p{{L}}\\p{{N}}_]/u.test(token);
    }}

    function normalize(values, domain = null) {{
      if (!values.length) return [];
      const range = domain || valueRange(values);
      const min = range[0];
      const max = range[1];
      if (min === max) return values.map(() => 0);
      return values.map(value => (value - min) / (max - min));
    }}

    function scaleDomain(metric) {{
      return FIXED_SCALE_METRICS.includes(metric) ? [0, 1] : null;
    }}

    function valueRange(values) {{
      if (!values.length) return [0, 0];
      let min = values[0];
      let max = values[0];
      for (const value of values) {{
        if (value < min) min = value;
        if (value > max) max = value;
      }}
      return [min, max];
    }}

    function formatRange(range) {{
      return `${{range[0].toFixed(4)}}-${{range[1].toFixed(4)}}`;
    }}

    function maxOf(values) {{
      let max = 0;
      for (const value of values) if (value > max) max = value;
      return max;
    }}

    function colorFor(value) {{
      const v = Math.max(0, Math.min(1, value));
      const c = els.color.value;
      if (c === "blue-red") return [255 - v * 255, 0, v * 255];
      if (c === "heat") {{
        if (v < .33) return [v * 3 * 255, 0, 0];
        if (v < .66) return [255, (v - .33) * 3 * 255, 0];
        return [255, 255, (v - .66) * 3 * 255];
      }}
      if (c === "grayscale") return [v * 255, v * 255, v * 255];
      if (c === "green-purple") return [v * 255, 255 - v * 255, v * 255];
      if (c === "rainbow") return hsvToRgb(v, 1, 1);
      return [v * 255, 0, 255 - v * 255];
    }}

    function hsvToRgb(h, s, v) {{
      const i = Math.floor(h * 6);
      const f = h * 6 - i;
      const p = v * (1 - s);
      const q = v * (1 - f * s);
      const t = v * (1 - (1 - f) * s);
      const variants = [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]];
      return variants[i % 6].map(value => value * 255);
    }}

    function draw(values) {{
      const size = Math.ceil(Math.sqrt(values.length || 1));
      state.size = size;
      els.canvas.width = size;
      els.canvas.height = size;
      const displaySize = Math.min(900, Math.max(360, Math.floor(Math.min(window.innerWidth - 380, window.innerHeight - 160))));
      els.canvas.style.width = displaySize + "px";
      els.canvas.style.height = displaySize + "px";
      const ctx = els.canvas.getContext("2d");
      const image = ctx.createImageData(size, size);
      values.forEach((value, index) => {{
        const [r, g, b] = colorFor(value);
        image.data[index * 4] = r;
        image.data[index * 4 + 1] = g;
        image.data[index * 4 + 2] = b;
        image.data[index * 4 + 3] = 255;
      }});
      ctx.putImageData(image, 0, 0);
    }}

    function showDetails(event) {{
      const rect = els.canvas.getBoundingClientRect();
      const x = Math.floor((event.clientX - rect.left) / (rect.width / state.size));
      const y = Math.floor((event.clientY - rect.top) / (rect.height / state.size));
      const index = y * state.size + x;
      if (index < 0 || index >= state.labels.length) return setDetails("Hover over the image", "");
      setDetails(state.labels[index], `Position: ${{index.toLocaleString()}} · Value: ${{state.values[index].toFixed(4)}}`);
    }}

    function setDetails(title, text) {{
      els.details.querySelector("strong").textContent = title;
      els.details.querySelector("span").textContent = text;
    }}

    init();
  </script>
</body>
</html>
"""
