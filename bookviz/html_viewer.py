"""Static HTML viewer generation."""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

from PIL import Image


def image_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def write_viewer(
    image: Image.Image,
    output_path: Path,
    *,
    title: str,
    subtitle: str,
    labels: list[str],
    values: list[float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        viewer_html(
            image,
            title=title,
            subtitle=subtitle,
            labels=labels,
            values=values,
        ),
        encoding="utf-8",
    )


def viewer_html(image: Image.Image, *, title: str, subtitle: str, labels: list[str], values: list[float]) -> str:
    labels_json = json.dumps(labels)
    values_json = json.dumps([round(value, 5) for value in values])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #181818; color: #f2f2f2; font-family: system-ui, sans-serif; overflow: hidden; }}
    header {{ position: fixed; inset: 0 0 auto 0; height: 52px; display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 8px 16px; background: rgba(0,0,0,.78); z-index: 2; }}
    h1 {{ margin: 0; font-size: 15px; font-weight: 650; }}
    .sub {{ color: #aaa; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .buttons {{ display: flex; align-items: center; gap: 8px; }}
    button {{ background: #303030; color: #fff; border: 1px solid #555; border-radius: 4px; padding: 5px 10px; cursor: pointer; }}
    #stage {{ position: fixed; inset: 52px 0 0 0; overflow: hidden; cursor: grab; }}
    #stage.dragging {{ cursor: grabbing; }}
    #wrapper {{ position: absolute; transform-origin: 0 0; }}
    img {{ display: block; image-rendering: pixelated; image-rendering: crisp-edges; }}
    #tip {{ position: fixed; display: none; pointer-events: none; z-index: 3; max-width: 320px; padding: 8px 10px; background: rgba(0,0,0,.9); border: 1px solid #555; border-radius: 4px; font-size: 12px; }}
    #tip strong {{ display: block; margin-bottom: 4px; font-size: 14px; }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>{html.escape(title)}</h1>
      <div class="sub">{html.escape(subtitle)}</div>
    </div>
    <div class="buttons">
      <button type="button" onclick="zoomBy(1.5)">Zoom In</button>
      <button type="button" onclick="zoomBy(1/1.5)">Zoom Out</button>
      <button type="button" onclick="resetView()">Reset</button>
      <span id="zoom">100%</span>
    </div>
  </header>
  <main id="stage">
    <div id="wrapper"><img id="image" src="{image_data_uri(image)}" width="{image.width}" height="{image.height}" alt=""></div>
  </main>
  <div id="tip"><strong></strong><span></span></div>
  <script>
    const SIZE = {image.width};
    const LABELS = {labels_json};
    const VALUES = {values_json};
    const stage = document.getElementById("stage");
    const wrapper = document.getElementById("wrapper");
    const image = document.getElementById("image");
    const tip = document.getElementById("tip");
    const zoom = document.getElementById("zoom");
    let scale = 1, panX = 0, panY = 0, dragging = false, startX = 0, startY = 0, startPanX = 0, startPanY = 0;
    function update() {{ wrapper.style.transform = `translate(${{panX}}px, ${{panY}}px) scale(${{scale}})`; zoom.textContent = Math.round(scale * 100) + "%"; }}
    function center() {{ const r = stage.getBoundingClientRect(); panX = (r.width - image.width * scale) / 2; panY = (r.height - image.height * scale) / 2; update(); }}
    function resetView() {{ scale = 1; center(); }}
    function zoomBy(factor) {{ scale = Math.max(.1, Math.min(200, scale * factor)); center(); }}
    stage.addEventListener("wheel", (event) => {{
      event.preventDefault();
      const rect = stage.getBoundingClientRect();
      const mx = event.clientX - rect.left, my = event.clientY - rect.top;
      const ix = (mx - panX) / scale, iy = (my - panY) / scale;
      scale = Math.max(.1, Math.min(200, scale * (event.deltaY < 0 ? 1.2 : .8)));
      panX = mx - ix * scale; panY = my - iy * scale; update();
    }});
    stage.addEventListener("mousedown", (event) => {{ dragging = true; stage.classList.add("dragging"); startX = event.clientX; startY = event.clientY; startPanX = panX; startPanY = panY; }});
    window.addEventListener("mouseup", () => {{ dragging = false; stage.classList.remove("dragging"); }});
    window.addEventListener("mousemove", (event) => {{
      if (dragging) {{ panX = startPanX + event.clientX - startX; panY = startPanY + event.clientY - startY; update(); }}
      const rect = image.getBoundingClientRect();
      const x = Math.floor((event.clientX - rect.left) / (rect.width / image.width));
      const y = Math.floor((event.clientY - rect.top) / (rect.height / image.height));
      const index = y * image.width + x;
      if (x >= 0 && y >= 0 && x < image.width && y < image.height && index < LABELS.length) {{
        tip.querySelector("strong").textContent = LABELS[index];
        tip.querySelector("span").textContent = `Position: ${{index.toLocaleString()}} · Value: ${{VALUES[index]}}`;
        tip.style.left = event.clientX + 14 + "px"; tip.style.top = event.clientY + 14 + "px"; tip.style.display = "block";
      }} else {{ tip.style.display = "none"; }}
    }});
    addEventListener("load", center); addEventListener("resize", center);
  </script>
</body>
</html>
"""

