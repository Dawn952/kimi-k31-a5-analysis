"""Rasterize the same exported graph geometry for downloadable PNG previews."""
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT = Path("C:/Windows/Fonts/msyh.ttc")
if not FONT.exists():
    FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
SCALE = 1.5


def render(family, graph):
    w, h = graph["width"], graph["height"]
    image = Image.new("RGB", (int(w * SCALE), int(h * SCALE)), "#09151e")
    draw = ImageDraw.Draw(image)

    def rect(box, fill, outline, width=1):
        x, y, width_box, height_box = box
        draw.rounded_rectangle(tuple(int(v * SCALE) for v in (x, y, x + width_box, y + height_box)),
                               radius=int(8 * SCALE), fill=fill, outline=outline, width=width)

    def text(x, y, value, size=12, fill="#b6cbd5"):
        draw.text((int(x * SCALE), int(y * SCALE)), value,
                  font=ImageFont.truetype(str(FONT), int(size * SCALE)), fill=fill)

    for stage in graph["stages"]:
        rect([stage[k] for k in ("x", "y", "w", "h")], "#101e29", "#2b4253")
        text(stage["x"] + 14, 35, stage["title"], 13)
    for group in graph["parallelGroups"]:
        for part in group["parts"]:
            nodes = [node for node in graph["nodes"] if node["id"] in part]
            x, y = min(n["x"] for n in nodes) - 9, min(n["y"] for n in nodes) - 9
            right, bottom = max(n["x"] + n["w"] for n in nodes) + 9, max(n["y"] + n["h"] for n in nodes) + 9
            rect((x, y, right - x, bottom - y), None, group["color"], 2)
            text(x + 5, y - 23, group["title"], 11, group["color"])
    for edge in graph["edges"]:
        points = [(int(point["x"] * SCALE), int(point["y"] * SCALE)) for point in edge["points"]]
        color = "#9c91bc" if edge.get("order") else "#6f95aa"
        if edge.get("order"):
            for a, b in zip(points, points[1:]):
                distance = math.dist(a, b)
                for start in range(0, int(distance), 12):
                    if distance:
                        t, u = start / distance, min(1, (start + 7) / distance)
                        draw.line([(a[0] + (b[0] - a[0]) * v, a[1] + (b[1] - a[1]) * v) for v in (t, u)], fill=color, width=2)
        else:
            draw.line(points, fill=color, width=2)
        a, b = points[-2:]
        angle = math.atan2(b[1] - a[1], b[0] - a[0])
        draw.polygon([b, (b[0] - 9 * math.cos(angle - .5), b[1] - 9 * math.sin(angle - .5)),
                      (b[0] - 9 * math.cos(angle + .5), b[1] - 9 * math.sin(angle + .5))], fill=color)
    for node in graph["nodes"]:
        fill, outline = ("#163948", "#58cfdd") if node["kind"] == "comm" else ("#172b29", "#8bbca2") if node["kind"] == "cache" else ("#1b3040", "#4e7185")
        rect([node[k] for k in ("x", "y", "w", "h")], fill, outline)
        text(node["x"] + 14, node["y"] + 8, node["title"], 14, "#edf4f7")
        for i, line in enumerate(node["lines"]):
            text(node["x"] + 14, node["y"] + 33 + 19 * i, line)
        text(node["x"] + 14, node["y"] + 82, node["status"], 10, "#8ba8b9")
    text(36, h - 30, "A5 localTP · 3634a7eae948 · 首例相对时间 / 依赖布局非时间比例", 12)
    image.save(ROOT / f"docs/images/a5-{family}-localtp.png")


if __name__ == "__main__":
    for family, graph in json.loads((ROOT / "data/localtp-profile-graphs.json").read_text(encoding="utf-8")).items():
        render(family, graph)
    print("Rendered six PNGs from the exported dependency geometry")
