from pathlib import Path
import pandas as pd

import json


files_stem = [
    "all",
    "edge_curvature",
    "edge_density",
    "edge_length",
    "edge_ssim",
    "lab_color_stats",
    "lab_ssim",
    "line_angle"
]


def load_works(root):
    df = pd.read_csv(Path(root) / "data" / "work.csv")

    works_dict = {}

    for _, row in df.iterrows():
        i = int(row["id"])

        works_dict[i] = {
            "id": str(i),
            "artist": str(row["artist"]),
            "title": str(row["title"]),
            "title_jp": str(row["title_jp"]),
        }

    return works_dict


def load_map(path):
    df = pd.read_csv(path)

    map_dict = {}

    for _, row in df.iterrows():
        i = int(row["id"])

        map_dict[i] = {
            "id": str(i),
            "x": float(row["x"]),
            "y": float(row["y"]),
        }

    return map_dict


def load_maps(root):
    files = files_stem

    maps_dict = {}

    for f in files:
        path = Path(root) / "outputs" / "maps" / Path(f + ".csv")

        maps_dict[f] = load_map(path)

    return maps_dict


def generate_html(root):
    works_dict = load_works(root)

    works_json = json.dumps(works_dict, ensure_ascii = False)

    maps_dict = load_maps(root)

    maps_json = json.dumps(maps_dict, ensure_ascii = False)

    return f"""
<!DOCTYPE HTML>

<html lang="ja">
    <head>
        <meta charset="UTF-8">
        <title>Mapping Viewer</title>
        <style>

html, body {{
    margin: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
}}

canvas {{
    display: block;
}}

#tooltip {{
    position: fixed;
    display: none;
}}

        </style>
    </head>

    <body>
        <div id="tabs">
            <button data-map="all">all</button>
            <button data-map="edge_curvature">edge_curvature</button>
            <button data-map="edge_density">edge_density</button>
            <button data-map="edge_length">edge_length</button>
            <button data-map="edge_ssim">edge_ssim</button>
            <button data-map="lab_color_stats">lab_color_stats</button>
            <button data-map="lab_ssim">lab_ssim</button>
            <button data-map="line_angle">line_angle</button>
        </div>

        <canvas id="canvas"></canvas>

        <div id="tooltip"></div>

        <script>

const works = {works_json};
const maps = {maps_json};

const padding = 10;
const hoverRadius = 8;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const tabs = document.getElementById("tabs");

const buttons = document.querySelectorAll("#tabs button");

function resizeCanvas() {{
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight - tabs.offsetHeight
}}

function getTransform(points) {{
    let minX = points[0].x;
    let maxX = points[0].x;
    let minY = points[0].y;
    let maxY = points[0].y;

    for(const point of points) {{
        minX = Math.min(minX, point.x);
        maxX = Math.max(maxX, point.x);
        minY = Math.min(minY, point.y);
        maxY = Math.max(maxY, point.y);
    }}

    const scaleX = (canvas.width - padding * 2) / (maxX - minX);
    const scaleY = (canvas.height - padding * 2) / (maxY - minY);

    const scale = Math.min(scaleX, scaleY);

    const drawX = (maxX - minX) * scale;
    const drawY = (maxY - minY) * scale;

    const offsetX = (canvas.width - drawX) / 2;
    const offsetY = (canvas.height - drawY) / 2;

    return {{
        minX: minX,
        maxX: maxX,
        minY: minY,
        maxY: maxY,
        scale: scale,
        offsetX: offsetX,
        offsetY: offsetY
    }};
}}

function toCanvas(csvX, csvY, transform) {{
    const screenX = (csvX - transform.minX) * transform.scale + transform.offsetX;
    const screenY = (transform.maxY - csvY) * transform.scale + transform.offsetY;

    return {{
        x: screenX,
        y: screenY
    }}
}}

/* -------- 描画 -------- */

let currentMap = "all";

let renderedPoints = [];

function draw() {{
    renderedPoints = [];

    resizeCanvas();

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const points = Object.values(maps[currentMap]);
    const transform = getTransform(points);

    for (const point of points) {{
        const position = toCanvas(point.x, point.y, transform);

        ctx.beginPath();
        ctx.arc(position.x, position.y, 3, 0, 2 * Math.PI);
        ctx.fill();

        renderedPoints.push({{
            id: point.id,
            x: position.x,
            y: position.y
        }});
    }}
}}

draw();

buttons.forEach((button) => {{
    button.addEventListener("click", () => {{
        currentMap = button.dataset.map;

        draw();
    }});
}});

window.addEventListener("resize", () => {{
    draw();
}});

/* -------- マウス位置 -------- */

function findPoint(x, y) {{
    let foundPoint = null;
    let minD = Infinity;

    for (const point of renderedPoints) {{
        const distanceSquared = (point.x - x) ** 2 + (point.y - y) ** 2;

        if (minD > distanceSquared) {{
            minD = distanceSquared;
            foundPoint = point;
        }}
    }}

    if (minD < hoverRadius ** 2) {{
        return foundPoint;
    }}

    return null;
}}

canvas.addEventListener("mousemove", (event) => {{
    const rect = canvas.getBoundingClientRect();

    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    const point = findPoint(mouseX, mouseY);

    if (point != null) {{
        const w = works[point.id];
        document.getElementById("tooltip").innerText = `id: ${{w.id}}, artist: ${{w.artist}}, title: ${{w.title}}, title_jp: ${{w.title_jp}}`;
    }}
}});

        </script>
    </body>
</html>

"""

def main():
    root = Path(__file__).resolve().parent.parent

    html = generate_html(root)

    output_path = root / "outputs" / "viewer.html"

    output_path.write_text(html, encoding="utf-8")

    print("...SUCCESS")


if __name__ == "__main__":
    main()