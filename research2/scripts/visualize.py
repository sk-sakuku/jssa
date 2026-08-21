from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


FIGSIZE = (8, 8)
POINT_SIZE = 10
AXIS_MARGIN = 0.2

GROUP_COLORS = {
    "ai_mon": "#ff6666",
    "ai_kan": "#ff3333",
    "ai_mal": "#d60000",
    "ai_alb": "#990000",
    "ai_vas": "#550000",
    "mon": "#4dd2ff",
    "kan": "#0088ff",
    "mal": "#0055c4",
    "alb": "#003380",
    "vas": "#001144",
    "compare": "#00FF40"
}

GROUP_NAMES = {
    "ai_mon": "AI - Mondrian",
    "ai_kan": "AI - Kandinsky",
    "ai_mal": "AI - Malevich",
    "ai_alb": "AI - Albers",
    "ai_vas": "AI - Vasarely",
    "mon": "Mondrian",
    "kan": "Kandinsky",
    "mal": "Malevich",
    "alb": "Albers",
    "vas": "Vasarely",
    "compare": "For Comparison"
}

def visualize(csv_path, meta, output_path):
    print("Saving...")

    df = pd.read_csv(csv_path)

    df = df.merge(
        meta[["id", "group"]],
        on="id",
        how="left"
    )

    fig, ax = plt.subplots(
        figsize = FIGSIZE
    )

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    for group, color in GROUP_COLORS.items():
        subset = df[
            df["group"] == group
        ]

        ax.scatter(
            subset["x"],
            subset["y"],
            color=color,
            s=POINT_SIZE
        )

    x_range = df["x"].max() - df["x"].min()
    y_range = df["y"].max() - df["y"].min()

    data_range = max(x_range, y_range)

    x_center = (df["x"].min() + df["x"].max()) / 2
    y_center = (df["y"].min() + df["y"].max()) / 2

    half_range = data_range / 2

    half_range *= 1 + AXIS_MARGIN

    ax.set_xlim(
        x_center - half_range,
        x_center + half_range
    )

    ax.set_ylim(
        y_center - half_range,
        y_center + half_range
    )

    # 凡例を表示
    legend_handles = []

    for group, color in GROUP_COLORS.items():
        handle = plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor=color,
            label=GROUP_NAMES[group]
        )

        legend_handles.append(handle)

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=False,
        labelcolor="white",
        fontsize=9
    )

    # タイトルを表示
    fig.text(
        0.5,
        0.98,
        csv_path.stem + ".jpg",
        ha="center",
        va="top",
        color="white",
        fontsize=14
    )

    ax.set_axis_off()

    plt.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=1
    )

    plt.savefig(
        output_path,
        facecolor="black",
        bbox_inches=None,
        pad_inches=0
    )

    plt.close(fig)

    print("Success!")

def main():
    print("=== visualize.py 開始 ===")

    parser = argparse.ArgumentParser()

    parser.add_argument("input")
    parser.add_argument("meta")
    parser.add_argument("output")

    args = parser.parse_args()

    input_dir = Path(args.input)
    meta_path = Path(args.meta)
    output_dir = Path(args.output)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    meta = pd.read_csv(meta_path)

    sort = sorted(input_dir.glob("*.csv"))

    total = len(sort)

    for f in sort:
        output_path = output_dir / f"{f.stem}.jpg"

        visualize(
            f,
            meta,
            output_path
        )

    print("\n=== visualize.py 完了 ===")
    print(f"処理ファイル数: {total}")
    print(f"出力先: {output_dir}")

if __name__ == "__main__":
    main()