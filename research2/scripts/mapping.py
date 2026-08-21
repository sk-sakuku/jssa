from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


GROUPS = {
    "lab_color_stats": [
        "L_mean",
        "L_std",
        "a_mean",
        "a_std",
        "b_mean",
        "b_std"
    ],

    "lab_ssim": [
        "lab_ssim_hor",
        "lab_ssim_ver"
    ],

    "edge_ssim": [
        "edge_ssim_hor",
        "edge_ssim_ver"
    ],

    "edge_density": [
        "edge_density"
    ] + [
        f"edge_density_field_{row}_{column}"
        for row in range(1, 9)
        for column in range(1, 9)
    ],

    "edge_length": [
        "edge_length_mean",
        "edge_length_std"
    ],

    "edge_curvature": [
        "edge_curvature_mean",
        "edge_curvature_entropy"
    ] + [
        f"edge_curvature_hist_{i:02d}"
        for i in range(1, 21)
    ],

    "line_angle": [
        "line_angle_mean",
        "line_angle_entropy"
    ] + [
        f"line_angle_hist_{i:02d}"
        for i in range(1, 37)
    ]
}

def reduce_to_2d(data):
    scaler = StandardScaler() # 標準化用オブジェクト

    scaled = scaler.fit_transform(data)

    pca = PCA(n_components=2)

    coordinates = pca.fit_transform(scaled)

    return coordinates

def make_mapping(df, columns):
    ids = df["name"]
    data = df[columns]

    coordinates = reduce_to_2d(data)

    result = pd.DataFrame({
        "id": ids,
        "x": coordinates[:, 0],
        "y": coordinates[:, 1]
    })

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("input")
    parser.add_argument("output")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(input_path)

    # each groups

    for group_name, columns in GROUPS.items():
        data = df[columns]

        coordinates = reduce_to_2d(data)

        result = pd.DataFrame({
            "id": df["name"],
            "x": coordinates[:, 0],
            "y": coordinates[:, 1]
        })

        result.to_csv(
            output_dir / f"{group_name}.csv",
            index=False
        )

        print(
            f"[Saved] Output written to: {output_dir / f'{group_name}.csv'}"
        )

    # all features

    all_columns = []

    for columns in GROUPS.values():
        all_columns.extend(columns)

    data = df[all_columns]

    coordinates = reduce_to_2d(data)

    result = pd.DataFrame({
        "id": df["name"],
        "x": coordinates[:, 0],
        "y": coordinates[:, 1]
    })

    result.to_csv(
        output_dir / "all.csv",
        index=False
    )

    print(
        f"[Saved] Output written to: {output_dir / 'all.csv'}"
    )


if __name__ == "__main__":
    main()