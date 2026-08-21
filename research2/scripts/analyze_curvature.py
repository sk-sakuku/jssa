from pathlib import Path
import argparse

import cv2
import numpy as np
from scipy.interpolate import splprep, splev

import json


MIN_CONTOUR_LENGTH = 10


def get_contour(edges):
    contours, hier = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
    )

    tmp_countours = []

    for contour in contours:
        if cv2.arcLength(contour, closed=False) > MIN_CONTOUR_LENGTH:
            tmp_countours.append(contour)

    contours = tmp_countours

    return contours


def curvature(edges):
    contours = get_contour(edges)

    scale = max(edges.shape[0], edges.shape[1])

    curvatures = []

    for contour in contours:
        points = contour[:, 0, :].astype(float)

        x = points[:, 0] / scale
        y = points[:, 1] / scale

        # tck       3次タプル           曲線の数式データ
        # u         1次元のfloat配列    それぞれの点の曲線の中での位置
        tck, u = splprep(
            [x, y],
            k=3,
            s=5
        )

        # 0～1（uは0～1で表されるため）を100分割して、等間隔になった100個の点を取得する
        u_hundred = np.linspace(0, 1, 100)

        x_hundred, y_hundred = splev(
            u_hundred,
            tck
        )

        dx, dy = splev(
            u_hundred,
            tck,
            der=1
        )

        ddx, ddy = splev(
            u_hundred,
            tck,
            der=2
        )

        numerator = np.abs(
            dx * ddy - dy * ddx
        )

        denominator = (
            dx**2 + dy**2
        )**1.5

        valid = denominator > 1e-12

        if np.any(valid):
            curvature = numerator[valid] / denominator[valid]
            curvatures.extend(curvature)

    curvatures = np.array(curvatures)

    return curvatures


def collect_curvatures(input_dir):
    curvatures = []

    image_files = sorted(Path(input_dir).glob("*.jpg"))

    for image_file in image_files:
        edges = cv2.imread(
            str(image_file),
            cv2.IMREAD_GRAYSCALE
        )

        image_curvatures = curvature(edges)

        curvatures.extend(image_curvatures)

    print(f"curvatures: {len(curvatures)}")

    return np.array(curvatures)


def analyze_curvatures(curvatures):
    output = {}

    count = len(curvatures)
    output["min"] = np.min(curvatures)
    output["max"] = np.max(curvatures)
    output["mean"] = np.mean(curvatures)
    output["median"] = np.median(curvatures)
    output["standard deviation"] = np.std(curvatures)

    p1 = np.percentile(curvatures, 1)
    p99 = np.percentile(curvatures, 99)

    output["p1"] = p1
    output["p99"] = p99

    bin_borders = np.linspace(p1, p99, 21)

    for i in range(20):
        start = bin_borders[i]
        end = bin_borders[i + 1]

        s = f"{start:.6f}"
        e = f"{end:.6f}"

        output[f"{i + 1:02d}"] = (float(s), float(e))

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("input")
    parser.add_argument("output")

    args = parser.parse_args()

    result = analyze_curvatures(
        collect_curvatures(
            args.input
        )
    )

    print(
        json.dumps(
            result,
            indent=4
        )
    )

    output_dir = Path(args.output)

    json_file_path = output_dir / "curvatures_data.json"

    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"\n...\n[Saved] Output written to: {json_file_path.resolve()}")


if __name__ == "__main__":
    main()