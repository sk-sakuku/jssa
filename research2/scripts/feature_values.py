import json

from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd

from scipy.interpolate import splprep, splev
from scipy.stats import circmean, entropy
from skimage.metrics import structural_similarity as ssim


#============================================================
# 定数
#============================================================


MIN_CONTOUR_LENGTH = 10


def get_p99(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return float(data["p99"])


# ============================================================
# Labの平均・標準偏差
# ============================================================


def get_lab(lab):
    features = {}
    names = ["L", "a", "b"]

    for enumer in enumerate(names):
        i = enumer[0]
        name = enumer[1]

        channel = lab[:, :, i]

        mean = np.mean(channel)
        std = np.std(channel)

        features[
            f"{name}_mean"
        ] = mean

        features[
            f"{name}_std"
        ] = std

    return features


# ============================================================
# Labの対称性
# ============================================================


def get_lab_ssim(lab):
    flip_hor = np.fliplr(lab)
    flip_ver = np.flipud(lab)

    ssim_hor = ssim(
        lab,
        flip_hor,
        data_range=255,
        channel_axis=2
    )

    ssim_ver = ssim(
        lab,
        flip_ver,
        data_range=255,
        channel_axis=2
    )

    return {
        "lab_ssim_hor": ssim_hor,
        "lab_ssim_ver": ssim_ver
    }


# ============================================================
# エッジの対称性
# ============================================================


def get_edge_ssim(edges):
    flip_hor = np.fliplr(edges)
    flip_ver = np.flipud(edges)

    ssim_hor = ssim(edges, flip_hor, data_range=255)
    ssim_ver = ssim(edges, flip_ver, data_range=255)

    return {
        "edge_ssim_hor": ssim_hor,
        "edge_ssim_ver": ssim_ver
    }


# ============================================================
# エッジ密度
# ============================================================


def get_edge_density(edges):
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.shape[0] * edges.shape[1]

    density = edge_pixels / total_pixels

    return {
        "edge_density": density
    }


# ============================================================
# 8x8のブロックごとのエッジ密度
# ============================================================


def get_edge_density_field(edges):
    h = edges.shape[0]
    w = edges.shape[1]

    block_h = h / 8
    block_w = w / 8

    features = {}

    for ver in range(8):
        y_top = round(ver * block_h)
        y_bottom = round((ver + 1) * block_h)

        for hor in range(8):
            x_left = round(hor * block_w)
            x_right = round((hor + 1) * block_w)

            cell = edges[y_top:y_bottom, x_left:x_right]

            edge_pixels = np.sum(cell > 0)
            total_pixels = cell.shape[0] * cell.shape[1]

            density = edge_pixels / total_pixels

            features[
                f"edge_density_field_{ver + 1}_{hor + 1}"
            ] = density

    return features


# ============================================================
# エッジの輪郭
# ============================================================


def get_contour(edges):
    contours, hier = cv2.findContours(
        edges,
        cv2.RETR_LIST, # 階層構造を与えずに全ての輪郭を検出
        cv2.CHAIN_APPROX_NONE # 輪郭のすべてのピクセルを保持
    )

    tmp_countours = []

    for contour in contours:
        if cv2.arcLength(contour, closed=False) > MIN_CONTOUR_LENGTH:
            tmp_countours.append(contour)

    contours = tmp_countours

    return contours


# ============================================================
# エッジの長さ
# ============================================================


def get_edge_lengths(edges):
    lengths = []

    contours = get_contour(edges)

    for contour in contours:
        length = cv2.arcLength(contour, closed=False)
        lengths.append(length)

    if not lengths:
        return {
            "edge_length_mean": 0,
            "edge_length_std": 0
        }

    mean = np.mean(lengths)
    std = np.std(lengths)

    scale = max(edges.shape[0], edges.shape[1])

    return {
        "edge_length_mean": mean / scale,
        "edge_length_std": std / scale
    }

# ============================================================
# エッジの曲率
# ============================================================

def curvature(edges):
    contours = get_contour(edges)

    scale = max(edges.shape[0], edges.shape[1])

    curvatures = []

    for contour in contours:
        points = contour[:, 0, :].astype(float)

        x = points[:, 0] / scale
        y = points[:, 1] / scale

        # tck       3次元タプル         曲線の数式データ
        # u         1次元のfloat配列    それぞれの点の曲線の中での位置
        tck, u = splprep(
            [x, y],
            k=3,
            s=5
        )

        # 0～1（uは0～1で表されるため）を100分割し、等間隔の100点を取得
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

        curvature = numerator[valid] / denominator[valid]

        curvatures.extend(curvature)

    curvatures = np.array(curvatures)

    return curvatures


def get_curvature(edges, p99):
    features = {}

    curvatures = curvature(edges)

    # mean

    mean = np.mean(curvatures)

    features[
        "edge_curvature_mean"
    ] = mean

    # hist

    curvatures = np.clip(
        curvatures,
        0,
        p99
    )

    bins = np.linspace(
        0,
        p99,
        21
    )

    hist, n = np.histogram(
        curvatures,
        bins=bins
    )

    sum = np.sum(hist)
    probability = hist / sum

    for enumer in enumerate(probability):
        i = enumer[0]
        value = enumer[1]

        features[
            f"edge_curvature_hist_{i + 1:02d}"
        ] = value

    # entropy

    curvature_entropy = entropy(probability)

    features[
        "edge_curvature_entropy"
    ] = curvature_entropy

    return features


def get_line_angle(edges):
    features = {}

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=30,
        maxLineGap=10
    )

    angles = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line

            dx = x2 - x1
            dy = y2 - y1

            theta = np.arctan2(dy, dx)

            angle = np.degrees(theta) % 180

            angles.append(angle)

    angles = np.array(angles)

    if len(angles) == 0:
        return {}

    # mean

    mean_angle = circmean(
        angles,
        low=0,
        high=180
    )

    features[
        "line_angle_mean"
    ] = mean_angle

    # 36 histogram
    bins = np.linspace(
        0,
        180,
        37
    )

    hist, n = np.histogram(
        angles,
        bins=bins
    )

    sum = np.sum(hist)
    probability = hist / sum

    for enumer in enumerate(probability):
        i = enumer[0]
        value = enumer[1]

        features[
            f"line_angle_hist_{i + 1:02d}"
        ] = value

    # entropy

    curvature_entropy = entropy(probability)

    features[
        "line_angle_entropy"
    ] = curvature_entropy

    return features


def get_features(lab, edges, p99):
    features = {}

    features.update(get_lab(lab))
    features.update(get_lab_ssim(lab))
    features.update(get_edge_ssim(edges))
    features.update(get_edge_density(edges))
    features.update(get_edge_density_field(edges))
    features.update(get_edge_lengths(edges))
    features.update(get_curvature(edges, p99))
    features.update(get_line_angle(edges))

    return features


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("curvature")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    lab_dir = input_dir / "lab"
    edges_dir = input_dir / "edges"

    # 曲率のp99
    curve_path = Path(args.curvature) / "curvatures_data.json"
    p_99 = get_p99(curve_path)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    targets = sorted(lab_dir.glob("*.jpg"))

    for l in targets:
        name = l.stem

        e = edges_dir / f"{name}.jpg"

        if not e.exists():
            continue

        lab = cv2.imread(
            str(l)
        )

        edge = cv2.imread(
            str(e),
            cv2.IMREAD_GRAYSCALE
        )

        if lab is None or edge is None:
            continue

        features = get_features(
            lab,
            edge,
            p_99
        )

        features["name"] = name

        rows.append(features)

        print(name + " processed...")

    df = pd.DataFrame(rows)

    columns = ["name"]

    for column in df.columns:
        if column != "name":
            columns.append(column)

    df = df[columns]

    output_path = output_dir / "feature_values.csv"

    df.to_csv(
        output_path,
        index=False
    )

    print(f"\n...\n[Saved] Output written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()