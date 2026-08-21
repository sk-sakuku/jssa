from pathlib import Path
import argparse

import cv2
import numpy as np
from skimage.morphology import skeletonize


def process_image(rgb_path, output_root):
    rgb = cv2.imread(
        str(rgb_path)
    )

    if rgb is None:
        return False

    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    # edge_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    # edge_x = cv2.convertScaleAbs(edge_x)

    # edge_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # edge_y = cv2.convertScaleAbs(edge_y)

    # tv, binary = cv2.threshold(
    #     gray,
    #     0,
    #     255,
    #     cv2.THRESH_BINARY + cv2.THRESH_OTSU
    # )

    # binary_bool = binary > 0
    # skeleton = (skeletonize(binary_bool)*255).astype(np.uint8)

    relative = Path(rgb_path).name

    save_image(
        Path(output_root) / "lab" / relative,
        lab
    )

    """
    save_image(
        Path(output_root) / "gray" / relative,
        gray
    )
    """

    save_image(
        Path(output_root) / "edges" / relative,
        edges
    )

    """
    save_image(
        Path(output_root) / "edge_x" / relative,
        edge_x
    )
    """

    """
    save_image(
        Path(output_root) / "edge_y" / relative,
        edge_y
    )
    """

    """
    save_image(
        Path(output_root) / "binary" / relative,
        binary
    )
    """

    """
    save_image(
        Path(output_root) / "skeleton" / relative,
        skeleton
    )
    """

    return True


def save_image(path, img):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    return cv2.imwrite(
        str(path),
        img
    )


def process_all(input_dir, output_dir):
    extensions = [
        ".jpg"
    ]

    files = []

    for file in Path(input_dir).rglob("*"):
        if file.suffix.lower() in extensions:
            files.append(file)

    for t in enumerate(files):
        file = t[1]
        i = t[0]

        process_image(
            str(file),
            Path(output_dir)
        )

        print(
            f"[{i + 1}/{len(files)}] {file.name} processed."
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("input")
    parser.add_argument("output")

    args = parser.parse_args()

    process_all(args.input, args.output)


if __name__ == "__main__":
    main()