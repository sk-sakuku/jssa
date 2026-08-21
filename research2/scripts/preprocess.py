from pathlib import Path
import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()

import cv2
import numpy as np
import argparse


SIZE = 1024


def resize(img, size=SIZE):
    h = img.shape[0]
    w = img.shape[1]

    scale = size / max(h, w)

    new_h = int(h * scale)
    new_w = int(w * scale)

    resized = cv2.resize(img, (new_w, new_h))

    top = (size - new_h) // 2
    bottom = size - new_h - top

    left = (size - new_w) // 2
    right = size - new_w - left

    border_color = (255, 255, 255)  # 白色の境界線

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=border_color
    )

    return padded


def process_image(img_path, output_path):
    img = cv2.imread(img_path)

    if img is None:
        return False

    resized = resize(img)

    output_dir = os.path.dirname(output_path) # name(/ + stem + suffix)を除いたパス

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    success = cv2.imwrite(output_dir + "/" + output_path.stem + ".jpg", resized) # True or False

    return success # True or False


def process_all(input_dir, output_dir):
    extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
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
            Path(output_dir) / file.name
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