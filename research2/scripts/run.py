from pathlib import Path
import subprocess
import sys

from win11toast import toast


ROOT = Path(__file__).resolve().parent.parent

SCRIPTS = [
    "preprocess.py",
    "generate_figures.py",
    "analyze_curvature.py",
    "feature_values.py",
    "mapping.py",
    "visualize.py",
]

def run(script, args):
    command = [
        sys.executable,
        str(ROOT / "scripts" / script),
        *args,
    ]

    print(f"\n=== {script} ===")
    print(" ".join(command))

    subprocess.run(command, check=True)

def main():
    # python scripts/preprocess.py data/raw data/processed/rgb
    run(
        "preprocess.py",
        [
            str(ROOT / "data" / "raw"),
            str(ROOT / "data" / "processed" / "rgb"),
        ],
    )

    # python scripts/generate_figures.py data/processed/rgb data/processed
    run(
        "generate_figures.py",
        [
            str(ROOT / "data" / "processed" / "rgb"),
            str(ROOT / "data" / "processed"),
        ],
    )

    # python scripts/analyze_curvature.py data/processed/edges data
    run(
        "analyze_curvature.py",
        [
            str(ROOT / "data" / "processed" / "edges"),
            str(ROOT / "data")
        ]
    )

    # python scripts/feature_values.py data/processed outputs
    run(
        "feature_values.py",
        [
            str(ROOT / "data" / "processed"),
            str(ROOT / "outputs"),
            str(ROOT / "data")
        ],
    )

    # python scripts/mapping.py outputs/feature_values.csv outputs/maps
    run(
        "mapping.py",
        [
            str(ROOT / "outputs" / "feature_values.csv"),
            str(ROOT / "outputs" / "maps"),
        ],
    )

    # python scripts/visualize.py outputs/maps data/meta.csv outputs/figures
    run(
        "visualize.py",
        [
            str(ROOT / "outputs" / "maps"),
            str(ROOT / "data" / "meta.csv"),
            str(ROOT / "outputs" / "figures"),
        ],
    )

    toast("SUCCESS!", "すべての処理が終了しました!")


if __name__ == "__main__":
    main()