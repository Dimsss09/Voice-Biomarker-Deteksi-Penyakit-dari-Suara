"""Download the default UCI Parkinson's voice dataset."""

from __future__ import annotations

import argparse

from src.data import download_dataset, load_raw_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if the file exists.")
    args = parser.parse_args()

    path = download_dataset(force=args.force)
    frame = load_raw_dataset()
    print(f"Dataset saved to: {path}")
    print(f"Shape: {frame.shape[0]} rows x {frame.shape[1]} columns")
    print(f"Columns: {', '.join(frame.columns)}")


if __name__ == "__main__":
    main()
