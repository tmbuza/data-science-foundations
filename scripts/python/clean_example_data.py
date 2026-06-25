#!/usr/bin/env python3
"""
Clean an example tidy table and write a cleaning report.

Usage:
    python scripts/python/clean_example_data.py <input_table> <output_table> <output_dir>

Example:
    python scripts/python/clean_example_data.py data/iris.csv data/iris_clean.csv results/cleaning
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or TSV table based on file extension."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")

    raise ValueError(
        f"Unsupported input extension: {suffix}. "
        "Use .csv, .tsv, or .txt."
    )


def write_table(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV or TSV table based on file extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(path, index=False)
        return

    if suffix in {".tsv", ".txt"}:
        df.to_csv(path, sep="\t", index=False)
        return

    raise ValueError(
        f"Unsupported output extension: {suffix}. "
        "Use .csv, .tsv, or .txt."
    )


def clean_column_name(name: object) -> str:
    """Convert a column name into a simple snake_case-style name."""
    cleaned = str(name).strip().lower()
    cleaned = re.sub(r"[()\[\]{}]", "", cleaned)
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("_")
    return cleaned


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using simple teaching rules."""
    cleaned = df.copy()

    numeric_cols = cleaned.select_dtypes(include="number").columns.tolist()
    categorical_cols = cleaned.select_dtypes(exclude="number").columns.tolist()

    for column in numeric_cols:
        if cleaned[column].isna().any():
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    for column in categorical_cols:
        if cleaned[column].isna().any():
            mode_values = cleaned[column].mode(dropna=True)
            fill_value = "unknown" if mode_values.empty else mode_values.iloc[0]
            cleaned[column] = cleaned[column].fillna(fill_value)

    return cleaned


def write_report(
    report_path: Path,
    input_path: Path,
    output_path: Path,
    before_shape: tuple[int, int],
    after_shape: tuple[int, int],
    duplicate_count: int,
    duplicate_after: int,
    missing_before: int,
    missing_after: int,
    original_columns: list[str],
    cleaned_columns: list[str],
) -> None:
    """Write a plain-text cleaning report."""
    lines = [
        "CDI Data Science Foundations System",
        "Cleaning Report",
        "",
        f"Input file: {input_path}",
        f"Output file: {output_path}",
        "",
        f"Rows before cleaning: {before_shape[0]}",
        f"Columns before cleaning: {before_shape[1]}",
        f"Rows after cleaning: {after_shape[0]}",
        f"Columns after cleaning: {after_shape[1]}",
        "",
        f"Duplicate rows detected before cleaning: {duplicate_count}",
        f"Duplicate rows after cleaning: {duplicate_after}",
        f"Missing values before cleaning: {missing_before}",
        f"Missing values after cleaning: {missing_after}",
        "",
        "Original columns:",
        *[f"- {column}" for column in original_columns],
        "",
        "Cleaned columns:",
        *[f"- {column}" for column in cleaned_columns],
        "",
        "Cleaning rules applied:",
        "- Standardized column names",
        "- Removed duplicate rows",
        "- Filled missing numeric values with column medians",
        "- Filled missing categorical values with column modes",
        "- Validated missing values and duplicate rows after cleaning",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 1

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])

    if not input_path.exists():
        print(f"ERROR: input table not found: {input_path}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(input_path)

    before_shape = df.shape
    original_columns = df.columns.tolist()
    missing_before = int(df.isna().sum().sum())
    duplicate_count = int(df.duplicated().sum())

    cleaned = df.copy()
    cleaned.columns = [clean_column_name(column) for column in cleaned.columns]

    if duplicate_count > 0:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    cleaned = fill_missing_values(cleaned)

    for column in cleaned.select_dtypes(include="number").columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if "species" in cleaned.columns:
        cleaned["species"] = cleaned["species"].astype("category")

    missing_after = int(cleaned.isna().sum().sum())
    duplicate_after = int(cleaned.duplicated().sum())

    if missing_after != 0:
        raise ValueError("Missing values remain after cleaning.")

    if duplicate_after != 0:
        raise ValueError("Duplicate rows remain after cleaning.")

    write_table(cleaned, output_path)

    write_report(
        report_path=output_dir / "cleaning-report.txt",
        input_path=input_path,
        output_path=output_path,
        before_shape=before_shape,
        after_shape=cleaned.shape,
        duplicate_count=duplicate_count,
        duplicate_after=duplicate_after,
        missing_before=missing_before,
        missing_after=missing_after,
        original_columns=original_columns,
        cleaned_columns=cleaned.columns.tolist(),
    )

    print("Data cleaning complete.")
    print(f"Input: {input_path}")
    print(f"Cleaned output: {output_path}")
    print(f"Report: {output_dir / 'cleaning-report.txt'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
