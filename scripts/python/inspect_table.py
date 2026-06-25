#!/usr/bin/env python3
"""
Inspect a tidy table and write basic structure reports.

Usage:
    python scripts/python/inspect_table.py <input_table> <output_dir>

Example:
    python scripts/python/inspect_table.py data/iris.csv results/inspection
"""

from __future__ import annotations

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
        f"Unsupported file extension: {suffix}. "
        "Use .csv, .tsv, or .txt."
    )


def classify_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create a column-level summary table."""
    rows = []

    for column in df.columns:
        series = df[column]
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "missing_values": int(series.isna().sum()),
                "missing_percent": round(float(series.isna().mean() * 100), 2),
                "unique_values": int(series.nunique(dropna=True)),
                "example_value": "" if series.dropna().empty else str(series.dropna().iloc[0]),
            }
        )

    return pd.DataFrame(rows)


def write_summary(df: pd.DataFrame, input_path: Path, output_path: Path) -> None:
    """Write a plain-text inspection summary."""
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(exclude="number").columns.tolist()

    lines = [
        "CDI Data Science Foundations System",
        "Table Inspection Summary",
        "",
        f"Input file: {input_path}",
        f"Rows: {df.shape[0]}",
        f"Columns: {df.shape[1]}",
        "",
        "Column names:",
        *[f"- {column}" for column in df.columns],
        "",
        "Numeric columns:",
        *([f"- {column}" for column in numeric_columns] or ["- None detected"]),
        "",
        "Non-numeric / categorical columns:",
        *([f"- {column}" for column in categorical_columns] or ["- None detected"]),
        "",
        f"Total missing values: {int(df.isna().sum().sum())}",
    ]

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1

    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not input_path.exists():
        print(f"ERROR: input table not found: {input_path}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(input_path)

    column_summary = classify_columns(df)
    missing_summary = (
        df.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_values"})
    )
    missing_summary["missing_percent"] = (
        missing_summary["missing_values"] / len(df) * 100
    ).round(2)

    write_summary(
        df=df,
        input_path=input_path,
        output_path=output_dir / "table-inspection-summary.txt",
    )

    column_summary.to_csv(
        output_dir / "table-column-summary.tsv",
        sep="\t",
        index=False,
    )

    missing_summary.to_csv(
        output_dir / "table-missing-values.tsv",
        sep="\t",
        index=False,
    )

    print("Table inspection complete.")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
