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
from pandas.api.types import is_numeric_dtype


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


def infer_analytical_role(series: pd.Series) -> str:
    """Infer a broad analytical role without changing the stored dtype."""
    if is_numeric_dtype(series):
        return "numeric"

    return "categorical_or_text"


def classify_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create a column-level summary table.

    CSV files do not preserve pandas categorical dtypes. For example, the
    chapter's ``species`` column is read as ``object`` even though it is used
    analytically as a categorical grouping variable. The output therefore
    records both the pandas dtype and a broad analytical role.
    """
    rows = []

    for column in df.columns:
        series = df[column]
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "analytical_role": infer_analytical_role(series),
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
    non_numeric_columns = df.select_dtypes(exclude="number").columns.tolist()

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
        "Non-numeric columns (possible categorical or text variables):",
        *([f"- {column}" for column in non_numeric_columns] or ["- None detected"]),
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
    if df.empty:
        missing_summary["missing_percent"] = 0.0
    else:
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
