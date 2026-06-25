#!/usr/bin/env python3
"""
Wrangle a cleaned tidy table and write downstream analysis outputs.

Usage:
    python scripts/python/wrangle_example_data.py <input_table> <output_table> <output_dir>

Example:
    python scripts/python/wrangle_example_data.py data/iris_clean.csv data/iris_wrangled.csv results/wrangling
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species",
}


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


def validate_required_columns(df: pd.DataFrame) -> None:
    """Validate that expected Iris columns are present."""
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))

    if missing:
        raise ValueError(
            "Input table is missing required columns: "
            + ", ".join(missing)
        )


def create_species_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create a species-level summary table."""
    return (
        df.groupby("species", observed=False)
        .agg(
            n=("species", "size"),
            mean_petal_area=("petal_area", "mean"),
            median_petal_area=("petal_area", "median"),
            mean_sepal_length=("sepal_length", "mean"),
            mean_sepal_width=("sepal_width", "mean"),
            mean_petal_length=("petal_length", "mean"),
            mean_petal_width=("petal_width", "mean"),
        )
        .reset_index()
        .sort_values("mean_petal_area", ascending=False)
    )


def create_long_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a long-format measurement table."""
    return df.melt(
        id_vars=["species", "size_category"],
        value_vars=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        var_name="measurement",
        value_name="value",
    )


def write_report(
    report_path: Path,
    input_path: Path,
    output_table: Path,
    output_dir: Path,
    rows_before: int,
    rows_after: int,
    species_count: int,
) -> None:
    """Write a plain-text wrangling report."""
    lines = [
        "CDI Data Science Foundations System",
        "Wrangling Report",
        "",
        f"Input file: {input_path}",
        f"Wrangled table: {output_table}",
        f"Output directory: {output_dir}",
        "",
        f"Rows before wrangling: {rows_before}",
        f"Rows after wrangling: {rows_after}",
        f"Species groups detected: {species_count}",
        "",
        "Outputs created:",
        f"- {output_table}",
        f"- {output_dir / 'species-summary.tsv'}",
        f"- {output_dir / 'iris-long.tsv'}",
        f"- {report_path}",
        "",
        "Wrangling steps applied:",
        "- Created petal_area",
        "- Created size_category using median petal_area",
        "- Sorted table by petal_area",
        "- Created species-level summary table",
        "- Created long-format measurement table",
        "- Validated missing values and expected columns",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 1

    input_path = Path(sys.argv[1])
    output_table = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])

    if not input_path.exists():
        print(f"ERROR: input table not found: {input_path}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(input_path)
    rows_before = len(df)

    validate_required_columns(df)

    if int(df.isna().sum().sum()) != 0:
        raise ValueError("Input table contains missing values. Clean it first.")

    wrangled = df.copy()
    wrangled["petal_area"] = wrangled["petal_length"] * wrangled["petal_width"]

    median_petal_area = wrangled["petal_area"].median()
    wrangled["size_category"] = wrangled["petal_area"].apply(
        lambda value: "large" if value > median_petal_area else "small"
    )

    wrangled = wrangled.sort_values(
        by=["petal_area", "species"],
        ascending=[False, True],
    ).reset_index(drop=True)

    if int(wrangled.isna().sum().sum()) != 0:
        raise ValueError("Missing values detected after wrangling.")

    species_summary = create_species_summary(wrangled)
    long_table = create_long_table(wrangled)

    write_table(wrangled, output_table)

    species_summary.to_csv(
        output_dir / "species-summary.tsv",
        sep="\t",
        index=False,
    )

    long_table.to_csv(
        output_dir / "iris-long.tsv",
        sep="\t",
        index=False,
    )

    write_report(
        report_path=output_dir / "wrangling-report.txt",
        input_path=input_path,
        output_table=output_table,
        output_dir=output_dir,
        rows_before=rows_before,
        rows_after=len(wrangled),
        species_count=wrangled["species"].nunique(),
    )

    print("Data wrangling complete.")
    print(f"Input: {input_path}")
    print(f"Wrangled output: {output_table}")
    print(f"Output directory: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
