#!/usr/bin/env python3
"""
Summarize a wrangled tidy table and write an evidence package.

Usage:
    python scripts/python/summarize_table.py <input_table> <output_dir>

Example:
    python scripts/python/summarize_table.py data/iris_wrangled.csv results/summary
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
    "petal_area",
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


def validate_required_columns(df: pd.DataFrame) -> None:
    """Validate that expected summary columns are present."""
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))

    if missing:
        raise ValueError(
            "Input table is missing required columns: "
            + ", ".join(missing)
        )

    if int(df.isna().sum().sum()) != 0:
        raise ValueError("Input table contains missing values.")


def create_numeric_summary(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Create a descriptive statistics table for numeric columns."""
    return df[numeric_cols].describe().transpose().reset_index(names="feature")


def create_species_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Create a species count table."""
    counts = df["species"].value_counts().sort_index().reset_index()
    counts.columns = ["species", "n"]
    return counts


def create_grouped_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create grouped mean and standard deviation summaries."""
    return (
        df.groupby("species", observed=False)
        .agg(
            n=("species", "size"),
            sepal_length_mean=("sepal_length", "mean"),
            sepal_length_sd=("sepal_length", "std"),
            sepal_width_mean=("sepal_width", "mean"),
            sepal_width_sd=("sepal_width", "std"),
            petal_length_mean=("petal_length", "mean"),
            petal_length_sd=("petal_length", "std"),
            petal_width_mean=("petal_width", "mean"),
            petal_width_sd=("petal_width", "std"),
            petal_area_mean=("petal_area", "mean"),
            petal_area_sd=("petal_area", "std"),
        )
        .reset_index()
    )


def create_median_summary(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Create a grouped median table."""
    return (
        df.groupby("species", observed=False)[numeric_cols]
        .median()
        .reset_index()
    )


def create_feature_separation(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Estimate descriptive separation using range of species means."""
    records = []

    for column in numeric_cols:
        group_means = df.groupby("species", observed=False)[column].mean()
        records.append(
            {
                "feature": column,
                "min_group_mean": group_means.min(),
                "max_group_mean": group_means.max(),
                "range_of_group_means": group_means.max() - group_means.min(),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values("range_of_group_means", ascending=False)
        .reset_index(drop=True)
    )


def markdown_table_from_dataframe(df: pd.DataFrame) -> str:
    """Create a simple markdown table without requiring tabulate."""
    if df.empty:
        return "_No rows available._"

    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows = []
    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator, *rows])


def create_insights_report(
    df: pd.DataFrame,
    feature_separation: pd.DataFrame,
) -> str:
    """Create a short markdown insights report."""
    top_sep = feature_separation.iloc[0]

    petal_length_means = (
        df.groupby("species", observed=False)["petal_length"]
        .mean()
        .sort_values(ascending=False)
        .reset_index(name="petal_length_mean")
    )

    petal_length_table = markdown_table_from_dataframe(petal_length_means)

    return f"""# Insights Report: Iris Dataset

## 1. Dataset size

- Rows: {df.shape[0]}
- Columns: {df.shape[1]}

## 2. Group balance

The dataset contains {df["species"].nunique()} species groups.

## 3. Group separation

Petal-related features show strong descriptive differences across species.

Mean petal length by species:

{petal_length_table}

## 4. Strongest descriptive separation

The feature with the largest range of group means is `{top_sep["feature"]}`.

- Minimum group mean: {top_sep["min_group_mean"]:.3f}
- Maximum group mean: {top_sep["max_group_mean"]:.3f}
- Range of group means: {top_sep["range_of_group_means"]:.3f}

## 5. Caution

These are descriptive patterns only. They support comparison, but they do not justify causal claims.
"""


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
    validate_required_columns(df)

    numeric_cols = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "petal_area",
    ]

    numeric_summary = create_numeric_summary(df, numeric_cols)
    species_counts = create_species_counts(df)
    grouped_summary = create_grouped_summary(df)
    median_summary = create_median_summary(df, numeric_cols)
    feature_separation = create_feature_separation(df, numeric_cols)
    correlation_matrix = df[numeric_cols].corr()
    insights_report = create_insights_report(
        df=df,
        feature_separation=feature_separation,
    )

    numeric_summary.to_csv(
        output_dir / "numeric-summary.tsv",
        sep="\t",
        index=False,
    )

    species_counts.to_csv(
        output_dir / "species-counts.tsv",
        sep="\t",
        index=False,
    )

    grouped_summary.to_csv(
        output_dir / "grouped-summary.tsv",
        sep="\t",
        index=False,
    )

    median_summary.to_csv(
        output_dir / "median-summary.tsv",
        sep="\t",
        index=False,
    )

    feature_separation.to_csv(
        output_dir / "feature-separation.tsv",
        sep="\t",
        index=False,
    )

    correlation_matrix.to_csv(
        output_dir / "correlation-matrix.tsv",
        sep="\t",
    )

    (output_dir / "analysis-insights.md").write_text(
        insights_report,
        encoding="utf-8",
    )

    print("Summary outputs complete.")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Insights report: {output_dir / 'analysis-insights.md'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
