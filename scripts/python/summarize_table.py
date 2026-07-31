#!/usr/bin/env python3
"""
Summarize the wrangled Iris table and write an evidence package.

Usage:
    python scripts/python/summarize_table.py <input_table> <output_dir>

Example:
    python scripts/python/summarize_table.py data/iris_wrangled.csv results/summary
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


NUMERIC_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "petal_area",
]
GROUP_COLUMN = "species"
REQUIRED_COLUMNS = {*NUMERIC_COLUMNS, GROUP_COLUMN}


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or tab-separated table."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(
        f"Unsupported input extension: {suffix}. Use .csv, .tsv, or .txt."
    )


def validate_input(df: pd.DataFrame) -> None:
    """Validate the columns and values required by this chapter."""
    missing_columns = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing_columns:
        raise ValueError(
            "Input table is missing required columns: "
            + ", ".join(missing_columns)
        )

    missing_values = int(df[list(REQUIRED_COLUMNS)].isna().sum().sum())
    if missing_values:
        raise ValueError(
            f"Required summary columns contain {missing_values} missing values."
        )

    non_numeric = [
        column
        for column in NUMERIC_COLUMNS
        if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric:
        raise ValueError(
            "Expected numeric columns are not numeric: " + ", ".join(non_numeric)
        )

    if df[GROUP_COLUMN].nunique() < 2:
        raise ValueError("At least two species groups are required.")


def create_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize center, conventional spread, and robust spread."""
    summary = (
        df[NUMERIC_COLUMNS]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .transpose()
        .reset_index(names="feature")
    )
    summary["q1"] = df[NUMERIC_COLUMNS].quantile(0.25).values
    summary["q3"] = df[NUMERIC_COLUMNS].quantile(0.75).values
    summary["iqr"] = summary["q3"] - summary["q1"]
    return summary[
        ["feature", "count", "mean", "median", "std", "q1", "q3", "iqr", "min", "max"]
    ]


def create_species_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Report group counts and percentages."""
    counts = (
        df[GROUP_COLUMN]
        .value_counts()
        .sort_index()
        .rename_axis(GROUP_COLUMN)
        .reset_index(name="n")
    )
    counts["percent"] = counts["n"] / counts["n"].sum() * 100
    return counts


def create_grouped_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create a long-format summary for every feature and species."""
    records: list[dict[str, object]] = []

    for species, group in df.groupby(GROUP_COLUMN, observed=False):
        for feature in NUMERIC_COLUMNS:
            values = group[feature]
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            records.append(
                {
                    GROUP_COLUMN: species,
                    "feature": feature,
                    "n": values.count(),
                    "mean": values.mean(),
                    "median": values.median(),
                    "sd": values.std(),
                    "q1": q1,
                    "q3": q3,
                    "iqr": q3 - q1,
                    "min": values.min(),
                    "max": values.max(),
                }
            )

    return pd.DataFrame(records)


def create_feature_differentiation(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate eta-squared as descriptive group differentiation."""
    records = []

    for feature in NUMERIC_COLUMNS:
        overall_mean = df[feature].mean()
        between_ss = sum(
            len(group) * (group[feature].mean() - overall_mean) ** 2
            for _, group in df.groupby(GROUP_COLUMN, observed=False)
        )
        total_ss = ((df[feature] - overall_mean) ** 2).sum()
        eta_squared = between_ss / total_ss if total_ss else 0.0
        records.append(
            {
                "feature": feature,
                "eta_squared": eta_squared,
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values("eta_squared", ascending=False)
        .reset_index(drop=True)
    )


def markdown_table(df: pd.DataFrame) -> str:
    """Create a Markdown table without an optional tabulate dependency."""
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        values = [
            f"{value:.3f}" if isinstance(value, float) else str(value)
            for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def create_insights_report(
    df: pd.DataFrame,
    species_counts: pd.DataFrame,
    grouped_summary: pd.DataFrame,
    feature_differentiation: pd.DataFrame,
) -> str:
    """Create a concise evidence-based Markdown report."""
    top_feature = feature_differentiation.iloc[0]
    petal_length_summary = grouped_summary.loc[
        grouped_summary["feature"].eq("petal_length"),
        [GROUP_COLUMN, "n", "mean", "median", "sd", "iqr"],
    ]

    return f"""# Insights Report: Iris Dataset

## Dataset and group balance

- The dataset contains {df.shape[0]} observations and {df.shape[1]} columns.
- It contains {df[GROUP_COLUMN].nunique()} species groups.
- Group sizes range from {species_counts["n"].min()} to {species_counts["n"].max()} observations.

## Petal length by species

{markdown_table(petal_length_summary)}

## Strongest descriptive differentiation

`{top_feature["feature"]}` has the largest eta-squared value
({top_feature["eta_squared"]:.3f}) among the summarized features.

## Interpretation

Petal measurements provide clearer descriptive differentiation among species
than sepal measurements in this dataset.

## Limitations

These results describe this dataset. They do not establish causation or
classification performance. Correlations involving `petal_area` partly reflect
how that derived feature was calculated.
"""


def write_outputs(
    output_dir: Path,
    numeric_summary: pd.DataFrame,
    species_counts: pd.DataFrame,
    grouped_summary: pd.DataFrame,
    feature_differentiation: pd.DataFrame,
    pearson_correlation: pd.DataFrame,
    spearman_correlation: pd.DataFrame,
    insights_report: str,
) -> None:
    """Write all summary outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric_summary.to_csv(
        output_dir / "numeric-summary.tsv", sep="\t", index=False
    )
    species_counts.to_csv(
        output_dir / "species-counts.tsv", sep="\t", index=False
    )
    grouped_summary.to_csv(
        output_dir / "grouped-summary.tsv", sep="\t", index=False
    )
    feature_differentiation.to_csv(
        output_dir / "feature-differentiation.tsv", sep="\t", index=False
    )
    pearson_correlation.to_csv(
        output_dir / "pearson-correlation.tsv", sep="\t"
    )
    spearman_correlation.to_csv(
        output_dir / "spearman-correlation.tsv", sep="\t"
    )
    (output_dir / "analysis-insights.md").write_text(
        insights_report, encoding="utf-8"
    )


def main() -> int:
    """Run the command-line workflow."""
    if len(sys.argv) != 3:
        print(__doc__)
        return 1

    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not input_path.is_file():
        print(f"ERROR: input table not found: {input_path}", file=sys.stderr)
        return 1

    try:
        df = read_table(input_path)
        validate_input(df)

        numeric_summary = create_numeric_summary(df)
        species_counts = create_species_counts(df)
        grouped_summary = create_grouped_summary(df)
        feature_differentiation = create_feature_differentiation(df)
        pearson_correlation = df[NUMERIC_COLUMNS].corr(method="pearson")
        spearman_correlation = df[NUMERIC_COLUMNS].corr(method="spearman")

        insights_report = create_insights_report(
            df,
            species_counts,
            grouped_summary,
            feature_differentiation,
        )

        write_outputs(
            output_dir,
            numeric_summary,
            species_counts,
            grouped_summary,
            feature_differentiation,
            pearson_correlation,
            spearman_correlation,
            insights_report,
        )
    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("Summary outputs complete.")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Insights report: {output_dir / 'analysis-insights.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
