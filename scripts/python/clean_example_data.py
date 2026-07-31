#!/usr/bin/env python3
"""
Clean the Chapter 03 Iris table and write a cleaning report.

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


MEASUREMENT_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
REQUIRED_COLUMNS = [*MEASUREMENT_COLUMNS, "species"]


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or tab-separated table based on its file extension."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")

    raise ValueError(
        f"Unsupported input extension: {suffix or '(none)'}. "
        "Use .csv, .tsv, or .txt."
    )


def write_table(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV or tab-separated table based on its file extension."""
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix in {".tsv", ".txt"}:
        df.to_csv(path, sep="\t", index=False)
        return

    raise ValueError(
        f"Unsupported output extension: {suffix or '(none)'}. "
        "Use .csv, .tsv, or .txt."
    )


def clean_column_name(name: object) -> str:
    """Convert a column name to a simple snake_case-style name."""
    cleaned = str(name).strip().lower()
    cleaned = re.sub(r"[()\[\]{}]", "", cleaned)
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_")


def standardize_and_validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and confirm the expected Iris schema."""
    cleaned = df.copy()
    cleaned.columns = [clean_column_name(column) for column in cleaned.columns]

    empty_names = [index for index, name in enumerate(cleaned.columns) if not name]
    if empty_names:
        positions = ", ".join(str(index + 1) for index in empty_names)
        raise ValueError(
            "Column-name standardization produced an empty name at "
            f"position(s): {positions}."
        )

    duplicate_names = cleaned.columns[cleaned.columns.duplicated()].unique().tolist()
    if duplicate_names:
        raise ValueError(
            "Column-name standardization produced duplicate names: "
            + ", ".join(duplicate_names)
        )

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in cleaned.columns
    ]
    if missing_columns:
        raise ValueError(
            "Required Iris column(s) not found after name standardization: "
            + ", ".join(missing_columns)
        )

    return cleaned


def convert_measurements(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert expected measurement columns and count newly coerced values."""
    cleaned = df.copy()
    coercions: dict[str, int] = {}

    for column in MEASUREMENT_COLUMNS:
        missing_before = cleaned[column].isna()
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        newly_missing = cleaned[column].isna() & ~missing_before
        coercions[column] = int(newly_missing.sum())

    return cleaned, coercions


def fill_missing_values(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, str]]:
    """Apply the Chapter 03 median-and-mode teaching rules."""
    cleaned = df.copy()
    filled_counts: dict[str, int] = {}
    fill_methods: dict[str, str] = {}

    numeric_columns = cleaned.select_dtypes(include="number").columns.tolist()
    categorical_columns = cleaned.select_dtypes(exclude="number").columns.tolist()

    for column in numeric_columns:
        missing_count = int(cleaned[column].isna().sum())
        if missing_count == 0:
            continue

        median = cleaned[column].median()
        if pd.isna(median):
            raise ValueError(
                f"Cannot impute numeric column '{column}': "
                "the entire column is missing or non-numeric."
            )

        cleaned[column] = cleaned[column].fillna(median)
        filled_counts[column] = missing_count
        fill_methods[column] = f"median ({median:g})"

    for column in categorical_columns:
        missing_count = int(cleaned[column].isna().sum())
        if missing_count == 0:
            continue

        modes = cleaned[column].mode(dropna=True)
        if modes.empty:
            raise ValueError(
                f"Cannot impute categorical column '{column}': "
                "the entire column is missing."
            )

        mode = modes.iloc[0]
        cleaned[column] = cleaned[column].fillna(mode)
        filled_counts[column] = missing_count
        fill_methods[column] = f"mode ({mode})"

    return cleaned, filled_counts, fill_methods


def validate_cleaned_table(df: pd.DataFrame) -> None:
    """Confirm that the cleaned table satisfies this lesson's expectations."""
    non_numeric = [
        column
        for column in MEASUREMENT_COLUMNS
        if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric:
        raise ValueError(
            "Expected numeric column(s) are not numeric: " + ", ".join(non_numeric)
        )

    missing_after = int(df.isna().sum().sum())
    if missing_after:
        raise ValueError(
            f"{missing_after} missing value(s) remain after cleaning."
        )


def write_report(
    report_path: Path,
    *,
    input_path: Path,
    output_path: Path,
    before_shape: tuple[int, int],
    after_shape: tuple[int, int],
    original_columns: list[str],
    cleaned_columns: list[str],
    missing_original: int,
    missing_after_conversion: int,
    missing_after_cleaning: int,
    repeated_rows: int,
    identical_group_rows: int,
    coercions: dict[str, int],
    filled_counts: dict[str, int],
    fill_methods: dict[str, str],
) -> None:
    """Write a plain-text record of what the workflow found and changed."""
    total_coercions = sum(coercions.values())
    total_filled = sum(filled_counts.values())

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
        f"Missing values in original table: {missing_original}",
        f"Values converted to missing during numeric coercion: {total_coercions}",
        f"Missing values after numeric conversion: {missing_after_conversion}",
        f"Missing values filled: {total_filled}",
        f"Missing values after cleaning: {missing_after_cleaning}",
        "",
        f"Repeated rows after the first occurrence: {repeated_rows}",
        f"Rows participating in identical-row groups: {identical_group_rows}",
        "Identical rows removed: 0",
        "Decision: Identical rows were reported and retained because identical",
        "values alone do not prove that records are accidental duplicates.",
        "",
        "Original columns:",
        *[f"- {column}" for column in original_columns],
        "",
        "Standardized columns:",
        *[f"- {column}" for column in cleaned_columns],
        "",
        "Numeric conversion results:",
        *[
            f"- {column}: {coercions[column]} value(s) coerced to missing"
            for column in MEASUREMENT_COLUMNS
        ],
        "",
        "Missing-value actions:",
    ]

    if filled_counts:
        lines.extend(
            f"- {column}: filled {filled_counts[column]} value(s) using "
            f"{fill_methods[column]}"
            for column in filled_counts
        )
    else:
        lines.append("- No missing values required imputation")

    lines.extend(
        [
            "",
            "Validation results:",
            "- All required Iris columns are present",
            "- All four measurement columns are numeric",
            "- No missing values remain",
            "- Identical rows were measured and retained",
            "",
            "Note: The pandas category type assigned to 'species' is not stored",
            "in CSV or TSV output. Convert it again after reloading if needed.",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run the Chapter 03 cleaning workflow."""
    if len(sys.argv) != 4:
        print(__doc__)
        return 1

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])

    try:
        if not input_path.is_file():
            raise FileNotFoundError(f"Input table not found: {input_path}")

        if input_path.resolve() == output_path.resolve():
            raise ValueError(
                "Input and output paths must be different. "
                "Keep the original table and save a new cleaned table."
            )

        df = read_table(input_path)
        before_shape = df.shape
        original_columns = [str(column) for column in df.columns]
        missing_original = int(df.isna().sum().sum())

        cleaned = standardize_and_validate_columns(df)

        repeated_rows = int(cleaned.duplicated().sum())
        identical_group_rows = int(cleaned.duplicated(keep=False).sum())

        cleaned, coercions = convert_measurements(cleaned)
        missing_after_conversion = int(cleaned.isna().sum().sum())

        cleaned, filled_counts, fill_methods = fill_missing_values(cleaned)
        cleaned["species"] = cleaned["species"].astype("category")

        validate_cleaned_table(cleaned)
        missing_after_cleaning = int(cleaned.isna().sum().sum())

        write_table(cleaned, output_path)

        report_path = output_dir / "cleaning-report.txt"
        write_report(
            report_path,
            input_path=input_path,
            output_path=output_path,
            before_shape=before_shape,
            after_shape=cleaned.shape,
            original_columns=original_columns,
            cleaned_columns=cleaned.columns.tolist(),
            missing_original=missing_original,
            missing_after_conversion=missing_after_conversion,
            missing_after_cleaning=missing_after_cleaning,
            repeated_rows=repeated_rows,
            identical_group_rows=identical_group_rows,
            coercions=coercions,
            filled_counts=filled_counts,
            fill_methods=fill_methods,
        )

    except (FileNotFoundError, OSError, pd.errors.ParserError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("Data cleaning complete.")
    print(f"Input: {input_path}")
    print(f"Cleaned output: {output_path}")
    print(f"Report: {report_path}")
    print(f"Identical-group rows retained: {identical_group_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
