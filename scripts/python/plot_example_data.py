#!/usr/bin/env python3
"""
Create foundational exploratory figures from a wrangled tidy table.

Usage:
    python scripts/python/plot_example_data.py <input_table> <output_dir>

Example:
    python scripts/python/plot_example_data.py data/iris_wrangled.csv results/figures
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
    """Validate that expected plotting columns are present."""
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))

    if missing:
        raise ValueError(
            "Input table is missing required columns: "
            + ", ".join(missing)
        )


def save_current_figure(path: Path) -> None:
    """Save and close the current matplotlib figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def create_histogram_sepal_length(df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(8, 5.5))

    sns.histplot(
        data=df,
        x="sepal_length",
        bins=12,
        kde=True,
        ax=ax,
    )

    ax.set_title("Distribution of Sepal Length")
    ax.set_xlabel("Sepal Length")
    ax.set_ylabel("Count")

    filename = "histogram-sepal-length.png"
    save_current_figure(output_dir / filename)

    return {
        "filename": filename,
        "title": "Distribution of Sepal Length",
        "description": "Histogram showing the distribution of sepal length.",
    }


def create_distribution_views(df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    """Show one variable as a histogram, density curve, and ECDF."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    sns.histplot(data=df, x="sepal_length", bins=12, ax=axes[0])
    axes[0].set_title("Histogram")
    axes[0].set_ylabel("Count")

    sns.kdeplot(data=df, x="sepal_length", fill=True, ax=axes[1])
    axes[1].set_title("Density")
    axes[1].set_ylabel("Density")

    sns.ecdfplot(data=df, x="sepal_length", ax=axes[2])
    axes[2].set_title("ECDF")
    axes[2].set_ylabel("Proportion at or below")

    fig.suptitle("Three Views of Sepal Length", y=1.03)

    filename = "distribution-views-sepal-length.png"
    save_current_figure(output_dir / filename)

    return {
        "filename": filename,
        "title": "Three Views of Sepal Length",
        "description": "Histogram, density, and ECDF views of the same variable.",
    }


def create_boxplot_sepal_length(df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(8, 5.5))

    sns.boxplot(
        data=df,
        x="species",
        y="sepal_length",
        ax=ax,
    )

    ax.set_title("Sepal Length by Species")
    ax.set_xlabel("Species")
    ax.set_ylabel("Sepal Length")

    filename = "boxplot-sepal-length-by-species.png"
    save_current_figure(output_dir / filename)

    return {
        "filename": filename,
        "title": "Sepal Length by Species",
        "description": "Boxplot comparing sepal length across species.",
    }


def create_scatterplot(df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(8, 5.5))

    sns.scatterplot(
        data=df,
        x="sepal_length",
        y="petal_length",
        hue="species",
        s=70,
        alpha=0.8,
        ax=ax,
    )

    ax.set_title("Sepal Length vs Petal Length")
    ax.set_xlabel("Sepal Length")
    ax.set_ylabel("Petal Length")

    filename = "scatter-sepal-length-vs-petal-length.png"
    save_current_figure(output_dir / filename)

    return {
        "filename": filename,
        "title": "Sepal Length vs Petal Length",
        "description": "Scatter plot showing the relationship between sepal length and petal length by species.",
    }


def create_grouped_histogram(df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(8, 5.5))

    sns.histplot(
        data=df,
        x="petal_length",
        hue="species",
        bins=15,
        kde=True,
        ax=ax,
    )

    ax.set_title("Petal Length Distribution by Species")
    ax.set_xlabel("Petal Length")
    ax.set_ylabel("Count")

    filename = "histogram-petal-length-by-species.png"
    save_current_figure(output_dir / filename)

    return {
        "filename": filename,
        "title": "Petal Length Distribution by Species",
        "description": "Grouped histogram comparing petal length distributions across species.",
    }


def create_small_multiples(df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    """Create one petal-length distribution panel per species."""
    g = sns.displot(
        data=df,
        x="petal_length",
        col="species",
        col_wrap=3,
        bins=10,
        kde=True,
        height=3.5,
        aspect=1,
    )

    g.set_axis_labels("Petal Length", "Count")
    g.set_titles("{col_name}")
    g.fig.suptitle("Petal Length Distribution within Each Species", y=1.05)

    filename = "small-multiples-petal-length-by-species.png"
    output_path = output_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(g.fig)

    return {
        "filename": filename,
        "title": "Petal Length within Each Species",
        "description": "Small-multiple histograms of within-species distributions.",
    }


def create_petal_area_boxplot(df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(8, 5.5))

    sns.boxplot(
        data=df,
        x="species",
        y="petal_area",
        ax=ax,
    )

    sns.stripplot(
        data=df,
        x="species",
        y="petal_area",
        color="black",
        alpha=0.55,
        size=4,
        jitter=0.22,
        ax=ax,
    )

    ax.set_title("Petal Area by Species")
    ax.set_xlabel("Species")
    ax.set_ylabel("Petal Area")

    filename = "boxplot-petal-area-by-species.png"
    save_current_figure(output_dir / filename)

    return {
        "filename": filename,
        "title": "Petal Area by Species",
        "description": "Boxplot with observed points comparing derived petal area across species.",
    }


def create_petal_area_comparison(df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    """Compare boxplot and violin-plot views of petal area."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    sns.boxplot(data=df, x="species", y="petal_area", ax=axes[0])
    axes[0].set_title("Boxplot")
    axes[0].set_xlabel("Species")
    axes[0].set_ylabel("Petal Area")

    sns.violinplot(
        data=df,
        x="species",
        y="petal_area",
        inner="quartile",
        cut=0,
        ax=axes[1],
    )
    sns.stripplot(
        data=df,
        x="species",
        y="petal_area",
        color="black",
        alpha=0.4,
        size=3,
        jitter=0.16,
        ax=axes[1],
    )
    axes[1].set_title("Violin Plot with Observations")
    axes[1].set_xlabel("Species")
    axes[1].set_ylabel("")

    fig.suptitle("Two Views of Petal Area by Species", y=1.02)

    filename = "comparison-petal-area-by-species.png"
    save_current_figure(output_dir / filename)

    return {
        "filename": filename,
        "title": "Two Views of Petal Area by Species",
        "description": "Side-by-side boxplot and violin plot with observations.",
    }


def create_scatter_trends(df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    """Create a scatter plot with a separate fitted trend for each species."""
    g = sns.lmplot(
        data=df,
        x="sepal_length",
        y="petal_length",
        hue="species",
        height=5.5,
        aspect=1.35,
        scatter_kws={"s": 55, "alpha": 0.75},
        ci=None,
    )

    g.set_axis_labels("Sepal Length", "Petal Length")
    g.fig.suptitle(
        "Within-Species Trends: Sepal Length vs Petal Length",
        y=1.03,
    )

    filename = "scatter-trends-by-species.png"
    output_path = output_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(g.fig)

    return {
        "filename": filename,
        "title": "Within-Species Scatter-Plot Trends",
        "description": "Scatter plot with a separate fitted trend for each species.",
    }


def create_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    """Create an annotated correlation heatmap for numeric features."""
    numeric_columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "petal_area",
    ]
    correlations = df[numeric_columns].corr()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        correlations,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
    )
    ax.set_title("Correlation among Iris Numeric Features")

    filename = "correlation-heatmap.png"
    save_current_figure(output_dir / filename)

    return {
        "filename": filename,
        "title": "Correlation among Iris Numeric Features",
        "description": "Annotated heatmap of correlations among numeric features.",
    }


def create_pairplot(df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "petal_area",
        "species",
    ]

    g = sns.pairplot(
        df[columns],
        hue="species",
        corner=True,
        plot_kws={"alpha": 0.7},
    )

    g.fig.suptitle("Iris — Pairwise Relationships by Species", y=1.02)

    filename = "pairplot-iris-by-species.png"
    output_path = output_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(g.fig)

    return {
        "filename": filename,
        "title": "Iris Pairwise Relationships by Species",
        "description": "Pairplot showing multivariate relationships among Iris measurements by species.",
    }


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

    sns.set_theme(style="whitegrid", context="notebook")

    df = read_table(input_path)
    validate_required_columns(df)

    figure_records = [
        create_histogram_sepal_length(df, output_dir),
        create_distribution_views(df, output_dir),
        create_boxplot_sepal_length(df, output_dir),
        create_scatterplot(df, output_dir),
        create_grouped_histogram(df, output_dir),
        create_small_multiples(df, output_dir),
        create_petal_area_boxplot(df, output_dir),
        create_petal_area_comparison(df, output_dir),
        create_scatter_trends(df, output_dir),
        create_correlation_heatmap(df, output_dir),
        create_pairplot(df, output_dir),
    ]

    figure_index = pd.DataFrame(figure_records)
    figure_index.to_csv(output_dir / "figure-index.tsv", sep="\t", index=False)

    print("Visualization outputs complete.")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Figure index: {output_dir / 'figure-index.tsv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
