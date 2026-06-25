# Data Science Foundations System

A reusable Python workflow for tidy-table analysis, interpretation, and reproducible reporting across CDI pathways.

Part of the **Complex Data Insights (CDI)** ecosystem  
**Data Science Pathway — Foundations System (v1.0.0)**

---

## System role

The **CDI Data Science Foundations System** is a reusable parent system for working with tidy, analysis-ready tables.

It supports the common analytical workflow used across CDI pathways once domain-specific processing has produced structured tables.

```text
Tidy analysis-ready table
        ↓
Data Science Foundations System
        ↓
inspection, cleaning, wrangling, visualization, summary, interpretation
```

This system can support:

- Omics result tables
- Clinical and medical cohort tables
- AI evaluation tables
- Decision-support tables
- Other structured analytical datasets

---

## What this project is

This repository contains a complete, runnable analytical workflow.

The Quarto chapter files (`.qmd`) are the source of the guide.  
The Python scripts in `scripts/python/` implement reusable workflow steps.  
Running the workflow produces structured outputs in `data/` and `results/`.

The system teaches and implements:

- data inspection
- data cleaning
- data wrangling
- visualization
- summary statistics
- evidence-based interpretation
- transition from analysis to modeling readiness

---

## Workflow

```text
data/iris.csv
        ↓
inspect
        ↓
data/iris_clean.csv
        ↓
clean
        ↓
data/iris_wrangled.csv
        ↓
wrangle
        ↓
figures
        ↓
summary tables
        ↓
analysis-insights.md
```

Core workflow scripts:

```text
scripts/python/
├── inspect_table.py
├── clean_example_data.py
├── wrangle_example_data.py
├── plot_example_data.py
└── summarize_table.py
```

---

## How to run

Clone the repository:

```bash
git clone https://github.com/tmbuza/data-science.git
cd data-science
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

Run the full Data Science Foundations workflow:

```bash
bash scripts/bash/end-to-end-dsfs.sh
```

Render the Quarto book:

```bash
quarto render
```

Then open:

```text
docs/index.html
```

---

## End-to-end workflow

The end-to-end script runs the complete Foundations System:

```bash
bash scripts/bash/end-to-end-dsfs.sh
```

It executes:

```text
inspect → clean → wrangle → visualize → summarize
```

Expected core outputs:

```text
data/
├── iris.csv
├── iris_clean.csv
└── iris_wrangled.csv

results/
├── inspection/
├── cleaning/
├── wrangling/
├── figures/
└── summary/
```

Key interpretation output:

```text
results/summary/analysis-insights.md
```

---

## Data

The example workflow uses the Iris dataset.

The working input table is stored as:

```text
data/iris.csv
```

No external data downloads are required for the core workflow.

---

## Project structure

```text
data-science/
├── index.qmd
├── 00-preface.qmd
├── 01-setting-up-environment.qmd
├── 02-load-and-explore-dataset.qmd
├── 03-data-cleaning-and-preparation.qmd
├── 04-data-wrangling-basics.qmd
├── 05-visualization-basics.qmd
├── 06-summary-statistics-and-insights.qmd
├── 07-from-analysis-to-ml.qmd
├── 99-complete-free-track.qmd
├── 999-appendix.qmd
├── 999-references.qmd
├── data/
├── results/
├── assets/
├── scripts/
│   ├── bash/
│   └── python/
├── docs/
├── library/
├── _quarto.yml
├── README.md
└── requirements.txt
```

---

## CDI pathway connection

This system sits inside the **CDI Data Science Pathway**, but it is designed to support the wider CDI ecosystem.

```text
CDI Ecosystem
│
├── Data Science Pathway
│   └── Data Science Foundations System
│
├── Omics Pathway
│   └── calls this system after tidy result tables
│
├── Clinical & Medical Data Pathway
│   └── calls this system after clean cohort tables
│
└── AI, Thinking & Decision Pathway
    └── calls this system after evaluation or decision tables
```

The system does not replace pathway-specific processing.

Instead, it provides a shared analytical layer once structured tables are ready.

---

## Reproducibility

This project is designed to be:

- environment-controlled using `.venv`
- dependency-defined using `requirements.txt`
- executable through reusable Python scripts
- renderable through Quarto
- structured with source files separated from generated outputs
- suitable for reuse across CDI systems

The workflow has been tested with:

```bash
bash scripts/bash/end-to-end-dsfs.sh
quarto render
```

---

## Version

**v1.0.0**

System status:

```text
Data Science Foundations System complete
End-to-end workflow validated
Quarto book rendered successfully
```

---

## Next step

This system prepares learners and CDI pathways for the next layer:

```text
Applied Data Science System
```

The Applied system extends the foundation into:

- feature engineering
- model building
- model evaluation
- model interpretation
- decision-making
- responsible use

---

## License

See `LICENSE` for details.
