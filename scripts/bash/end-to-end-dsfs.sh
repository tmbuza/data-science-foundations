#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# CDI Data Science Foundations System
# End-to-End Workflow
#
# Run from the project root:
#
#   bash scripts/bash/end-to-end-dsfs.sh
#
# This script runs the complete Foundations workflow:
# inspect → clean → wrangle → visualize → summarize
###############################################################################

echo "========================================"
echo "CDI Data Science Foundations System"
echo "End-to-End Workflow"
echo "========================================"
echo ""

###############################################################################
# Check required input
###############################################################################

if [ ! -f "data/iris.csv" ]; then
  echo "ERROR: data/iris.csv not found."
  echo ""
  echo "Create data/iris.csv first using the example dataset step in Chapter 02."
  exit 1
fi

###############################################################################
# Check required scripts
###############################################################################

required_scripts=(
  "scripts/python/inspect_table.py"
  "scripts/python/clean_example_data.py"
  "scripts/python/wrangle_example_data.py"
  "scripts/python/plot_example_data.py"
  "scripts/python/summarize_table.py"
)

for script_path in "${required_scripts[@]}"; do
  if [ ! -f "$script_path" ]; then
    echo "ERROR: required script not found: $script_path"
    exit 1
  fi
done

###############################################################################
# Create output directories
###############################################################################

mkdir -p results/inspection
mkdir -p results/cleaning
mkdir -p results/wrangling
mkdir -p results/figures
mkdir -p results/summary

###############################################################################
# 01. Inspect input table
###############################################################################

echo ""
echo "Step 1/5: Inspecting input table..."
python scripts/python/inspect_table.py   data/iris.csv   results/inspection

###############################################################################
# 02. Clean dataset
###############################################################################

echo ""
echo "Step 2/5: Cleaning dataset..."
python scripts/python/clean_example_data.py   data/iris.csv   data/iris_clean.csv   results/cleaning

###############################################################################
# 03. Wrangle cleaned dataset
###############################################################################

echo ""
echo "Step 3/5: Wrangling cleaned dataset..."
python scripts/python/wrangle_example_data.py   data/iris_clean.csv   data/iris_wrangled.csv   results/wrangling

###############################################################################
# 04. Create visualization outputs
###############################################################################

echo ""
echo "Step 4/5: Creating visualization outputs..."
python scripts/python/plot_example_data.py   data/iris_wrangled.csv   results/figures

###############################################################################
# 05. Create summary outputs and insights report
###############################################################################

echo ""
echo "Step 5/5: Creating summary outputs..."
python scripts/python/summarize_table.py   data/iris_wrangled.csv   results/summary

###############################################################################
# Completion summary
###############################################################################

echo ""
echo "========================================"
echo "End-to-end DSFS workflow complete."
echo "========================================"
echo ""
echo "Core outputs:"
echo "  data/iris_clean.csv"
echo "  data/iris_wrangled.csv"
echo ""
echo "Result directories:"
echo "  results/inspection"
echo "  results/cleaning"
echo "  results/wrangling"
echo "  results/figures"
echo "  results/summary"
echo ""
echo "Key report:"
echo "  results/summary/analysis-insights.md"
echo ""
