#!/usr/bin/env bash
set -euo pipefail

# CDI Data Science Foundations System
# Environment setup script
#
# Before running this script, make sure requirements.txt exists
# in the project root. This file should be prepared manually and
# should list the Python packages required by the project.
#
# Example requirements.txt:
# pandas
# numpy
# matplotlib
# seaborn
# scikit-learn
# ipykernel

PY="${PYTHON:-python3}"

echo "CDI Data Science Foundations System"
echo "Setting up project environment..."
echo ""

if [ ! -f requirements.txt ]; then
  echo "ERROR: requirements.txt not found."
  echo ""
  echo "Create requirements.txt in the project root with:"
  echo ""
  echo "pandas"
  echo "numpy"
  echo "matplotlib"
  echo "seaborn"
  echo "scikit-learn"
  echo "ipykernel"
  echo ""
  exit 1
fi

echo "Using Python:"
$PY --version

echo ""
echo "Creating virtual environment..."
$PY -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing required packages from requirements.txt..."
python -m pip install -r requirements.txt

echo "Registering Jupyter kernel..."
python -m ipykernel install \
  --user \
  --name data-science \
  --display-name "CDI Data Science"

echo ""
echo "Environment ready."
echo "Activate with:"
echo "source .venv/bin/activate"