#!/usr/bin/env bash
set -euo pipefail

# create virtual env
python3 -m venv .venv

# activate virutal env
source .venv/bin/activate

# install ops
pip install -e .

