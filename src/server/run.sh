#!/usr/bin/env bash
# Quick runner for development
set -euo pipefail

# create venv if missing (optional)
python3 -m venv .venv || true
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Run uvicorn
uvicorn main:app --host 127.0.0.1 --port 8000 --reload