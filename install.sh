#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
export PYTHONIOENCODING="utf-8"

PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "[ERROR] Python 3.10+ not found."
  exit 1
fi

exec "$PYTHON_BIN" launcher.py install "$@"
