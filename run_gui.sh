#!/usr/bin/env bash
# Launch the registration GUI on Linux/macOS.
# Requires a Python with Tk support (python3-tk) plus the packages in
# requirements.txt.
cd "$(dirname "$0")"
PYEXE="${PYTHON:-python3}"
exec "$PYEXE" xrf_registration_gui.py "$@"
