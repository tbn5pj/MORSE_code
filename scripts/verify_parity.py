#!/usr/bin/env python3
"""Run the release test suite and print its fail-closed parity gates."""
import subprocess
import sys

raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "-q", "tests"]))

