#!/usr/bin/env python3
import sys
from run_morse import main

if "--K" not in sys.argv:
    sys.argv.extend(["--K", "1"])
main()

