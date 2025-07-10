#!/usr/bin/env python
"""
Convenience script to run the time series experiments.
"""

import os
import sys

# Add the time-series directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import and run the time series runner
from runner import main

if __name__ == "__main__":
    main()