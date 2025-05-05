#!/usr/bin/env python
"""
Convenience script to run the time series experiments.
"""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

# Import and run the time series runner
from time_series.runner import main

if __name__ == "__main__":
    main()