#!/usr/bin/env python
"""
Simple script to run the signal_gen module from the root directory.
This avoids import issues by ensuring the project root is in the Python path.
"""

import os
import sys

# Make sure we're running from the project root
if not os.path.exists('signal_gen'):
    print("Error: This script must be run from the project root directory.")
    print("Current directory:", os.getcwd())
    sys.exit(1)

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath('.'))

# Import and run the main function from signal_gen
from signal_gen.runner import main

if __name__ == "__main__":
    main()