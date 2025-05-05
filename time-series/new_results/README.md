# Time Series Benchmarking Tools

This directory contains scripts for benchmarking FAN model variants:

- `benchmark_models.py`: General performance benchmarking across models and datasets
- `phase_offset_benchmark.py`: Specific benchmarking for phase offset approaches
- `convergence_benchmark.py`: Convergence speed analysis and comparison

Run any script with Python to perform the corresponding benchmark:

```bash
python benchmark_models.py  # Full model comparison
python phase_offset_benchmark.py  # Phase offset comparison
python convergence_benchmark.py  # Convergence analysis
```

Results will be saved in the same directory as CSV files and visualizations.