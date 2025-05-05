# Time Series Forecasting with Fourier Activation Networks (FANs)

This module provides a framework for time series forecasting using Fourier Activation Networks (FANs) and their variants, with a focus on comparing phase offset initialization techniques and hybrid phase transition approaches.

## Overview

The time series forecasting module implements various neural network architectures designed for accurate time series prediction.

## Key Models

- **FANForecaster**: Standard FAN model for time series forecasting
- **FANGatedForecaster**: FAN with learnable gating mechanism
- **PhaseOffsetForecaster**: FAN with phase offset initialization (π/4)
- **HybridPhaseFANForecaster**: Novel model that transitions from phase offset to standard FAN during training
- **Transformer** variants of all the above models

## Datasets

The framework supports popular time series benchmarking datasets:
- ETTh1, ETTh2 (Electricity Transformer Temperature) 
- Electricity consumption
- Traffic occupancy rates
- Solar energy production

## Getting Started

### Data Setup

Raw datasets are provided in the `data/` directory, including ETTh1 and ETTh2, which are sufficient to run the default experiments. 

To automatically process the data or download additional datasets:

```bash
# Process provided datasets and download any missing ones
python -m time_series.data.download_data
```

This script will:
- Process existing raw data files
- Download and prepare any missing datasets
- Normalize and split data into train/val/test sets

### Running Experiments

To run time series experiments:

```bash
# From the project root directory
python run_time_series.py
# or
python -m time_series.runner
```

For specific benchmarks:

```bash
# Phase offset performance benchmark
python -m time_series.new_results.phase_offset_benchmark
```

## Configuration

The module uses a YAML configuration file (`config.yml`) with settings for:

- Model architecture selection
- Dataset selection
- Training hyperparameters
- Hardware acceleration options
- Phase decay parameters for hybrid models

## Key Features

1. **Phase Offset Initialization**: Models with π/4 phase offset consistently achieve faster convergence
2. **Hybrid Phase Transition**: Novel approach that combines fast initial convergence with stable final performance
3. **Transformer Integration**: FAN layers integrated into transformer architectures for better temporal modeling
4. **Comprehensive Benchmarking**: Tools for comparing models across multiple datasets and metrics