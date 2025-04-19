# Fourier Analysis Networks (FANs)

This repository contains implementations of Fourier Analysis Networks (FANs) for time series forecasting and signal analysis. FANs introduce Fourier principles into neural networks to better model periodic patterns and complex signals.

## Overview

Fourier Analysis Networks leverage trigonometric functions to efficiently represent periodic patterns in data. This approach is particularly effective for time series data that contains cyclical components. Our implementation includes several variants of the FAN architecture with different capabilities:

- **FAN**: Basic Fourier Analysis Network implementation
- **FANGated**: Adds gating mechanism to balance Fourier and nonlinear components
- **FANPhaseOffsetModel**: Incorporates phase offsets for improved flexibility
- **FANPhaseOffsetModelZero**: Uses zero initialization for phase offsets
- **FANAmplitudePhaseModel**: Controls both amplitude and phase of Fourier components
- **FANMultiFrequencyModel**: Leverages multiple frequency scales
- **FANUnconstrainedBasisModel**: Uses learned basis transformations

## Key Features

- **Efficient Signal Representation**: Better at capturing periodic patterns with fewer parameters
- **Flexible Architecture**: Multiple model variants to handle different signal types
- **Fast Convergence**: Learns complex patterns with faster training times
- **Comprehensive Benchmarking**: Automated performance and efficiency metrics
- **Visualization Tools**: Detailed plots for model comparison and analysis

## Performance

Our benchmarks demonstrate that FAN models can outperform traditional approaches (MLP, Transformers) on various time series tasks, especially for data with strong periodic components. The repository includes experiments on:

- Simple and complex sine wave combinations
- Modulated signals with varying noise levels
- Real-world time series datasets

Performance metrics include MSE, RMSE, MAE, R², as well as model efficiency metrics (parameters, FLOPs, training/inference speed).

## Implementation

The repository is structured to support experimentation with different FAN variants:

- Configuration-driven experiments via YAML files
- MLflow integration for experiment tracking
- Comprehensive visualization of model performance
- Support for various hardware (CPU, CUDA, MPS)
- Input feature normalization for numerical stability across all platforms

## Getting Started

1. Clone the repository
2. Install the required dependencies
3. Configure experiments using the YAML files in the `configs/` directory
4. Run experiments using `python runner.py --config configs/config.yml`

## Citation

This implementation is based on the paper "FAN: Fourier Analysis Networks" (2024) [arXiv:2410.02675](https://huggingface.co/papers/2410.02675)