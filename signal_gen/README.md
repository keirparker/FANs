# Signal Generation Module for FANs (Fourier Activation Networks)

This module provides utilities for generating synthetic periodic data and evaluating various neural network models, with a particular focus on Fourier Activation Networks (FANs) and their variants.

## Overview

The signal generation module generates synthetic time series data with varying complexity to evaluate the performance of different neural network architectures.

## Model Variants

- **FAN**: Standard Fourier Activation Network
- **FANGated**: FAN with learnable gating mechanism
- **FANPhaseOffsetModel**: FAN with phase offset initialization (π/4)
- **FANPhaseOffsetModelGated**: Phase offset FAN with learnable gating mechanism
- **MLP**: Standard multilayer perceptron (baseline)

## Usage

To run signal generation experiments:

```bash
# From the project root directory
python -m signal_gen.runner
```

Or with the runner script:

```bash
python run_signal_gen.py
```

## Configuration

The module uses a YAML configuration file located at `configs/config.yml`. Key configuration options include:

- `experiment_name`: Name for the experiment run (used in MLflow tracking)
- `models_to_run`: List of model variants to evaluate
- `datasets_to_run`: List of signal types to generate
- `data_versions`: Data transformation variants (original, noisy, sparse)
- `hyperparameters`: Training parameters, hardware acceleration options, etc.

## Results

Experiment results are stored in:

- `mlruns/`: MLflow tracking data
- `logs/`: Experiment logs
- `plots/`: Generated visualization plots
- `results/`: Summary tables and run IDs

## Signal Types

- `sin`: Simple sine wave
- `mod`: Modulo function
- `complex_1` through `complex_6`: Various complex periodic functions
- `increasing_amp_freq`: Signal with progressively increasing amplitude and frequency
- `gradually_increasing_frequency`: Wave with slowly increasing frequency
- `gradually_increasing_amplitude`: Wave with slowly increasing amplitude
- `combined_freq_amp_modulation`: Signal combining both frequency and amplitude modulation