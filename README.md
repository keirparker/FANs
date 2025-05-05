# Fourier Activation Networks (FANs) with Phase Offset Optimization

This repository presents **Fourier Activation Networks (FANs)** with a novel **π/4 Phase Offset** initialization technique that dramatically improves convergence speed and parameter efficiency in neural networks for time series forecasting.

<div align="center">
  <img src="images/Figure 1.png" alt="Comparison of MLP Layer vs. FAN Layer" width="700"/>
  <p><i>Figure 1: Comparison between a standard MLP layer and a FAN layer. FAN layers incorporate sine and cosine activations with learnable weights to explicitly model periodic patterns.</i></p>
</div>

## Key Results

Our research established both theoretical foundations using synthetic data and practical benefits using real-world time series data:

### Synthetic Data Results
- **Parameter Efficiency**: 85% reduction in parameters while maintaining similar approximation quality
- **Improved Accuracy**: 58.4% error reduction when using the same number of parameters
- **Theoretical Foundation**: Mathematical proof of how phase offsets create beneficial interference patterns

### Real-World Time Series Results
- **2x Faster Convergence**: Phase Offset models reach optimal performance in half the training time
- **State-of-the-Art Performance**: Hybrid FAN models outperform standard approaches across multiple datasets
- **40-80% Parameter Reduction**: Maintain or improve accuracy with significantly fewer parameters

## Repository Structure

Our approach was validated on two distinct types of data: controlled synthetic signals and real-world time series datasets.

### 1. Synthetic Signal Generation (`signal_gen/`)

This module focuses on **controlled synthetic signals** with known frequency components to establish theoretical foundations:

- Generates custom periodic signals with varying complexity
- Tests different FAN architectures on these controlled signals
- Demonstrates mathematical principles of wave superposition and interference
- Provides clear visual proof of the parameter efficiency benefits
- Isolates the effect of phase offsets in a controlled environment

```bash
python -m signal_gen.run_signal_gen
```

### 2. Real-World Time Series Forecasting (`time-series/`)

This module applies FAN architectures to **real-world multivariate time series data** to demonstrate practical benefits:

- Works with complex, noisy, multivariate time series datasets (ETTh1, ETTh2, electricity, solar-energy, traffic)
- Implements traditional and FAN-based forecasting models
- Introduces novel Hybrid Phase FAN models that transition from offset to standard behavior
- Integrates with transformer architectures for advanced sequence modeling
- Benchmarks against state-of-the-art time series models

```bash
python -m time_series.run_time_series
```

## Synthetic Data Experiments

Our synthetic data experiments establish the theoretical foundations of FAN models and phase offset benefits.

### Wave Superposition and Interference Effects

FAN models leverage wave physics principles to create precise constructive and destructive interference patterns, enabling more efficient approximation of complex signals with fewer parameters.

<div align="center">
  <img src="images/Figure 3.png" alt="Wave Superposition and Interference Effects" width="700"/>
  <p><i>Figure 3: Wave superposition and interference effects in FAN models. Phase offsets allow more efficient approximation with fewer terms (bottom right).</i></p>
</div>

### Parameter Efficiency Analysis

Phase offset parameters enable dramatic parameter efficiency in FAN models, with consistent reductions of 83-89% compared to standard approaches without sacrificing accuracy.
- Input feature normalization for numerical stability across all platforms

<div align="center">
  <img src="images/Figure 6.png" alt="Phase Offsets Enable Parameter Efficiency" width="700"/>
  <p><i>Figure 6: Phase offsets enable significant parameter efficiency. Left: 58.4% error reduction with the same parameter count. Right: 85% parameter reduction with similar quality.</i></p>
</div>

## Real-World Time Series Experiments

Our real-world experiments validate the practical benefits of FAN models on challenging time series forecasting tasks.

### Model Performance Comparison

<div align="center">
  <img src="images/Figure 4.png" alt="Model Family Error Comparison" width="700"/>
  <p><i>Figure 4: Model family error comparison showing FANPhaseOffset models achieve both the lowest error (1.099 MAE) and highest convergence efficiency (4.82).</i></p>
</div>

### Multi-Dataset Performance Evaluation

Our models were evaluated across multiple real-world time series datasets with detailed performance metrics:

<div align="center">
  <img src="images/Table 8.png" alt="Performance Comparison Across Datasets" width="800"/>
  <p><i>Table 8: Full performance comparison across datasets showing MSE and MAE metrics for ETTh1, ETTh2, electricity, solar-energy, and traffic datasets.</i></p>
</div>

### Comprehensive Efficiency Analysis

<div align="center">
  <img src="images/Figure 5.png" alt="Model Error Analysis" width="700"/>
  <p><i>Figure 5: Comprehensive model error analysis on real-world datasets. FANPhaseOffset achieves the highest overall efficiency score (155.82).</i></p>
</div>

### Efficiency vs. Accuracy Trade-offs

When comparing to standard transformer models, our FAN-based approaches offer substantial efficiency gains on real-world datasets:

<div align="center">
  <img src="images/Table 9.png" alt="Model Efficiency Analysis" width="800"/>
  <p><i>Table 9: Efficiency gain vs. accuracy trade-off analysis on real-world time series, using Transformer as baseline.</i></p>
</div>

### Computational Resource Requirements

<div align="center">
  <img src="images/Table 10.png" alt="Computational Resources Analysis" width="800"/>
  <p><i>Table 10: Detailed computational requirements for real-world time series forecasting, showing parameters, inference time, FLOPs, and training time.</i></p>
</div>

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. For synthetic signal experiments: `python -m signal_gen.run_signal_gen`
3. For time series forecasting: `python -m time_series.run_time_series`

## Key Advantages of Phase Offset Approach

1. **Parameter Efficiency**: Represent complex functions with up to 85% fewer parameters
2. **Faster Convergence**: Reach optimal performance 1.5-2x faster than standard models
3. **Early Learning**: Better starting point for modeling periodic patterns
4. **Wave Physics**: Leverages principles of wave superposition for more efficient signal representation
5. **Hybrid Models**: Our novel hybrid phase transition approach combines fast initial learning with stable final performance

## Dependencies

- Python 3.10
- PyTorch
- NumPy
- MLflow (for experiment tracking)
- pandas, Matplotlib, Loguru, scikit-learn