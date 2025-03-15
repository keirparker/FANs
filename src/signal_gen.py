import numpy as np
from loguru import logger

##############################################################################
# For signals like 'complex_5' that have special logic, we define a helper
##############################################################################
def sawtooth_wave(t, n):
    """Generate a single term of the sawtooth wave harmonic series."""
    return (t / np.pi) - np.floor(t / np.pi + 0.5)

def complex_5_fn(x: np.ndarray) -> np.ndarray:
    """
    For 'complex_5', sum up (1/n)*sawtooth_wave(n*x, n).
    """
    N = 5
    data = np.zeros_like(x)
    for n in range(1, N + 1):
        data += (1 / n) * sawtooth_wave(n * x, n)
    return data

def increasing_amp_freq_fn(x: np.ndarray) -> np.ndarray:
    """
    Generate a signal with progressively increasing amplitude and frequency
    that maintains phase continuity across any input range.
    """
    # Normalize x for consistent scaling
    x_centered = x / 50.0

    # Amplitude increases with |x|
    a_base = 0.2
    a_slope = 0.8
    amplitude = a_base + a_slope * np.abs(np.tanh(x_centered))

    # Base frequency component
    f_base = 0.5
    f_slope = 2.0

    # Use a proper phase accumulation approach
    # Integrate frequency to get phase (using a closed form for this specific case)
    # For a frequency function f(x) = f_base + f_slope*tanh(x), the phase is:
    phase = 2 * np.pi * (f_base * x_centered + f_slope * np.log(np.cosh(x_centered)))

    # Generate the signal with continuous phase
    signal = amplitude * np.sin(phase)

    # Apply consistent normalization
    signal = signal / 1.5

    return signal

def compressing_expanding_wave_fn(x: np.ndarray) -> np.ndarray:
    """
    Generate a wave that compresses and expands periodically like a sound wave.
    The wave's frequency changes smoothly in a sinusoidal pattern, creating
    many compression/expansion cycles within the training range.
    """
    # Normalize x for consistent scaling
    x_norm = x / 20.0
    
    # Base frequency and modulation parameters - more moderate values
    base_freq = 2.0
    freq_mod_depth = 0.5
    mod_rate = 0.5
    
    # Create a frequency modulation - frequency oscillates sinusoidally
    freq_mod = 1.0 + freq_mod_depth * np.sin(mod_rate * x_norm)
    
    # Calculate the phase by integrating the frequency
    phase = 2 * np.pi * (base_freq * x_norm - (freq_mod_depth/mod_rate) * np.cos(mod_rate * x_norm))
    
    # Generate the signal with a constant amplitude
    signal = np.sin(phase)
    
    # Clip to ensure there are no extreme values
    signal = np.clip(signal, -1.0, 1.0)
    
    return signal

def increasing_decreasing_amp_fn(x: np.ndarray) -> np.ndarray:
    """
    Generate a wave with an amplitude that increases then decreases repeatedly.
    Instead of a single Gaussian envelope, we use a periodic amplitude modulation.
    """
    # Normalize x for consistent scaling
    x_norm = x / 30.0
    
    # Base frequency - moderate value
    base_freq = 3.0
    
    # Compute a periodic envelope that rises and falls multiple times
    env_freq = 0.3  # Controls how many amplitude cycles occur
    amplitude = 0.5 + 0.5 * np.cos(2 * np.pi * env_freq * x_norm)
    
    # Generate a constant-frequency signal
    phase = 2 * np.pi * base_freq * x_norm
    
    # Apply the amplitude envelope to the signal
    signal = amplitude * np.sin(phase)
    
    # Clip to ensure there are no extreme values
    signal = np.clip(signal, -1.0, 1.0)
    
    return signal

def combined_compression_amp_fn(x: np.ndarray) -> np.ndarray:
    """
    Generate a wave that both compresses/expands and has increasing/decreasing amplitude.
    Both effects are periodic, creating multiple cycles within the training range.
    """
    # Normalize x for consistent scaling
    x_norm = x / 25.0
    
    # Amplitude modulation parameters - more stable values
    env_freq = 0.25
    amplitude = 0.5 + 0.5 * np.cos(2 * np.pi * env_freq * x_norm)
    
    # Frequency modulation parameters - more stable values
    base_freq = 2.0
    freq_mod_depth = 0.4
    mod_rate = 0.4
    
    # Create a frequency modulation - frequency oscillates sinusoidally
    freq_mod = 1.0 + freq_mod_depth * np.sin(mod_rate * x_norm)
    
    # Calculate the phase by integrating the frequency
    phase = 2 * np.pi * (base_freq * x_norm - (freq_mod_depth/mod_rate) * np.cos(mod_rate * x_norm))
    
    # Apply both the amplitude envelope and frequency modulation
    signal = amplitude * np.sin(phase)
    
    # Clip to ensure there are no extreme values
    signal = np.clip(signal, -1.0, 1.0)
    
    return signal


##############################################################################
# PERIODIC_SPECS holds all domain & data logic for each type.
# 'config' is an optional sub-dict for any extra metadata (batchsize, etc.).
##############################################################################
PERIODIC_SPECS = {
    "sin": {
        "period": 6,
        "domain_train": lambda p, ns: np.linspace(-10*p*np.pi, 10*p*np.pi, ns),
        "domain_test":  lambda p, ns: np.linspace(-25*p*np.pi, 25*p*np.pi, ns),
        "data_fn":      lambda t: np.sin(t),
        "config": {
            "batchsize": 32,
            "numepoch": 10000,
            "printepoch": 50,
            "lr": 1e-5,
            "wd": 0.01,
            "y_uper": 1.5,
            "y_lower": -1.5,
        }
    },
    "mod": {
        "period": 20,
        "domain_train": lambda p, ns: np.linspace(-10*p, 10*p, ns),
        "domain_test":  lambda p, ns: np.linspace(-25*p, 25*p, ns),
        "data_fn":      lambda t: np.mod(t, 5),
        "config": {
            "batchsize": 32,
            "numepoch": 10000,
            "printepoch": 50,
            "lr": 1e-5,
            "wd": 0.01,
            "y_uper": 10,
            "y_lower": -5,
        }
    },
    "complex_1": {
        "period": 4,
        "domain_train": lambda p, ns: np.linspace(-10*p, 10*p, ns),
        "domain_test":  lambda p, ns: np.linspace(-25*p, 25*p, ns),
        "data_fn":      lambda t: np.exp(np.sin(np.pi * t)**2 + np.cos(t) + np.mod(t, 3) - 1),
        "config": {
            "batchsize": 32,
            "numepoch": 10000,
            "printepoch": 50,
            "lr": 1e-5,
            "wd": 0.01,
            "y_uper": 20,
            "y_lower": -20,
        }
    },
    "complex_2": {
        "period": 4,
        "domain_train": lambda p, ns: np.linspace(-10*p, 10*p, ns),
        "domain_test":  lambda p, ns: np.linspace(-25*p, 25*p, ns),
        "data_fn":      lambda t: (1 + np.sin(t)) * np.sin(2 * t),
        "config": {
            "batchsize": 32,
            "numepoch": 10000,
            "printepoch": 50,
            "lr": 1e-5,
            "wd": 0.01,
            "y_uper": 4,
            "y_lower": -4,
        }
    },
    "complex_3": {
        "period": 4,
        "domain_train": lambda p, ns: np.linspace(-10*p, 10*p, ns),
        "domain_test":  lambda p, ns: np.linspace(-25*p, 25*p, ns),
        "data_fn":      lambda t: np.sin(t + np.sin(2 * t)),
        "config": {
            "batchsize": 32,
            "numepoch": 10000,
            "printepoch": 50,
            "lr": 1e-5,
            "wd": 0.01,
            "y_uper": 2,
            "y_lower": -2,
        }
    },
    "complex_4": {
        "period": 4,
        "domain_train": lambda p, ns: np.linspace(-10*p, 10*p, ns),
        "domain_test":  lambda p, ns: np.linspace(-25*p, 25*p, ns),
        "data_fn":      lambda t: np.sin(t)*(np.cos(2*t)**2) + np.cos(t)*(np.sin(3*t)**2),
        "config": {
            "batchsize": 32,
            "numepoch": 10000,
            "printepoch": 50,
            "lr": 1e-5,
            "wd": 0.01,
            "y_uper": 2,
            "y_lower": -2,
        }
    },
    "complex_5": {
        "period": 4,
        "domain_train": lambda p, ns: np.linspace(-10*p, 10*p, ns),
        "domain_test":  lambda p, ns: np.linspace(-25*p, 25*p, ns),
        "data_fn":      complex_5_fn,  # uses that helper
        "config": {
            "batchsize": 32,
            "numepoch": 10000,
            "printepoch": 50,
            "lr": 1e-5,
            "wd": 0.01,
            "y_uper": 1,
            "y_lower": -1,
        }
    },
    "complex_6": {
        "period": 4,
        "domain_train": lambda p, ns: np.linspace(-10*p, 10*p, ns),
        "domain_test":  lambda p, ns: np.linspace(-25*p, 25*p, ns),
        "data_fn":      lambda t: np.exp(np.sin(t)) / (1 + np.cos(2*t)**2),
        "config": {
            "batchsize": 32,
            "numepoch": 10000,
            "printepoch": 50,
            "lr": 1e-5,
            "wd": 0.01,
            "y_uper": 3,
            "y_lower": 0,
        }
    },
    "increasing_amp_freq": {
        "period": 10,
        "domain_train": lambda p, ns: np.linspace(-10*p, 10*p, ns),
        "domain_test":  lambda p, ns: np.linspace(-25*p, 25*p, ns),
        "data_fn":      increasing_amp_freq_fn,  # The new function handles train/test consistently
        "config": {
            "batchsize": 64,
            "numepoch": 5000,  # Reduced since convergence should be faster now
            "printepoch": 50,
            "lr": 1e-3,  # Increased further for faster convergence
            "wd": 0.0001,  # Reduced for less regularization
            "y_uper": 1.0,  # Adjusted for normalized output
            "y_lower": -1.0,
        }
    },
    "compressing_expanding_wave": {
        "period": 2,  # Smaller period = more repetitions
        "domain_train": lambda p, ns: np.linspace(-10*p, 10*p, ns),
        "domain_test":  lambda p, ns: np.linspace(-25*p, 25*p, ns),
        "data_fn":      compressing_expanding_wave_fn,
        "config": {
            "batchsize": 64,
            "numepoch": 5000,
            "printepoch": 50,
            "lr": 1e-4,  # More stable learning rate
            "wd": 0.001,  # Increased weight decay for better stability
            "y_uper": 1.0,
            "y_lower": -1.0,
        }
    },
    "increasing_decreasing_amp": {
        "period": 2,  # Smaller period = more repetitions
        "domain_train": lambda p, ns: np.linspace(-10*p, 10*p, ns),
        "domain_test":  lambda p, ns: np.linspace(-25*p, 25*p, ns),
        "data_fn":      increasing_decreasing_amp_fn,
        "config": {
            "batchsize": 64,
            "numepoch": 5000,
            "printepoch": 50,
            "lr": 1e-4,  # More stable learning rate
            "wd": 0.001,  # Increased weight decay for better stability
            "y_uper": 1.0,
            "y_lower": -1.0,
        }
    },
    "combined_compression_amp": {
        "period": 2,  # Smaller period = more repetitions
        "domain_train": lambda p, ns: np.linspace(-10*p, 10*p, ns),
        "domain_test":  lambda p, ns: np.linspace(-25*p, 25*p, ns),
        "data_fn":      combined_compression_amp_fn,
        "config": {
            "batchsize": 64,
            "numepoch": 5000,
            "printepoch": 50,
            "lr": 1e-4,  # More stable learning rate
            "wd": 0.001,  # Increased weight decay for better stability
            "y_uper": 1.0,
            "y_lower": -1.0,
        }
    }
}


##############################################################################
# 2) get_periodic_data: a single function with minimal repetition
#    It returns up to 6 values: 
#      (t_train, data_train, t_test, data_test, config, true_func)
##############################################################################
def get_periodic_data(periodic_type, num_train_samples=None, num_test_samples=None):
    """
    Retrieve training and test data for a given periodic type.

    Returns:
      t_train, data_train, t_test, data_test, config, true_func

    The 'true_func' is the same as 'data_fn' in the dictionary, which you can
    use to evaluate the ground-truth signal at any x-values for comparison.
    """
    if periodic_type not in PERIODIC_SPECS:
        logger.error(f"Unknown periodic_type: {periodic_type}")
        raise ValueError(f"Unsupported periodic_type: {periodic_type}")

    spec = PERIODIC_SPECS[periodic_type]
    period = spec["period"]
    domain_train = spec["domain_train"]
    domain_test = spec["domain_test"]
    data_fn = spec["data_fn"]
    config = spec["config"]  # e.g., batchsize, etc.

    # Decide how many samples if none provided
    if num_train_samples is None:
        num_train_samples = int(10000 * period)
    if num_test_samples is None:
        num_test_samples = 4000
        
    # Special handling for increasing_amp_freq to ensure train/test consistency
    if periodic_type == "increasing_amp_freq":
        # Use the same domain for training and testing for this specific function
        min_x, max_x = -50, 50
        
        # Generate training data - evenly spaced points
        t_train = np.linspace(min_x, max_x, num_train_samples)
        data_train = data_fn(t_train)
        
        # Generate test data - use more points in the same range for smooth testing
        # Also add some extrapolation regions for testing generalization
        t_test = np.concatenate([
            np.linspace(min_x - 20, min_x, num_test_samples // 4),  # Below training range
            np.linspace(min_x, max_x, num_test_samples // 2),       # Same as training range
            np.linspace(max_x, max_x + 20, num_test_samples // 4)   # Above training range
        ])
        data_test = data_fn(t_test)
    else:
        # Normal handling for other function types
        t_train = domain_train(period, num_train_samples)
        data_train = data_fn(t_train)
        
        t_test = domain_test(period, num_test_samples)
        data_test = data_fn(t_test)

    # The 'true_func' is just data_fn if you want to use it as ground truth
    true_func = data_fn

    return t_train, data_train, t_test, data_test, config, true_func

#
# if __name__ == "__main__":
#     """
#     This main section visualizes the new wave functions to verify their behavior.
#     Run this file directly to see the plots of these functions.
#     """
#     import matplotlib.pyplot as plt
#     from matplotlib.gridspec import GridSpec
#
#     # Set up the figure with GridSpec for nice layout
#     plt.figure(figsize=(15, 15))
#     gs = GridSpec(3, 1, height_ratios=[1, 1, 1])
#
#     # Generate data for all our new functions
#     x = np.linspace(-50, 50, 1000)
#     y_compress = compressing_expanding_wave_fn(x)
#     y_amp = increasing_decreasing_amp_fn(x)
#     y_combined = combined_compression_amp_fn(x)
#
#     # Plot 1: Compressing and expanding wave
#     ax1 = plt.subplot(gs[0])
#     ax1.plot(x, y_compress, 'b-', linewidth=1.5, alpha=0.8)
#     ax1.set_title("Compressing and Expanding Wave (Frequency Modulation)")
#     ax1.set_ylabel("Amplitude")
#     ax1.grid(True)
#
#     # Annotate compression and expansion regions
#     for i in range(-3, 4, 2):
#         # Mark compressed regions (high frequency)
#         region_start = i * 15 - 5
#         region_end = i * 15 + 5
#         section = (x > region_start) & (x < region_end)
#         if np.any(section):
#             ax1.axvspan(region_start, region_end, alpha=0.2, color='red')
#             ax1.text((region_start + region_end)/2, 0.8, "Compressed",
#                     horizontalalignment='center', fontsize=8,
#                     bbox=dict(facecolor='red', alpha=0.2))
#
#     for i in range(-2, 4, 2):
#         # Mark expanded regions (low frequency)
#         region_start = i * 15 - 5
#         region_end = i * 15 + 5
#         section = (x > region_start) & (x < region_end)
#         if np.any(section):
#             ax1.axvspan(region_start, region_end, alpha=0.2, color='green')
#             ax1.text((region_start + region_end)/2, -0.8, "Expanded",
#                     horizontalalignment='center', fontsize=8,
#                     bbox=dict(facecolor='green', alpha=0.2))
#
#     # Plot 2: Increasing then decreasing amplitude
#     ax2 = plt.subplot(gs[1])
#     ax2.plot(x, y_amp, 'g-', linewidth=1.5, alpha=0.8)
#     ax2.set_title("Increasing then Decreasing Amplitude Wave")
#     ax2.set_ylabel("Amplitude")
#     ax2.grid(True)
#
#     # Add envelope visualization
#     envelope = np.exp(-(x/30.0)**2 / 1.5)
#     ax2.plot(x, envelope, 'r--', linewidth=1.0, alpha=0.6, label="Amplitude Envelope")
#     ax2.plot(x, -envelope, 'r--', linewidth=1.0, alpha=0.6)
#     ax2.legend()
#
#     # Plot 3: Combined effect wave
#     ax3 = plt.subplot(gs[2])
#     ax3.plot(x, y_combined, 'purple', linewidth=1.5, alpha=0.8)
#     ax3.set_title("Combined Wave (Frequency Modulation + Amplitude Envelope)")
#     ax3.set_xlabel("x")
#     ax3.set_ylabel("Amplitude")
#     ax3.grid(True)
#
#     # Highlight envelope
#     combined_envelope = np.exp(-(x/25.0)**2 / 1.8)
#     ax3.plot(x, combined_envelope, 'r--', linewidth=1.0, alpha=0.6, label="Amplitude Envelope")
#     ax3.plot(x, -combined_envelope, 'r--', linewidth=1.0, alpha=0.6)
#
#     # Mark compression/expansion regions
#     for i in range(-1, 2, 1):
#         region_mid = i * 15
#         if i % 2 == 0:
#             # Compressed region
#             region_start = region_mid - 7
#             region_end = region_mid + 7
#             if region_start > -50 and region_end < 50:
#                 ax3.axvspan(region_start, region_end, alpha=0.1, color='blue', label="Compressed" if i == 0 else "")
#         else:
#             # Expanded region
#             region_start = region_mid - 7
#             region_end = region_mid + 7
#             if region_start > -50 and region_end < 50:
#                 ax3.axvspan(region_start, region_end, alpha=0.1, color='green', label="Expanded" if i == 1 else "")
#
#     ax3.legend()
#
#     plt.tight_layout()
#     plt.savefig("new_wave_functions.png")
#
#     # Create comparison visualization of all three waves
#     plt.figure(figsize=(15, 10))
#
#     # Plot all three waves on the same axes for comparison
#     plt.subplot(2, 1, 1)
#     plt.plot(x, y_compress, 'b-', linewidth=1.0, alpha=0.7, label="Compressing/Expanding Wave")
#     plt.plot(x, y_amp, 'g-', linewidth=1.0, alpha=0.7, label="Increasing/Decreasing Amplitude")
#     plt.plot(x, y_combined, 'purple', linewidth=1.0, alpha=0.7, label="Combined Effects")
#     plt.title("Comparison of All Three Wave Types")
#     plt.ylabel("Amplitude")
#     plt.legend()
#     plt.grid(True)
#
#     # Zoomed section to show detail
#     plt.subplot(2, 1, 2)
#     section = (x > -20) & (x < 20)
#     plt.plot(x[section], y_compress[section], 'b-', linewidth=1.5, alpha=0.7, label="Compressing/Expanding Wave")
#     plt.plot(x[section], y_amp[section], 'g-', linewidth=1.5, alpha=0.7, label="Increasing/Decreasing Amplitude")
#     plt.plot(x[section], y_combined[section], 'purple', linewidth=1.5, alpha=0.7, label="Combined Effects")
#     plt.title("Zoomed Section of Wave Comparison")
#     plt.xlabel("x")
#     plt.ylabel("Amplitude")
#     plt.legend()
#     plt.grid(True)
#
#     plt.tight_layout()
#     plt.savefig("wave_comparison.png")
#
#     # Generate and plot training/test data for each new wave type
#     wave_types = [
#         "compressing_expanding_wave",
#         "increasing_decreasing_amp",
#         "combined_compression_amp"
#     ]
#
#     plt.figure(figsize=(15, 12))
#
#     for i, wave_type in enumerate(wave_types):
#         # Get training and test data
#         t_train, data_train, t_test, data_test, config, true_func = get_periodic_data(
#             wave_type, num_train_samples=1000, num_test_samples=500
#         )
#
#         # Plot training and testing data
#         plt.subplot(3, 1, i+1)
#         plt.plot(t_train, data_train, 'b-', linewidth=0.5, alpha=0.7, label="Training Data")
#         plt.plot(t_test, data_test, 'r-', linewidth=0.5, alpha=0.7, label="Testing Data")
#
#         # Mark training region boundaries
#         train_min, train_max = np.min(t_train), np.max(t_train)
#         plt.axvline(x=train_min, color='k', linestyle='--', alpha=0.5)
#         plt.axvline(x=train_max, color='k', linestyle='--', alpha=0.5)
#
#         # Add training region shading
#         plt.axvspan(train_min, train_max, alpha=0.1, color='blue')
#
#         # Add title and labels
#         plt.title(f"{wave_type} - Training and Testing Data")
#         plt.xlabel("x")
#         plt.ylabel("Amplitude")
#         plt.legend()
#         plt.grid(True)
#
#     plt.tight_layout()
#     plt.savefig("train_test_data_visualization.png")
#     plt.show()
#
#     print("Wave function visualization complete!")
#     print("\nNew wave functions:")
#     print("1. compressing_expanding_wave - A wave that periodically compresses and expands like a sound wave")
#     print("2. increasing_decreasing_amp - A wave with amplitude that increases then decreases")
#     print("3. combined_compression_amp - A wave that combines both of the above effects")