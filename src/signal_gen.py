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

def gradually_increasing_frequency_fn(x: np.ndarray) -> np.ndarray:
    """
    Generate a wave with a frequency that slowly increases from low to high
    across the domain. This creates a non-repeating pattern that gradually
    becomes more compressed toward the right edge.
    
    FREQUENCY REDUCED FOR BETTER VISUALIZATION
    """
    x_min, x_max = np.min(x), np.max(x)
    position = (x - x_min) / (x_max - x_min)
    
    # REDUCED FREQUENCIES by a factor of 75 (5 × 15)
    base_freq = 0.133   # was 10.0, then 2.0, now 10.0/75
    max_freq = 2.0   # was 150.0, then 30.0, now 150.0/75
    
    freq_factor = 1 / (1 + np.exp(-10 * (position - 0.7)))
    instantaneous_freq = base_freq + (max_freq - base_freq) * freq_factor
    
    dx = np.mean(np.diff(x)) if len(x) > 1 else 1.0
    phase = np.cumsum(instantaneous_freq * dx)
    phase = phase - phase[0]
    
    signal = np.sin(2 * np.pi * phase)
    signal = np.clip(signal, -1.0, 1.0)
    
    return signal

def gradually_increasing_amplitude_fn(x: np.ndarray) -> np.ndarray:
    """
    Generate a wave with an amplitude that slowly increases from near zero to 1.0 
    at the furthest right point. This function creates a non-repeating pattern 
    that gradually expands to full amplitude.
    
    FREQUENCY REDUCED FOR BETTER VISUALIZATION
    """
    x_min, x_max = np.min(x), np.max(x)
    # Reduced carrier frequency by a factor of 6
    carrier_freq = 30.0  # was 180.0
    carrier = np.sin(2 * np.pi * carrier_freq * x / 60.0)
    
    position = (x - x_min) / (x_max - x_min)
    amplitude = 0.02 + 0.98 * (1 / (1 + np.exp(-10 * (position - 0.7))))
    
    signal = amplitude * carrier
    signal = np.clip(signal, -1.0, 1.0)
    
    return signal

def combined_freq_amp_modulation_fn(x: np.ndarray) -> np.ndarray:
    """
    Generate a wave that combines both gradually increasing frequency and amplitude.
    This function creates a signal that becomes both higher in frequency and amplitude
    from left to right.
    
    FREQUENCY REDUCED FOR BETTER VISUALIZATION
    """
    x_min, x_max = np.min(x), np.max(x)
    position = (x - x_min) / (x_max - x_min)
    
    # Frequency modulation parameters - REDUCED BY FACTOR OF 75 (5 × 15)
    base_freq = 0.133   # was 10.0, then 2.0, now 10.0/75
    max_freq = 2.0   # was 150.0, then 30.0, now 150.0/75
    freq_factor = 1 / (1 + np.exp(-10 * (position - 0.7)))
    instantaneous_freq = base_freq + (max_freq - base_freq) * freq_factor
    
    # Integrate frequency to get phase
    dx = np.mean(np.diff(x)) if len(x) > 1 else 1.0
    phase = np.cumsum(instantaneous_freq * dx)
    phase = phase - phase[0]
    
    # Amplitude modulation
    amplitude = 0.02 + 0.98 * (1 / (1 + np.exp(-10 * (position - 0.7))))
    
    signal = amplitude * np.sin(2 * np.pi * phase)
    signal = np.clip(signal, -1.0, 1.0)
    
    return signal


# PERIODIC_SPECS holds domain parameters and data functions for each signal type
PERIODIC_SPECS = {
    "sin": {
        "period": 6,
        "domain_train": lambda p, ns: np.linspace(-10*p, 10*p, ns),
        "domain_test":  lambda p, ns: np.linspace(-25*p, 25*p, ns),
        "data_fn": lambda t: np.sin(t),
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
        "data_fn": lambda t: np.mod(t, 5),
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
    "gradually_increasing_frequency": {
        "period": 5,  # Larger period for more room to show frequency increase
        "domain_train": lambda p, ns: np.linspace(-10*p, 10*p, ns),  # Asymmetric range
        "domain_test":  lambda p, ns: np.linspace(-25*p, 25*p, ns), # Extend further right for testing
        "data_fn":      gradually_increasing_frequency_fn,
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
    "gradually_increasing_amplitude": {
        "period": 5,  # Larger period to show the full amplitude increase
        "domain_train": lambda p, ns: np.linspace(-10*p, 10*p, ns),  # Asymmetric range
        "domain_test":  lambda p, ns: np.linspace(-25*p, 25*p, ns), # Extend further right for testing
        "data_fn":      gradually_increasing_amplitude_fn,
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
    "combined_freq_amp_modulation": {
        "period": 5,  # Match other gradually increasing functions
        "domain_train": lambda p, ns: np.linspace(-10*p, 10*p, ns),  # Asymmetric range
        "domain_test":  lambda p, ns: np.linspace(-25*p, 25*p, ns), # Extend further right for testing
        "data_fn": combined_freq_amp_modulation_fn,
        "config": {
            "batchsize": 64,
            "numepoch": 5000,
            "printepoch": 50,
            "lr": 1e-4,
            "wd": 0.001,
            "y_uper": 1.0,
            "y_lower": -1.0,
        }
    }
}


def get_periodic_data(periodic_type, num_train_samples=None, num_test_samples=None):
    """
    Retrieve training and test data for a given periodic type.

    Returns:
      t_train, data_train, t_test, data_test, config, true_func
    """
    if periodic_type not in PERIODIC_SPECS:
        logger.error(f"Unknown periodic_type: {periodic_type}")
        raise ValueError(f"Unsupported periodic_type: {periodic_type}")

    spec = PERIODIC_SPECS[periodic_type]
    
    # Handle both old and new format specs
    if "data_params" in spec:
        # New format with separated parameter groups
        data_params = spec["data_params"]
        period = data_params["period"]
        domain_train = data_params["domain_train"]
        domain_test = data_params["domain_test"]
        
        # Merge training and visualization params for backward compatibility
        config = {}
        if "training_params" in spec:
            config.update(spec["training_params"])
        if "viz_params" in spec:
            config.update(spec["viz_params"])
            
        # Rename viz params to match old format
        if "y_upper" in config:
            config["y_uper"] = config.pop("y_upper")
    else:
        # Old format - direct access
        period = spec["period"]
        domain_train = spec["domain_train"]
        domain_test = spec["domain_test"]
        config = spec["config"]  # e.g., batchsize, etc.
    
    # Data function is always at the top level
    data_fn = spec["data_fn"]

    # Decide how many samples if none provided
    if num_train_samples is None:
        num_train_samples = int(10000 * period)
    if num_test_samples is None:
        num_test_samples = 4000
        
    # Use consistent domain handling for all function types
    # Scale domain based on period for all functions
    min_x_train, max_x_train = -10*period, 10*period
    
    t_train = np.linspace(min_x_train, max_x_train, num_train_samples)
    data_train = data_fn(t_train)
    
    # For test, extend beyond training domain in both directions
    min_x_test, max_x_test = -25*period, 25*period
    
    # Create test data with extra points in the extrapolation regions
    t_test = np.linspace(min_x_test, max_x_test, num_test_samples)
    data_test = data_fn(t_test)

    true_func = data_fn
    return t_train, data_train, t_test, data_test, config, true_func
