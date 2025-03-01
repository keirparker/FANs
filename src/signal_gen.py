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


##############################################################################
# PERIODIC_SPECS holds all domain & data logic for each type.
# 'config' is an optional sub-dict for any extra metadata (batchsize, etc.).
##############################################################################
PERIODIC_SPECS = {
    "sin": {
        "period": 6,
        "domain_train": lambda p, ns: np.linspace(-4*p*np.pi, 4*p*np.pi, ns),
        "domain_test":  lambda p, ns: np.linspace(-12*p*np.pi, 12*p*np.pi, ns),
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
        "domain_train": lambda p, ns: np.linspace(-4*p, 4*p, ns),
        "domain_test":  lambda p, ns: np.linspace(-12*p, 12*p, ns),
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
        "domain_train": lambda p, ns: np.linspace(-4*p, 4*p, ns),
        "domain_test":  lambda p, ns: np.linspace(-12*p, 12*p, ns),
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
        "domain_train": lambda p, ns: np.linspace(-4*p, 4*p, ns),
        "domain_test":  lambda p, ns: np.linspace(-12*p, 12*p, ns),
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
        "domain_train": lambda p, ns: np.linspace(-p, p, ns),
        "domain_test":  lambda p, ns: np.linspace(-2*p, 2*p, ns),
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
        "domain_train": lambda p, ns: np.linspace(-p, p, ns),
        "domain_test":  lambda p, ns: np.linspace(-2*p, 2*p, ns),
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
        "domain_train": lambda p, ns: np.linspace(-p, p, ns),
        "domain_test":  lambda p, ns: np.linspace(-2*p, 2*p, ns),
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
        "domain_train": lambda p, ns: np.linspace(-p, p, ns),
        "domain_test":  lambda p, ns: np.linspace(-2*p, 2*p, ns),
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

    # Generate training data
    t_train = domain_train(period, num_train_samples)
    data_train = data_fn(t_train)

    # Generate test data
    t_test = domain_test(period, num_test_samples)
    data_test = data_fn(t_test)

    # The 'true_func' is just data_fn if you want to use it as ground truth
    true_func = data_fn

    return t_train, data_train, t_test, data_test, config, true_func