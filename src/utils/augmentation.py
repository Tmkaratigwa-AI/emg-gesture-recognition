import numpy as np
from typing import Optional, Union


def inject_noise(emg_data: np.ndarray, noise_std: float, random_state: Optional[int] = None) -> np.ndarray:
    """Inject Gaussian noise."""
    if not isinstance(emg_data, np.ndarray) or noise_std < 0:
        raise ValueError("Invalid inputs.")
    if random_state is not None:
        np.random.seed(random_state)
    noise = np.random.normal(0, noise_std, emg_data.shape)
    return emg_data + noise


def scale_amplitude(emg_data: np.ndarray, scaling_factor: Union[float, int]) -> np.ndarray:
    """Scale amplitude by a factor."""
    if not isinstance(emg_data, np.ndarray) or not isinstance(scaling_factor, (int, float)):
        raise ValueError("Invalid inputs.")
    return emg_data * scaling_factor
