import numpy as np

def get_hard_sphere_radius_from_gaussian(gaussian_radius: float) -> float:
    """
    Converts a Gaussian radius to a hard sphere radius.
    
    Args:
        gaussian_radius: The Gaussian radius.
        
    Returns:
        The equivalent hard sphere radius.
    """
    return gaussian_radius * np.power(3.0 * np.sqrt(np.pi) / 4.0, 1.0 / 3.0)
