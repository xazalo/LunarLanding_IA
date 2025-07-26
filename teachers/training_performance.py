import numpy as np

def training_performance(points, last_points, epsilon, scale=5.0):
    """
    Calculates a reward based on performance improvement.
    Calcula una recompensa basada en la mejora del rendimiento.

    Args:
        points (float): Current episode reward.
        last_points (float): Previous episode reward.
        epsilon (float): Current epsilon value.
        scale (float): Scaling factor for sensitivity.

    Returns:
        float: Reward based on performance improvement.
    """
    delta = points - last_points
    scaled_delta = np.tanh(delta / 200.0)  # Normalized to [-1, 1]
    reward = scaled_delta * epsilon * scale
    return {
        'reward': reward
    }
