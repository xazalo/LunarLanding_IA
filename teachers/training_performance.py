import numpy as np

def training_performance(points, last_points, scale=100.0):
    """
    Calcula una recompensa basada en la mejora del rendimiento,
    normalizada en el rango [-100, 100].

    Args:
        points (float): Recompensa del episodio actual.
        last_points (float): Recompensa del episodio anterior.
        epsilon (float): Valor actual de epsilon.
        scale (float): Escala máxima de recompensa (por defecto 100).

    Returns:
        dict: Recompensa normalizada.
    """

    delta = points - last_points
    scaled_delta = np.tanh(delta / 1000.0)  # Normaliza entre [-1, 1]
    reward = scaled_delta * scale  # Escala a [-100, 100]
    return {
        'reward': reward
    }
