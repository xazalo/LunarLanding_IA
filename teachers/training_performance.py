import numpy as np

def training_performance(points, last_points, scale=100.0):
    """
    Calculates a reward based on performance improvement, normalized in the range [-100, 100].
    / Calcula una recompensa basada en la mejora del rendimiento, normalizada en el rango [-100, 100].

    Args / Argumentos:
        points (float): Current episode reward / Recompensa del episodio actual.
        last_points (float): Previous episode reward / Recompensa del episodio anterior.
        scale (float): Maximum reward scale (default 100) / Escala máxima de recompensa (por defecto 100).

    Returns / Retorna:
        dict: Dictionary containing:
              - 'reward': Normalized performance reward (float)
              / Diccionario conteniendo:
              - 'reward': Recompensa de rendimiento normalizada (float)
    """
    # Calculate raw improvement / Calcular mejora bruta
    delta = points - last_points
    
    # Normalize using tanh to handle large variations
    # Normalizar usando tanh para manejar variaciones grandes
    scaled_delta = np.tanh(delta / 1000.0)  # Normalizes to [-1, 1] / Normaliza a [-1, 1]
    
    # Scale to desired range / Escalar al rango deseado
    reward = scaled_delta * scale
    
    return {
        'reward': reward
    }