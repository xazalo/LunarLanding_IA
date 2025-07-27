import numpy as np

def adjust_reward(points, last_points, reward, max_bonus=100):
    """
    Adjusts the base reward based on performance improvement between episodes.
    The adjustment is proportional to the percentage change in points,
    bounded within [-max_bonus, +max_bonus].

    Ajusta la recompensa base según la mejora de rendimiento entre episodios.
    El ajuste es proporcional al cambio porcentual en puntos,
    limitado entre [-max_bonus, +max_bonus].
    """
    # Handle division by zero in first episode
    # Manejar división por cero en el primer episodio
    if last_points == 0:
        return reward  # Return base reward without adjustment / Retornar recompensa base sin ajuste

    # Calculate percentage change / Calcular cambio porcentual
    percent_change = (points - last_points) / abs(last_points)

    # Limit percentage change to ±100% / Limitar cambio porcentual a ±100%
    percent_change = np.clip(percent_change, -1.0, 1.0)

    # Calculate bonus proportional to performance change
    # Calcular bonus proporcional al cambio de rendimiento
    bonus = percent_change * max_bonus

    # Apply adjustment to reward with base boost
    # Aplicar ajuste a recompensa con impulso base
    return reward + bonus + 100