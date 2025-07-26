import numpy as np

def adjust_reward(points, last_points, max_bonus=100):
    """
    Calculates a bonus proportional to the percentage change in points between episodes,
    bounded within [-max_bonus, max_bonus].

    Calcula un bonus proporcional al cambio porcentual en los puntos entre episodios,
    acotado en [-max_bonus, max_bonus].

    Args:
        points (float): Puntos actuales
        last_points (float): Puntos del episodio anterior
        max_bonus (float): Valor máximo absoluto del bonus

    Returns:
        float: Bonus positivo o negativo
    """
    if last_points == 0:
        return 0.0  # Evitar división por cero en el primer episodio

    percent_change = (points - last_points) / abs(last_points)  # Cambio relativo
    percent_change = np.clip(percent_change, -1.0, 1.0)          # Limitar a ±100%

    bonus = percent_change * max_bonus
    return bonus
