import numpy as np

def adjust_reward(points, last_points, max_bonus=100):
    """
    Calculates a bonus proportional to the point change between episodes,
    bounded within [-max_bonus, max_bonus].
    
    Calcula un bonus proporcional al cambio en puntos entre episodios,
    acotado en [-max_bonus, max_bonus].

    Args/Argumentos:
        points (float): Current points / Puntos actuales
        last_points (float): Previous points / Puntos anteriores
        max_bonus (float): Maximum absolute bonus value / Valor máximo absoluto del bonus

    Returns/Retorna:
        float: Positive or negative bonus / Bonus positivo o negativo
    """
    # Calculate point difference / Calcular diferencia de puntos
    delta = points - last_points

    # Scale delta so ±300 change ≈ ±100 bonus
    # Escalar delta para que ±300 de cambio ≈ ±100 bonus
    normalized_delta = np.clip(delta / 300.0, -1.0, 1.0)

    # Calculate final bonus / Calcular bonus final
    bonus = normalized_delta * max_bonus
    
    return bonus