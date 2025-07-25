import numpy as np

def adjust_epsilon(epsilon, current_points, prev_points,
                   min_epsilon=0.01, max_epsilon=1.0,
                   decay_base=0.01, reward_influence=0.005):
    """
    Adjusts epsilon based on points change:
    - Decreases epsilon if improving (less exploration)
    - Increases epsilon if worsening (more exploration)
    - Keeps epsilon within given bounds

    Ajusta epsilon basado en el cambio de puntos:
    - Disminuye epsilon si mejora (menos exploración)
    - Aumenta epsilon si empeora (más exploración)
    - Mantiene epsilon dentro de los límites dados

    Args/Argumentos:
        epsilon (float): current epsilon value / valor actual de epsilon
        current_points (float): current points / puntos actuales
        prev_points (float): previous points / puntos anteriores
        min_epsilon (float): minimum epsilon / epsilon mínimo
        max_epsilon (float): maximum epsilon / epsilon máximo
        decay_base (float): base change / cambio base
        reward_influence (float): performance influence / influencia del rendimiento

    Returns/Retorna:
        float: new epsilon value / nuevo valor de epsilon
        float: current points (for next call) / puntos actuales (para siguiente llamada)
    """

    # Calculate point difference / Calcular diferencia de puntos
    point_delta = current_points - prev_points
    
    # Normalize effect using tanh (range -1 to 1)
    # Normalizar efecto usando tanh (rango -1 a 1)
    normalized_effect = np.tanh(point_delta / 100.0)

    if point_delta >= 0:
        # Improvement: decrease epsilon / Mejora: disminuir epsilon
        epsilon -= (decay_base + reward_influence * normalized_effect)
    else:
        # Worsening: increase epsilon / Empeora: aumentar epsilon
        epsilon += (decay_base - reward_influence * normalized_effect)

    # Keep epsilon within allowed range
    # Mantener epsilon dentro del rango permitido
    epsilon = max(min(epsilon, max_epsilon), min_epsilon)

    # Update previous points for next call
    # Actualizar puntos anteriores para siguiente llamada
    prev_points = current_points

    return round(epsilon, 3)  # Return rounded value / Retornar valor redondeado