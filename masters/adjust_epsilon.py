import numpy as np

def adjust_epsilon(epsilon, points, last_points,
                   min_epsilon=0.01, max_epsilon=1.0,
                   decay_base=0.01, increase_base=0.01,
                   sensitivity=500.0,):
    """
    Ajusta epsilon dependiendo del progreso y nivel de exploración actual.

    - Si epsilon > 0.5:
        * Mejora → bajar más rápido
        * Empeora → subir más lento
    - Si epsilon <= 0.5:
        * Mejora → bajar más lento
        * Empeora → subir más rápido
    - Si los puntos > 0: fijar epsilon en 0.2
    """

    # ⬇️ Lógica estándar de ajuste
    delta = points - last_points
    adjustment = np.tanh(delta / sensitivity)  # Normaliza a [-1, 1]

    if epsilon > 0.5:
        if delta > 0:
            # Baja rápido (poca exploración)
            epsilon -= (decay_base * 1) * adjustment * (epsilon / max_epsilon)
        else:
            # Sube lento
            epsilon += (increase_base * 0.5) * abs(adjustment) * ((max_epsilon - epsilon) / max_epsilon)
    else:
        if delta > 0:
            # Baja lento
            epsilon -= (decay_base * 1) * adjustment * (epsilon / max_epsilon)
        else:
            # Sube rápido
            epsilon += (increase_base * 0.5) * abs(adjustment) * ((max_epsilon - epsilon) / max_epsilon)

    # Asegura límites
    epsilon = max(min(epsilon, max_epsilon), min_epsilon)
    return round(epsilon, 3), points
